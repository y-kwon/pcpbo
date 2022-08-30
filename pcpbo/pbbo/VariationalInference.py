import numpy as np
import torch
from torch.optim import LBFGS
from torch.distributions.multivariate_normal import MultivariateNormal as mvnrand

from pcpbo.pbbo.NearestSymmetricPositiveDefinite import NearestSymmetricPositiveDefinite


class ApproxMethod:
    def __init__(self, cfg):
        self.device = torch.device(cfg.device)

    def predict_mean(self, *args, **kwargs):
        raise NotImplementedError()

    def predict_cov(self, *args, **kwargs):
        raise NotImplementedError()

    def approx_posterior(self, *args, **kwargs):
        raise NotImplementedError()


class VariationalInference(ApproxMethod):
    def __init__(self, cfg, gp, lik):
        super(VariationalInference, self).__init__(cfg)

        self.gp = gp
        self.lik = lik
        self.npd = NearestSymmetricPositiveDefinite(cfg.device)
        self.post_lr = cfg.post_lr
        self.num_reparam = cfg.num_reparam

        # Todo: we should set appropriate N
        self.N = 2 * 4 * (cfg.num_init_query + cfg.num_query)
        self.mean_param = torch.zeros(self.N, dtype=torch.float64, requires_grad=True, device=self.device)
        self.sigma_param = torch.ones(self.N, dtype=torch.float64, requires_grad=True, device=self.device)
        self.post_mean, self.post_cov = None, None
        self.eye_mat = torch.eye(self.N, device=self.device, dtype=torch.float64)

        self.vfe = []

    def predict_mean(self, train_x, pred_x):
        # todo: impl. prediction before learning
        with torch.no_grad():
            ks = self.gp.kernel.gen_gram_matrix(train_x, pred_x)
            mean = ks.T @ self.mean_param[:int(self.gp.K.shape[0])]
        return mean

    def predict_cov(self, train_x, pred_x):
        # todo: impl. prediction before learning
        with torch.no_grad():
            n = int(self.gp.K.shape[0])
            ks = self.gp.kernel.gen_gram_matrix(train_x, pred_x)
            kss = self.gp.kernel.gen_gram_matrix(pred_x, pred_x)
            sqrt_sig = torch.diag(torch.exp(self.sigma_param[:n]).sqrt())
            B = torch.eye(n, dtype=torch.float64, device=self.device) + (sqrt_sig @ self.gp.K @ sqrt_sig)
            L = self.get_cholesky(B)
            linv = torch.linalg.solve(L, torch.eye(n, device=self.device, dtype=torch.float64))
            V = linv @ (sqrt_sig @ ks)
            cov = kss - (V.T @ V)
        return cov

    def closure(self):
        self.target_opt = self.post_opt
        vfe = self.calc_vfe()
        return vfe

    @staticmethod
    def get_cholesky(mat):
        try:
            chol = torch.linalg.cholesky(mat)
        except RuntimeError:
            print(torch.diag(mat))
            raise RuntimeError()
        return chol

    def calc_vfe(self):
        # self.target_opt.zero_grad()
        lik, K = self.lik, self.gp.K

        if torch.linalg.matrix_rank(K) != K.shape[0]:
            # print(f'rank deficient (K: {torch.matrix_rank(K)} <- {K.shape[0]})')
            # print(self.gp.kernel.hyper_params_dict)
            raise NotImplementedError()
            # K = self.npd.return_pd_for_K(K)

        n, d = int(K.shape[0]), int(K.shape[0] / 2)
        mean_param, sigma_param = self.mean_param[:n], self.sigma_param[:n]
        eye_mat = self.eye_mat[:n, :n]

        sqrt_sig = sigma_param.exp().sqrt().diag()
        B = eye_mat + sqrt_sig @ K @ sqrt_sig

        L = self.get_cholesky(B)
        linv, ltinv = torch.linalg.solve(L, eye_mat), torch.linalg.solve(L.T, eye_mat)
        a = eye_mat - sqrt_sig @ ltinv @ linv @ sqrt_sig @ K

        self.post_mean = K @ mean_param
        self.post_cov = K @ a

        if not self.npd.check_pd(self.post_cov):
            print(f'Post Covariance is not PD!!! ({torch.matrix_rank(K)} <- {K.shape[0]})')
            self.post_cov = self.npd.find_pd(self.post_cov)

        sum_log_lik = 0
        for j in range(d):
            try:
                samples = mvnrand(self.post_mean[j::d], self.post_cov[j::d, j::d]
                                  ).rsample((self.num_reparam,)).to(self.device)
            except ValueError as e:
                print(self.post_cov[j::d, j::d])
                print(self.sigma_param)
                raise ValueError(e)

            # todo: Can we remove the parameter for numerical stability? ( 1e-12 )
            likelihood = self.lik(samples[:, 0], samples[:, 1])
            likelihood[likelihood == 0.] += 1e-12

            log_lik = -torch.log(likelihood)
            sum_log_lik += torch.mean(log_lik)
            if torch.isinf(sum_log_lik):
                raise ValueError('some likelihood values are 0.')

        vfe = sum_log_lik + 0.5 * (torch.trace(a) + mean_param.T @ K @ mean_param + torch.logdet(B) - n)

        self.vfe.append(vfe.cpu().detach().numpy())
        self.target_opt.zero_grad()
        vfe.backward(retain_graph=True)
        return vfe

    def approx_posterior(self, lik):
        self.post_opt = LBFGS([self.mean_param, self.sigma_param], lr=self.post_lr, max_iter=1)
        self.post_opt.step(self.closure)
        self.post_opt.zero_grad()
        return self.vfe
