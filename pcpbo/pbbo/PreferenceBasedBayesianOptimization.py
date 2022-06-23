import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MultiNormal
from torch.optim import LBFGS

from pcpbo.pbbo.GaussianProcess import GaussianProcess


class PreferenceBasedBayesianOptimization:
    def __init__(self, cfg, task, kernel, approx, acq):
        """

        :param cfg:
        :param task:
        :param kernel: GP's kernel
        :param approx: Approx. Method (e.g., VI)
        :param acq: BO's Acquisition Function (e.g., ThompsonSampling)
        """
        self.device = cfg.device
        self.gp = GaussianProcess(kernel=kernel)
        self.approx = approx(cfg, self.gp)
        self.acq = acq(cfg, task)

        self.init_kern_sigma = cfg.sigma
        self.init_kern_eta = cfg.eta
        self.hyp_lr = cfg.hyp_lr

    @property
    def sqrt_two(self):
        return torch.sqrt(torch.tensor(2, dtype=torch.float64, device=self.device))

    def predict(self, x, covariance=True):
        mean = self.approx.predict_mean(self.X, x)
        cov = self.approx.predict_cov(self.X, x) if covariance else None
        return mean, cov

    def gen_sample_path(self, x, num_samples=1):
        mean, cov = self.predict(x)
        # todo: Temporary settings (validate_args = False). We need speed.
        path = MultiNormal(mean, cov, validate_args=False).sample((num_samples,))
        return path

    def update_post_and_hyper(self, update_num=100, hyper=True):
        self.approx.vfe = []
        for _ in range(update_num):
            if hyper:
                self.update_hyper_param()
            self.update_post_param()
        return self.approx.vfe

    def update_hyper_param(self):
        self.hyp_opt = LBFGS(self.gp.kernel.hyper_params_dict.values(), lr=self.hyp_lr, max_iter=1)

        def closure():
            self.approx.target_opt = self.hyp_opt
            vfe = self.approx.calc_vfe()
            return vfe

        self.hyp_opt.step(closure)
        self.update_gram()
        self.hyp_opt.zero_grad()

    def calc_cross_entropy(self):
        mean, _ = self.predict(self.X, covariance=False)
        pred_y = self.pbbo_likelihood(mean[:int(len(self.X) / 2)], mean[int(len(self.X) / 2):])
        # pred_y[pred_y == 0], pred_y[pred_y == 1] = 1e-12, 1 - 1e-12
        pred_y[pred_y == 0] = 1e-12
        # class label 'y' is always "0"
        cross_entropy = -torch.mean(torch.log(pred_y))
        return float(cross_entropy.cpu().detach())

    def update_post_param(self):
        self.approx.approx_posterior(self.pbbo_likelihood)

    def reset_hyper_params(self):
        self.gp.kernel.sigma.data.fill_((torch.tensor(self.init_kern_sigma, dtype=torch.float64).log()))
        self.gp.kernel.eta.data.fill_((torch.tensor(self.init_kern_eta, dtype=torch.float64).log()))

    def reset_approx_params(self):
        self.approx.mean_param.data.fill_(0.)
        self.approx.sigma_param.data.fill_(1.)

    def reset_each_param(self):
        self.reset_approx_params()
        self.reset_hyper_params()

    def pbbo_likelihood(self, fi, fj):
        zij = (fj - fi) / (self.sqrt_two * torch.exp(self.gp.kernel.eta))
        return Normal(0, 1).cdf(zij)

    def update_dataset(self, xi, xj):
        self.X = torch.cat([torch.from_numpy(xi).to(self.device), torch.from_numpy(xj).to(self.device)], dim=0)
        self.update_gram(self.X)

    def update_gram(self, X=None):
        if X is not None:
            self.gp.update_gram(X)
        else:
            self.gp.update_gram(self.X)
