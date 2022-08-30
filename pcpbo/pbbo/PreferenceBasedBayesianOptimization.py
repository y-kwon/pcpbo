import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MultiNormal
from torch.optim import LBFGS

from pcpbo.pbbo.GaussianProcess import GaussianProcess


class PreferenceBasedBayesianOptimization:
    def __init__(self, cfg, task, kernel, approx, acq, storage):
        """

        :param cfg:
        :param task:
        :param kernel: GP's kernel
        :param approx: Approx. Method (e.g., VI)
        :param acq: BO's Acquisition Function (e.g., ThompsonSampling)
        """
        self.device = cfg.device
        self.gp = GaussianProcess(kernel=kernel)
        self.approx = approx(cfg, self.gp, self.pbbo_likelihood)
        self.acq = acq(cfg, task)
        self.storage = storage
        self.fixed_hp = cfg.fixed_hp
        self.synth_fl = cfg.synth
        self.num_opt = cfg.num_opt
        self.num_init_query = cfg.num_init_query
        self.num_valid_answer = 0

        self.init_kern_sigma = cfg.sigma
        self.init_kern_eta = cfg.eta
        self.hyp_lr = cfg.hyp_lr
        self.X = None
        self.x_for_pred = torch.from_numpy(self.acq.gen_predict_x(num_in_dim=cfg.num_in_dim)).to(cfg.device)

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

    def gen_next_duel(self, save_query=True):
        weights, _ = self.acq.choose_next_points(self, random=self.num_init_query > self.num_valid_answer)
        weights = np.array(weights).squeeze()
        if save_query:
            self.storage.append_query(weights, None, None)
        return weights

    def get_answer(self, answer, gt_answer):
        self.storage.append_answer(answer, gt_answer)
        duel = np.array([self.storage.wi[-1], self.storage.wj[-1]])

        if answer != -1:
            self.num_valid_answer += 1
            self.update_dataset(duel[answer], duel[int(not answer)])

            if self.num_valid_answer == 1 and np.sum(np.array(self.storage.answers) == -1):
                if self.synth_fl:
                    for e in np.concatenate([self.storage.wi[:-1], self.storage.wj[:-1]], 0):
                        self.update_dataset(duel[answer], e)

                # predict and save optimal weight
                pred_mean, pred_cov = self.predict(self.x_for_pred)
                for _ in self.storage.wi[:-1]:
                    self.storage.log_opt_weight(self.x_for_pred[torch.argmin(pred_mean)])
                    self.storage.log_pred_post(self.x_for_pred, pred_mean, pred_cov)

        else:
            if self.synth_fl and self.num_valid_answer:
                np_X = self.X.cpu().detach().numpy().copy()
                if len(np_X) > 2:
                    uniq_np_X = np.unique(np_X[:int(len(np_X) / 2)], axis=0)
                    past_winner = uniq_np_X[np.random.randint(0, len(uniq_np_X))]
                else:
                    past_winner = np_X[:int(len(np_X) / 2)]
                for e in duel:
                    self.update_dataset(past_winner, e)

        if self.num_valid_answer:
            self.update_post_and_hyper(update_num=self.num_opt, hyper=not self.fixed_hp)

            # predict and save optimal weight
            pred_mean, pred_cov = self.predict(self.x_for_pred)
            self.storage.log_opt_weight(self.x_for_pred[torch.argmin(pred_mean)])
            self.storage.log_pred_post(self.x_for_pred, pred_mean, pred_cov)

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

    def update_dataset(self, winner, loser):
        duel = torch.from_numpy(np.array([[winner], [loser]])).to(self.device)
        if self.X is None:
            self.X = duel[:, 0]
        else:
            self.X = torch.cat((self.X[:int(len(self.X) / 2)], duel[0], self.X[int(len(self.X) / 2):], duel[1]), 0)
        self.update_gram(self.X)

    def update_gram(self, X=None):
        if X is not None:
            self.gp.update_gram(X)
        else:
            self.gp.update_gram(self.X)

    def terminate(self, fl):
        self.storage.save_all_data()
