import numpy as np
from pcpbo.phy_opt.Optimizer import TrjOptimizer


class CrossEntropyMethod(TrjOptimizer):
    def __init__(self, cfg, sim, renderer):
        super(CrossEntropyMethod, self).__init__(cfg, sim)

        self.renderer = renderer
        self.T, self.K = cfg.num_dynamic_obj, cfg.num_path
        self.var_u = np.diag(cfg.var_u)
        self.elite_num = cfg.elite_num
        self.smooth_mean, self.smooth_var = cfg.smooth_mean, cfg.smooth_var
        self.init_smooth_mean, self.init_smooth_var = self.smooth_mean, self.smooth_var
        self.init_var_u = self.var_u.copy()
        self.num_opt = cfg.num_opt_cem

    def __reset_smooth_param(self):
        self.smooth_mean, self.smooth_var = self.init_smooth_mean, self.init_smooth_var
        self.var_u = self.init_var_u

    def __optimize(self, cand_ctrl, trj_cost):
        # Select elite from candidate
        idx_cand = trj_cost[:, 0].argsort()[:self.elite_num]
        elite_mean = np.mean(cand_ctrl[idx_cand], axis=0)
        elite_var = np.mean((cand_ctrl[idx_cand] - elite_mean) ** 2, axis=0)

        # Update mean and variance (cand_ctrl[0]: noiseless ctrl)
        update_ctrl = cand_ctrl[0] * self.smooth_mean + elite_mean * (1 - self.smooth_mean)
        update_var = np.diag(self.var_u) * self.smooth_var + np.mean(elite_var, 0) * (1 - self.smooth_var)
        self.var_u = np.diag(update_var)
        return update_ctrl

    def run_optimizer(self, ctrl_series, weight):
        ctrl = ctrl_series.copy()
        for up in range(self.num_opt):
            noise = np.random.randn(self.K, self.T, self.dim_ctrl) @ self.var_u
            noise[0] *= 0  # MPPI's heuristic
            # todo: Impl. truncated normal version
            cand_ctrl = self.sim.gen_cand(ctrl, noise)
            trj = self.sim.ctrl_to_trj(cand_ctrl)
            trj_cost = self.sim.sum_cost_fcn(trj, weight)
            ctrl = self.__optimize(cand_ctrl, trj_cost)
        opt_trj = self.sim.ctrl_to_trj(np.tile(ctrl.copy()[None, ...], (self.K, 1, 1)))[0]
        return ctrl, opt_trj

    def weight_to_trj(self, weight_list):
        trjs, imgs, ctrls, costs = [], [], [], []
        for each_weight in weight_list:
            ctrl_series = self.sim.gen_near_optimal_ctrl_series(each_weight)
            self.__reset_smooth_param()

            opt_ctrl_series, trj = self.run_optimizer(ctrl_series, each_weight)
            img = self.renderer.gen_images(trj[None, :].copy())
            opt_cost = self.sim.sum_cost_fcn(trj[None, :], each_weight)
            costs.append(opt_cost)
            trjs.append(trj)
            imgs.append(img)
            ctrls.append(opt_ctrl_series)
        return np.array(trjs), np.array(imgs), np.array(ctrls), np.array(costs)
