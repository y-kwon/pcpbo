import numpy as np


class BaseEnv:
    def __init__(self, cfg):
        self.dim_ctrl = cfg.dim_ctrl
        self.dim_state = cfg.dim_state
        self.dim_w = cfg.dim_w

        self.pref_weight = np.array([cfg.cost_weight])

    def step(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError()

    def ctrl_to_trj(self, *args, **kwargs):
        raise NotImplementedError()

    def clip_ctrl(self, ctrl):
        raise NotImplementedError()

    def gen_cand(self, ctrl, noise_seq):
        """

        :param ctrl: control input
        :param noise_seq: noise sequence for control input
        :return:
            cand_ctrl:
        """
        cand_ctrl = ctrl + noise_seq
        cand_ctrl = self.clip_ctrl(cand_ctrl)
        return cand_ctrl

    def sum_cost_fcn(self, trj, weight):
        """

        :param trj: state trajectory
        :param weight: preference weight
        :return:
        """
        raise NotImplementedError()

    def gen_near_optimal_ctrl_series(self, weight):
        raise NotImplementedError()
