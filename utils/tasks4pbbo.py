import numpy as np


class Sin4PBBO:
    def __init__(self, cfg):
        self.dim_w = 2
        self.pref_weight = np.array(cfg.pref_weight)

    def sum_cost_fcn(self, w, pref_w):
        return 100 * np.sum((w - pref_w) ** 2, 1) ** 0.5
