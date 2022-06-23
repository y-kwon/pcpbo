class TrjOptimizer:
    def __init__(self, cfg, sim):
        self.dim_state, self.dim_ctrl = cfg.dim_state, cfg.dim_ctrl
        self.sim = sim

    def run_optimizer(self, ctrl_series, weight):
        """

        :param ctrl_series:
        :param weight:
        :return:
        """
        raise NotImplementedError()

    def weight_to_trj(self, weight):
        """

        :param weight:
        :return:
        """
        raise NotImplementedError()
