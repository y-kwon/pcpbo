import numpy as np
import torch

from pcpbo.pbbo import PreferenceBasedBayesianOptimization
from pcpbo.pbbo import RBFKernel
from pcpbo.pbbo import VariationalInference
from pcpbo.pbbo import RandomAcq, ThompsonSampling
from pcpbo.phy_opt import CrossEntropyMethod


class PhysicallyConsistentPreferentialBayesianOptimization:
    def __init__(self, cfg, task, renderer):
        self.task, self.renderer = task, renderer

        gp_kernel = RBFKernel(cfg,
                              sigma=np.repeat(np.log(cfg.sigma), len(cfg.cost_weight)),
                              eta=np.log(cfg.eta),
                              observation_noise=cfg.ob_noise,
                              learnable=not cfg.fixed_hp)
        self.pbbo = PreferenceBasedBayesianOptimization(cfg,
                                                        task=task,
                                                        kernel=gp_kernel,
                                                        approx=VariationalInference,
                                                        acq=ThompsonSampling if not cfg.random_acq else RandomAcq
                                                        )
        self.phy_opt = CrossEntropyMethod(cfg, task, renderer)

        self.pbbo.approx.load_lik(self.pbbo.pbbo_likelihood)
        self.x_for_pred = torch.from_numpy(self.pbbo.acq.gen_predict_x(num_in_dim=cfg.num_in_dim)).to(cfg.device)

    def terminate(self):
        self.task.terminate()
        self.renderer.terminate()
