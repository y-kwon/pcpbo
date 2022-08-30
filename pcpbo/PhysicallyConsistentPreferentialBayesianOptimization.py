import numpy as np
import torch

from pcpbo.pbbo import PreferenceBasedBayesianOptimization
from pcpbo.pbbo import RBFKernel
from pcpbo.pbbo import VariationalInference
from pcpbo.pbbo import RandomAcq, ThompsonSampling
from pcpbo.phy_opt import CrossEntropyMethod
from utils.storage import DataStorage


class PhysicallyConsistentPreferentialBayesianOptimization:
    def __init__(self, cfg, task, renderer):
        self.task, self.renderer = task, renderer
        self.num_opt = cfg.num_opt
        self.fixed_hp = cfg.fixed_hp

        self.storage = DataStorage(cfg)

        gp_kernel = RBFKernel(cfg,
                              sigma=np.repeat(np.log(cfg.sigma), len(cfg.pref_weight)),
                              eta=np.log(cfg.eta),
                              observation_noise=cfg.ob_noise,
                              learnable=not cfg.fixed_hp)
        self.pbbo = PreferenceBasedBayesianOptimization(cfg,
                                                        task=task,
                                                        kernel=gp_kernel,
                                                        approx=VariationalInference,
                                                        acq=ThompsonSampling if not cfg.random_acq else RandomAcq,
                                                        storage=self.storage)
        self.phy_opt = CrossEntropyMethod(cfg, task, renderer)

        self.get_answer = self.pbbo.get_answer

    def gen_query(self):
        # generate new weights
        weights = self.pbbo.gen_next_duel(save_query=False)

        # run Physical simulation-based optimization
        trjs, imgs, ctrls, costs = self.phy_opt.weight_to_trj(weights)
        self.storage.append_query(weights, trjs, imgs)

        return (trjs[0].copy(), trjs[1].copy()), (imgs[0].copy(), imgs[1].copy())

    def terminate(self, fl=True):
        self.task.terminate()
        self.renderer.terminate()
        if fl:
            self.storage.save_all_data()
