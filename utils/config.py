import json
import os
import os.path as osp
import random
import time

from dotmap import DotMap
import numpy as np
import torch

from simulators import PyrepRenderer
from simulators import IsaacShrimpEnv
from pcpbo.pbbo import IdealArtificialUser, UncertainArtificialUser


def __gen_directories(cfg):
    if not osp.exists(cfg.logdir):
        dir_list = ['images', 'models', 'data']
        [os.makedirs(f'{cfg.logdir}/{each_dir}') for each_dir in dir_list]


def load_config(args, read_only=False):
    """

    :param args: (argparse.Namespace)
    :param read_only: (bool)
    :return:
    """

    default_cfg = BaseConfig().__dict__
    default_cfg.update(vars(args))
    cfg = DotMap(default_cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg, sim, renderer = load_task(cfg)
    cfg, user = load_artificial_user(cfg, sim)

    if not read_only:
        method = 'pcpbo' if cfg.pcpbo else 'pbo'
        cfg.logdir = f'logdir/{cfg.task}/{method}/{cfg.pref_weight}'
        cfg.logdir = f'{cfg.logdir}/{time.strftime("%Y%m%d%H%M%S", time.localtime())}'
        __gen_directories(cfg)

        with open(cfg.logdir + '/config.json', 'w') as f:
            json.dump(cfg, f, indent=2)
    return cfg, sim, renderer, user


def load_task(cfg):
    """

    :param cfg: (dotmap)
    :return:
    """
    if cfg.task == 'deep-fried_shrimp':
        cfg.var_u = [3e-2, 3e-2, 5e-1]
        cfg.obj_name_list = ['T', 'CA0', 'CA1', 'L', 'S0', 'S1', 'S2']
        cfg.num_dynamic_obj = 3
        cfg.dim_state = 7 * cfg.num_dynamic_obj
        cfg.dim_ctrl = 3
        cfg.dim_w = 2
        cfg.render_scene_name = 'simulators/assets/deep-fried_shrimp_env.ttt'
        sim = IsaacShrimpEnv(cfg)
        renderer = PyrepRenderer(cfg)
    elif cfg.task == 'simmered_taros':
        cfg.var_u = [1e-2, 1e-2, 1e-2]
        cfg.dim_w = 1
        # sim = IsaacTaroEnv(cfg)
        sim = NotImplementedError
        renderer = PyrepRenderer(cfg)
    elif cfg.task == 'tempura':
        cfg.var_u = [3e-2, 3e-2, 5e-1]
        cfg.dim_w = 2
        # sim = IsaacTempuraEnv(cfg)
        sim = NotImplementedError
        renderer = PyrepRenderer(cfg)
    else:
        raise NotImplementedError(cfg.task)

    return cfg, sim, renderer


def load_artificial_user(cfg, task):
    if cfg.user_model == DotMap():
        return cfg, None

    if cfg.user_model == 0:
        user = IdealArtificialUser(cfg, task)
    elif cfg.user_model == 1:
        user = UncertainArtificialUser(cfg, task, skip_range=task.less_unc_th)
        cfg.skip_range = task.less_unc_th
    elif cfg.user_model == 2:
        user = UncertainArtificialUser(cfg, task, skip_range=task.more_unc_th)
        cfg.skip_range = task.more_unc_th
    else:
        raise NotImplementedError(f'{cfg.user_model}')

    return cfg, user


class BaseConfig:
    def __init__(self):
        self.seed = int(str(time.time() * 1000).split('.')[0][-5:])
        # Kernel Hyper Parameters
        self.sigma = 1e-1
        self.eta = 1e-0
        self.ob_noise = 1e-2  # todo: chk this
        self.gt_eta = 1e-20  # todo: chk this
        # for PbBO
        self.hyp_lr = 1e-2
        self.post_lr = 1e-1
        self.num_opt = 20  # number of update loop (post. and hyper-param.)
        self.num_init_query = 1
        self.num_query = 50
        self.num_in_dim = 21
        # for CEM
        self.num_path = 400
        self.num_opt_cem = 10
        self.elite_num = 5
        self.smooth_mean = 0.5
        self.smooth_var = 0.8


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='deep-fried_shrimp')
    parser = parser.parse_args()

    load_config(parser)
