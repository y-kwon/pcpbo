import argparse
import numpy as np
import matplotlib.pyplot as plt

# Do not remove these
from isaacgym import gymapi
from isaacgym import gymtorch

from config import load_config
from pcpbo.phy_opt import CrossEntropyMethod


def main(parser):
    """

    :param parser: (argparse.Namespace)
    """
    cfg, sim, renderer, _ = load_config(parser, read_only=False)
    cem = CrossEntropyMethod(cfg, sim, renderer)

    weight = np.array([cfg.pref_weight])
    trjs, imgs, ctrls, costs = cem.weight_to_trj(weight)

    plt.imsave(f'{cfg.logdir}/images/test.png', imgs[0])

    sim.terminate()
    renderer.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--pref_weight', type=float, nargs='+', required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--random_acq', action='store_true')
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser = parser.parse_args()

    main(parser)
