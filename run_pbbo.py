import argparse
import numpy as np

# Do not remove these
from isaacgym import gymapi
from isaacgym import gymtorch

from utils.config import load_config
from pcpbo.pbbo import PreferenceBasedBayesianOptimization
from pcpbo.pbbo import RBFKernel, VariationalInference, ThompsonSampling, RandomAcq
from pcpbo.pbbo import NaiveQuestionnaireSystem
from utils.storage import DataStorage


def main(parser):
    """

    :param parser: (argparse.Namespace)
    """
    cfg, task, _, user = load_config(parser, read_only=False)
    pbbo = PreferenceBasedBayesianOptimization(cfg,
                                               task=task,
                                               kernel=RBFKernel(cfg,
                                                                sigma=np.repeat(np.log(cfg.sigma),
                                                                                len(cfg.pref_weight)),
                                                                eta=np.log(cfg.eta),
                                                                observation_noise=cfg.ob_noise,
                                                                learnable=not cfg.fixed_hp),
                                               approx=VariationalInference,
                                               acq=ThompsonSampling if not cfg.random_acq else RandomAcq,
                                               storage=DataStorage(cfg)
                                               )
    qsys = NaiveQuestionnaireSystem(cfg, pbbo)

    for i in range(cfg.num_query):
        query = qsys.gen_query()
        answer, gt_answer = user.return_answer(query)
        qsys.get_answer(answer, gt_answer=gt_answer)
        print(f'No. {i + 1}')
    qsys.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--pref_weight', type=float, nargs='+', required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--random_acq', action='store_true')
    parser.add_argument('--fixed_hp', action='store_true', help='Hyper parameters are fixed')
    parser.add_argument('--user_model', type=int, help="0: ideal, 1: less unc, 2: more unc", required=True)
    parser.add_argument('--skip', action='store_true', help="query skip is available")
    parser.add_argument('--synth', action='store_true', help="query synthesis is available")
    parser.add_argument('--debug', action='store_true')
    parser = parser.parse_args()
    parser.pbbo = True

    main(parser)
