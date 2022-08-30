import argparse

# Do not remove these
from isaacgym import gymapi
from isaacgym import gymtorch

from utils.config import load_config
from pcpbo import PhysicallyConsistentPreferentialBayesianOptimization
from pcpbo.pbbo import TrjBasedQuestionnaireSystem


def main(parser):
    """

    :param parser: (argparse.Namespace)
    """
    cfg, task, renderer, user = load_config(parser, read_only=False)
    pcpbo = PhysicallyConsistentPreferentialBayesianOptimization(cfg, task, renderer)
    qsys = TrjBasedQuestionnaireSystem(cfg, pcpbo)

    try:
        for i in range(cfg.num_query):
            print(f'No. {i + 1}')
            query = qsys.gen_query()
            answer, gt_answer = user.return_answer(query)
            qsys.get_answer(answer, gt_answer=gt_answer)
    except Exception as e:
        qsys.terminate(fl=False)
        raise e
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
    parser.pcpbo = True

    main(parser)
