import numpy as np
import pickle as pkl


class DataStorage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logdir = cfg.logdir
        self.num_total_query = cfg.num_init_query + cfg.num_query

        self.w_count = 0
        self.answers, self.gt_answers = [], []
        # todo: None -> []
        self.wi, self.wj, self.trji, self.trjj, self.imgi, self.imgj = None, None, None, None, None, None
        self.optw_list = []
        self.pred_post_dict = {'x': [], 'mean': [], 'std': []}

    def append_query(self, weights, trjs, imgs):
        wi, wj = weights
        if type(wi) == float:
            wi, wj = np.array(wi), np.array(wj)
        self.w_count += 1
        self.wi, self.wj = self.__chk_and_concat(self.wi, self.wj, wi, wj)

        if trjs is not None:
            trji, trjj = trjs
            if type(trji) == float:
                trji, trjj = np.array(trji), np.array(trjj)
            self.trji, self.trjj = self.__chk_and_concat(self.trji, self.trjj, trji, trjj)
        if imgs is not None:
            imgi, imgj = imgs
            self.imgi, self.imgj = self.__chk_and_concat(self.imgi, self.imgj, imgi, imgj)

    def append_answer(self, answer, gt_answer):
        self.answers.append(answer)
        self.gt_answers.append(gt_answer)

    @staticmethod
    def __chk_and_concat(a, b, new_a, new_b):
        if a is not None:
            a = np.concatenate([a, new_a[None, :]], axis=0)
            b = np.concatenate([b, new_b[None, :]], axis=0)
        else:
            a, b = new_a[None, :], new_b[None, :]
        return a, b

    def log_opt_weight(self, opt_weight):
        self.optw_list.append(opt_weight.cpu().detach().numpy())

    def log_pred_post(self, x, mean, cov):
        self.pred_post_dict['x'].append(x.cpu().detach().numpy())
        self.pred_post_dict['mean'].append(mean.cpu().detach().numpy())
        self.pred_post_dict['std'].append(cov.diag().sqrt().cpu().detach().numpy())

    def save_all_data(self):
        np.save(f'{self.logdir}/data/query.npy', np.concatenate([self.wi, self.wj], axis=-1))
        np.save(f'{self.logdir}/data/answer.npy', np.array(self.answers))
        if self.trji is not None:
            np.save(f'{self.logdir}/data/trj.npy', np.concatenate([self.trji, self.trjj], axis=-1))
        if self.imgi is not None:
            np.save(f'{self.logdir}/data/img.npy', np.concatenate([self.imgi, self.imgj], axis=-1))
        np.save(f'{self.logdir}/data/gt_answer.npy', np.array(self.gt_answers))
        np.savetxt(f'{self.logdir}/data/opt_weight_list.csv', self.optw_list, delimiter=',')
        np.savez(f'{self.logdir}/data/pred_post_dict.npz', **self.pred_post_dict)
