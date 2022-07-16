import numpy as np


class ArtificialUser:
    def __init__(self, cfg, task):
        self.task = task
        self.pref_weight = cfg.pref_weight

    def pref_fcn(self, trj):
        return self.task.pref_fcn(trj, self.pref_weight)

    def return_answer(self, left_trj, right_trj):
        raise NotImplementedError()


class IdealArtificialUser(ArtificialUser):
    def __init__(self, cfg, task):
        super(IdealArtificialUser, self).__init__(cfg, task)

    def return_answer(self, left_trj, right_trj):
        left_dis, right_dis = self.pref_fcn(left_trj[None, ...]), self.pref_fcn(right_trj[None, ...])
        return int(right_dis < left_dis), int(right_dis < left_dis)


class UncertainArtificialUser(ArtificialUser):
    def __init__(self, cfg, task, skip_range):
        super(UncertainArtificialUser, self).__init__(cfg, task)
        self.skip_range = skip_range
        self.skip_fl = cfg.skip

    def return_answer(self, left_trj, right_trj):
        left_dis, right_dis = self.pref_fcn(left_trj[None, ...]), self.pref_fcn(right_trj[None, ...])
        gt_answer = int(right_dis < left_dis)
        min_dis = np.min([left_dis, right_dis])

        if min_dis > self.skip_range[1]:
            if self.skip_fl:
                return -1, gt_answer
            else:
                return np.random.randint(0, 2), gt_answer
        elif min_dis > self.skip_range[0]:
            norm_dis = (min_dis - self.skip_range[0]) / (self.skip_range[1] - self.skip_range[0])
            if norm_dis > np.random.rand():
                if self.skip_fl:
                    return -1, gt_answer
                else:
                    return np.random.randint(0, 2), gt_answer
            else:
                return gt_answer, gt_answer
        else:
            return gt_answer, gt_answer
