import numpy as np
import torch
from torch.distributions import Normal


class AcquisitionFunction:
    def __init__(self, cfg, task):
        self.task = task
        self.device = cfg.device
        self.range_w = [0, 1]

    def choose_next_points(self, num_points):
        raise NotImplementedError()

    def gen_predict_x(self, num_in_dim):
        xmin, xmax = np.array([0]).repeat(self.task.dim_w), np.array([1]).repeat(self.task.dim_w)
        if self.task.dim_w == 1:
            return np.array(np.round(np.linspace(xmin, xmax, num_in_dim), 3))
        elif self.task.dim_w == 2:
            x_list = np.array([np.round(np.linspace(exmin, exmax, num_in_dim), 3)
                               for exmin, exmax in zip(xmin, xmax)])
            XX, YY = np.meshgrid(x_list[0], x_list[1])
            return np.concatenate([XX.flatten()[:, None], YY.flatten()[:, None]], axis=1)
        elif self.task.dim_w == 3:
            x_list = np.array(np.round(np.linspace(xmin, xmax, num_in_dim), 3))
            mesh_grid = np.meshgrid(x_list[:, 0], x_list[:, 1], x_list[:, 2])
            mesh = np.concatenate([each_mesh.flatten()[:, None] for each_mesh in mesh_grid], axis=1)
            return mesh
        else:
            raise NotImplementedError()


class RandomAcq(AcquisitionFunction):
    def __init__(self, cfg, task):
        super(RandomAcq, self).__init__(cfg, task)
        self.prob_len = np.array(self.range_w[1] - self.range_w[0], dtype=np.float64)
        self.prob_mean = np.array(self.prob_len / 2. + self.range_w[0], dtype=np.float64)

    def choose_next_points(self, num_points=1):
        def gen_next_points():
            return np.array(
                [(np.random.rand(num_points) - 0.5) * self.prob_len[i] + self.prob_mean[i] for
                 i in range(self.task.task.dim_w)],
                dtype=np.float64).T

        next_xi, next_xj = gen_next_points(), gen_next_points()
        return next_xi, next_xj


class ThompsonSampling(AcquisitionFunction):
    def __init__(self, cfg, task):
        super(ThompsonSampling, self).__init__(cfg, task)
        self.num_in_dim = cfg.num_in_dim
        self.cand_x = self.gen_predict_x(num_in_dim=self.num_in_dim)
        self.torch_cand_x = torch.tensor(self.cand_x, dtype=torch.float64, device=cfg.device)
        self.normal = Normal(torch.tensor([0.0], device=cfg.device), torch.tensor([1.0], device=cfg.device))

    def choose_next_points(self, pbbo, num_duels=1, random=False, vis_path=True):
        if random:
            if self.num_in_dim < self.task.task.dim_w * 2:
                raise NotImplementedError()
            cand_in_dim = np.round(np.linspace(self.range_w[0], self.range_w[1], self.num_in_dim), 3)
            next_xi, next_xj = np.random.choice(cand_in_dim, size=(2, self.task.task.dim_w), replace=False)
            return (next_xi, next_xj), None
        else:
            if num_duels > 1:
                raise NotImplementedError(f'Number of duels > 1 is not available.')
            duel_pos, path_idx = [], []
            while len(path_idx) < 2:
                cand_paths = pbbo.gen_sample_path(self.torch_cand_x, num_samples=4)
                cand_min_xs = self.cand_x[torch.argmin(cand_paths, 1).cpu().detach().numpy()]
                for idx, cand_min_x in enumerate(cand_min_xs):
                    if len(np.unique(cand_min_x)) == len(cand_min_x):
                        duel_pos.append(cand_min_x)
                        path_idx.append(idx)
                    if len(path_idx) == 2:
                        break
            first_path, second_path = cand_paths[path_idx]

            duel_pos = np.array(duel_pos)
            next_xi, next_xj = duel_pos[::2], duel_pos[1::2]
            if vis_path:
                return (next_xi, next_xj), torch.cat([first_path[None, ...], second_path[None, ...]], 0)
            else:
                return next_xi, next_xj
