import numpy as np
import torch


class GaussianProcess:
    def __init__(self, kernel):
        self.kernel = kernel
        self.K = None

    def update_gram(self, X):
        self.K = self.kernel.gen_gram_matrix(X, X)


class Kernel:
    def __init__(self, cfg, eta, learnable=False, observation_noise=0.):
        """

        Args:
            cfg (dotmap): config
            eta (float or int): Variance of observation noise (log(eta) is actual variance)
            learnable (bool, optional): eta is fixed or not
            observation_noise(float):
        """
        self.eta = torch.tensor(float(eta), dtype=torch.float64, requires_grad=learnable, device=cfg.device)
        self.hyper_params_dict = {'eta': self.eta}
        self.observation_noise = observation_noise
        self.device = torch.device(cfg.device)

    def gen_gram_matrix(self, xi, xj):
        """

        Args:
            xi (torch.tensor): shape [N, dim]
            xj (torch.tensor): shape [M, dim]

        Returns:
            gram_matrix (torch.tensor): shape [N, M]
                if N == M, gram matrix is added observation noise.

        """
        if xi.shape[-1] != xj.shape[-1]:
            raise ValueError('Each dimension is different ({} != {})'.format(xi.shape[-1], xj.shape[-1]))
        if not (xi.ndim == xj.ndim == 2):
            raise ValueError('Dimension != 2')

        N, M = xi.shape[0], xj.shape[0]
        if self.observation_noise and N == M:
            return self.kernel_func(xi, xj) + self.observation_noise * torch.eye(N, dtype=torch.float64,
                                                                                 device=self.device)
        else:
            return self.kernel_func(xi, xj)

    def kernel_func(self, *args):
        raise NotImplementedError()


class RBFKernel(Kernel):
    def __init__(self, cfg, sigma, eta, learnable=False, observation_noise=0.):
        """

        Args:
            sigma (float or ndarray): Length scale of RBF (log(sigma) is actual value)
            eta (float or int): Variance of observation noise (log(eta) is actual variance)
            learnable (bool, optional) : sigma and eta are fixed or not
        """
        super(RBFKernel, self).__init__(cfg, eta, learnable, observation_noise)
        if sigma.ndim == 0:
            self.sigma = torch.tensor(float(sigma), dtype=torch.float64, requires_grad=learnable, device=cfg.device)
        else:
            self.sigma = torch.tensor(np.array(sigma), dtype=torch.float64, requires_grad=learnable, device=cfg.device)
        self.hyper_params_dict['sigma'] = self.sigma
        self.kernel_func = self.kernel_func_ond_dim if self.sigma.ndim == 0 else self.kernel_func_mul_dim

    def kernel_func_ond_dim(self, xi, xj):
        """

        Args:
            xi (torch.tensor): shape [N, 1]
            xj (torch.tensor): shape [M, 1]

        Returns:
            gram_matrix (torch.tensor): shape [N, M]
        """
        N, M = xi.shape[0], xj.shape[0]
        mat_xi, mat_xj = xi[:, None, :].repeat(1, M, 1), xj[None, :, :].repeat(N, 1, 1)
        krn_value = torch.exp(-torch.sum((mat_xi - mat_xj) ** 2, dim=-1) / self.sigma.exp())
        return krn_value

    def kernel_func_mul_dim(self, xi, xj):
        """

        Args:
            xi (torch.tensor): shape [N, dim] (dim > 1)
            xj (torch.tensor): shape [M, dim] (dim > 1)

        Returns:
            gram_matrix (torch.tensor): shape [N, M]
        """
        N, M = xi.shape[0], xj.shape[0]
        mat_xi, mat_xj = xi[:, None, :].repeat(1, M, 1), xj[None, :, :].repeat(N, 1, 1)
        krn_value = torch.exp(-torch.sum(((mat_xi - mat_xj) ** 2) / self.sigma[None, None, :].exp(), dim=-1))
        return krn_value
