import numpy as np
import torch


# License of MATLAB ver. [1]
#
# Copyright (c) 2013, John D'Errico
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Numpy Version: https://github.com/alan-turing-institute/bocpdms/blob/master/nearestPD.py


class NearestSymmetricPositiveDefinite:
    def __init__(self, device):
        """
            The nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
        Args:
            device (str): device name which have a graph
        """
        self.device = torch.device(device)

    def find_pd(self, matrix):
        """
            Find the nearest positive-definite matrix to matrix.
            A Python/Pytorch version of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].
            [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
            matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        b = (matrix + matrix.T) / 2
        _, s, v = torch.svd(b)

        h = v.T @ (torch.diag(s) @ v)
        mat_hat = (b + h) / 2
        mat_hat = (mat_hat + mat_hat.T) / 2

        if self.check_pd(mat_hat):
            return mat_hat

        # todo: Do we need to get rid of Numpy?
        np_spacing = np.spacing(mat_hat.norm().cpu().detach().numpy())
        spacing = torch.tensor(np_spacing, device=self.device)
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrices of small dimension, be on
        # other order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        i = torch.eye(len(s), device=self.device)
        k = torch.tensor(1, device=self.device)

        while not self.check_pd(mat_hat):
            mat_hat_eig, _ = torch.eig(mat_hat, eigenvectors=False)
            min_real_eig = torch.min(mat_hat_eig[:, 0])
            mat_hat += i * (-min_real_eig * k ** 2 + spacing)
        return mat_hat

    @staticmethod
    def check_pd(matrix):
        """Returns True when matrix is positive-definite, using Cholesky decomposition"""
        try:
            torch.linalg.cholesky(matrix)
            return True
        except RuntimeError:
            return False


if __name__ == '__main__':
    dev_name = 'cuda:0'
    mat = torch.randn(2, 2, dtype=torch.float64).to(torch.device(dev_name))

    pd_cl = NearestSymmetricPositiveDefinite(dev_name)
    print(pd_cl.find_pd(mat).dtype)
