from scipy.sparse.linalg import LinearOperator
import numpy as np
import map_utils

class fractional_modified_laplacian(LinearOperator):
    def __init__(self, dims, kappa, alpha):
        self.dims = dims
        self.kappa = kappa
        self.alpha = alpha
        self.xsize = np.prod(dims)
        self.len = np.prod(dims)
        LinearOperator.__init__(self, (self.len,self.len), None, dtype='float')
    def matvec(self, x):
        x_ = x.reshape(self.dims)
        k = np.fft.fftn(x_)
        ktk = (k**2)
        y_ = np.fft.ifftn((self.kappa**2+ktk)**(self.alpha/2.)).real
        return y_.reshape(x.shape)