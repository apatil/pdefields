from scipy.sparse import linalg
import numpy as np

def expand_op_as_matrix(op):
    "Converts the linear operator to a dense matrix. For debugging only."
    out = np.empty((op.xsize, op.xsize))
    x = np.zeros(op.xsize)
    for i in xrange(op.xsize):
        x[i] = 1
        out[i] = op.matvec(x)
        x[i] = 0
    return out

def dim2w(dim):
    "A utility function for use with ffts."
    return np.arange(-np.ceil((dim-1)/2.), np.floor((dim-1)/2.)+1)

class fractional_modified_laplacian(linalg.LinearOperator):
    
    def __init__(self, dims, kappa, alpha):
        # Remember the size of the grid, kappa, and alpha.
        self.dims = dims
        self.xsize = np.prod(dims)
        
        wsq = dim2w(dims[0])**2
        for d in self.dims[1:]:
            wsq = np.add.outer(wsq, dim2w(d)**2)
        wsq = np.fft.ifftshift(wsq)
        self.freqtrans = (kappa**2+wsq)**(alpha/2.)
        
        # let the LinearOperator class do whatever it needs to do to initialize.
        linalg.LinearOperator.__init__(self, (self.xsize,self.xsize), None, dtype='float')
    
    def __call__(self, x):
        return self.matvec(x.ravel()).reshape(self.dims)
    
    def matvec(self, x):
        "Matvec computes Lx for given input x."
        # x is coming in as a vector, reshape it to an n-dimensional array.
        x_ = x.reshape(self.dims)
        
        # Apply an n-dimensional fft.
        k = np.fft.fftn(x_)
        
        # Apply the fractional modified Laplacian in Fourier space.
        # The fractional modified Laplacian is (\kappa^2 - \Delta)^{\alpha/2}
        k_trans = self.freqtrans*k

        # Apply an n-dimensional inverse fft, reshape to a vector and return.
        y_ = np.fft.ifftn(k_trans).real
        return y_.ravel()
        
def genfield(L):
    """
    Solves the system Lx = W using the gmres method, see 
    http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#scipy.sparse.linalg.gmres
    """
    # w = np.random.normal(size=L.dims)
    w = np.random.gamma(.01,.01,size=L.dims)
    import time
    t1 = time.time()
    result = linalg.lgmres(L, w.ravel())
    print "GMRES done in %ss, result code %i"%(time.time()-t1, result[1])
    return result[0].reshape(L.dims)

if __name__ == '__main__':
    import euclidean
    n = 1001
    L = euclidean.fractional_modified_laplacian((n,n), 1, 2)
    import pylab as pl
    pl.clf()
    
    f = genfield(L)
    pl.imshow(f,interpolation='nearest')
    pl.colorbar()