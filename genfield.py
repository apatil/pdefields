import numpy as np
from scipy.sparse import linalg

def genfield(op):
    "Op must be a scipy LinearOperator"
    return linalg.gmres(op, np.random.normal(size=op.shape[0]))[0].reshape(op.dims)
    
if __name__ == '__main__':
    import euclidean
    n = 11
    L = euclidean.fractional_modified_laplacian((n,), 1, 2)
    # import pylab as pl
    f = genfield(L)
    # pl.imshow(f,interpolation='nearest')