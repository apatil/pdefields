"""
This is the only interface to the linear algebra backend. 
If you redefine any of these functions, it will be used throughout pdefields.
The required interface is the __all__ list.
"""

__all__ = ['into_matrix_type', 'rmvn', 'm_xtyx', 'v_xtyx', 'm_mul_m', 'm_mul_v', 'dm_solve_m', 'm_solve_v', 'log_determinant']

import numpy as np
import scipy
from scipy import sparse

def into_matrix_type(m):
    return sparse.csr.csr_matrix(m)

def sqrtm_from_diags(tridiag):
    return scipy.linalg.cholesky_banded(tridiag, overwrite_ab=False,lower=True)

m_mul_v = np.dot

def norm(x):
    "Returns |x|"
    return np.sqrt(np.dot(x,x))

def axpy(a,x,y):
    "Returns ax+y"
    return a*x + y

def extract_diagonal(x):
    return x.diagonal()

def lanczos(A,z,m):
    V = np.empty((len(z), m))
    alpha = np.zeros(m)
    beta = np.zeros(m+1)

    nz = norm(z)

    V[:,0] = z/nz
    for k in xrange(1,m):
        V[:,k] = m_mul_v(A,V[:,k-1])
        if k > 1:
            V[:,k] -= beta[k]*V[:,k-2]
        alpha[k] = np.dot(V[:,k], V[:,k-1])
        V[:,k] -= alpha[k] * V[:,k-1]
        beta[k+1] = norm(V[:,k])
        V[:,k] /= beta[k+1]
        
    T = np.zeros((2,m-1))
    T[0,:] = alpha[1:]
    T[1,:-1] = beta[2:-1]
    return T, V

def krylov_product_Simpson(A,z,m):
    """
    Port of Matlab code provided by Daniel Simpson.
    r is the vector of i.i.d. standard normals
    A is the precision matrix
    m is the size of the krylov subspace
    """
    # FIXME: Doesn't work.
    T,V = lanczos(A,z,m)
    S = sqrtm_from_diags(T)
    e = np.hstack([1,np.zeros(m-2)])
    return norm(z)*np.dot(V[:,:m-1],scipy.linalg.solve_banded((1,0), S, e))

def rmvn(Q, m=200):
    return krylov_product_Simpson(Q, np.random.normal(size=Q.shape[0]), m)

def m_xtyx(x,y):
    return x.T*y*x

v_xtyx = m_xtyx

def m_mul_m(x,y):
    return x*y

def dm_solve_m(x,y):
    "A^{-1} B, where A is diagonal and B is CSR."
    out = y.copy()
    nzi = y.nonzero()
    for i in xrange(len(nzi[0])):
        i_ = nzi[0][i]
        j_ = nzi[1][i]
        out[i_, j_] /= x[i_,i_]
    return out

def m_solve_v(x,y,symm=False):
    if symm:
        sparse.linalg.cg(x,y)
    else:
        return sparse.linalg.gmres(x,y)

def m_mul_v(x,y):
    return x*y
        
def log_determinant(x):
    # FIXME: This is wrong and slow.
    # Can the method of Bai et al. be used?
    w,v = sparse.linalg.eigs(x, k=x.shape[0]-2)
    return np.log(w.real).sum()