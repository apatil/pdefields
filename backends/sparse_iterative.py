"""
A linear algebra backend that uses sparse, iterative operations.
"""


__all__ = ['into_matrix_type', 'precision_to_products' 'rmvn', 'mvn_logp', 'axpy', 'dm_solve_m', 'm_mul_m', 'm_xtyx']

import numpy as np
import scipy
from scipy import sparse
import warnings

def into_matrix_type(m):
    warnings.warn('sparse_iterative is not working yet.')
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
    Port of Matlab code provided by Daniel Simpson. Approximates Cholesky(A^{-1})*z.
    
    z is the vector of i.i.d. standard normals
    A is the precision matrix
    m is the size of the krylov subspace
    """
    # FIXME: Doesn't work.
    T,V = lanczos(A,z,m)
    S = sqrtm_from_diags(T)
    e = np.hstack([1,np.zeros(m-2)])
    return norm(z)*np.dot(V[:,:m-1],scipy.linalg.solve_banded((1,0), S, e))

def rmvn(M, Q, ldq, m=200):
    return M+krylov_product_Simpson(Q, np.random.normal(size=Q.shape[0]), m)

def m_xtyx(x,y):
    return x.T*y*x

v_xtyx = m_xtyx

def m_mul_m(x,y):
    return x*y
    
def m_add_m(x,y):
    return x+y

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

def prec_gibbs(m,q,conditional_obs_prec):
    return prec_mvn(m,q+conditional_obs_prec)
    
def precision_to_products(Q):
    return {'Q': Q, 'ldq': log_determinant(Q)}
    
def mvn_logp(x,M,Q,ldq):
    x_ = (x-M)
    return -(len(M)/2.)*np.log(2.*pi)-.5*ldq - .5*v_xtyx(x_,Q)