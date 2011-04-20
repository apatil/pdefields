"""This is the only interface to the linear algebra backend. 
If you redefine any of these functions, it will be used throughout pdefields."""

import numpy as np
import scipy
from scipy import linalg

def sqrtm_from_diags(tridiag):
    return scipy.linalg.cholesky_banded(tridiag, overwrite_ab=False,lower=True)

m_mul_v = np.dot

def norm(x):
    return np.sqrt(np.dot(x,x))

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

def m_xtyx(x,y):
    # FIXME: Sparse
    return np.dot(np.dot(x.T,y),x)

def v_xtyx(x,y):
    # FIXME: Sparse
    return m_xtyx(x,y)
    
def m_mul_m(x,y):
    # FIXME: Sparse
    return np.dot(x,y)

def m_solve_m(x,y):
    # FIXME: Sparse
    return np.linalg.solve(x,y)
    
def m_solve_v(x,y):
    # FIXME: Sparse
    return np.linalg.solve(x,y)

def m_mul_v(x,y):
    # FIXME: Sparse
    return np.dot(x,y)

def matrix_from_diag(d):
    # FIXME: Sparse
    return np.eye(d.shape[0])*d
    
def log_determinant(x):
    # FIXME: sparse
    # Can the method of Bai et al. be used?
    return np.log(np.linalg.det(x))

def diagonalize_conserving_rowsums(x):
    new_diag = m_mul_v(x, np.ones(x.shape[0]))
    return matrix_from_diag(new_diag)