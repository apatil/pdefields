"""
A linear algebra backend that uses sparse, cholesky-based operations.
"""

# OK. To build SuiteSparse's Cholmod on a Mac,
# add -fPIC to CFLAGS and F77FLAGS in UFconfig/UFconfig.mk in SuiteSparse.
# Then make.
# Then go to CHOLMOD/Lib and do 
# gcc -shared -Wl -o libcholmod.so *.o -lcblas -llapack -L/opt/local/lib -lmetis -L../../COLAMD/Lib -lcolamd -L../../CCOLAMD/Lib -lccolamd -L../../CAMD/Lib -lcamd -L../../AMD/Lib -lamd

__all__ = ['into_matrix_type', 'precision_to_products' 'rmvn', 'mvn_logp', 'axpy', 'dm_solve_m', 'm_mul_m', 'm_xtyx']

import numpy as np
import scipy
from scipy import sparse
from sparse_iterative import into_matrix_type, axpy, dm_solve_m, m_mul_m, m_xtyx
import cvxopt
from cvxopt import cholmod


# import ctypes
# from ctypes import cdll, util
# # This should be the path to the Matrix dll provided by R's Matrix library.
# matrix_so_path = '/Library/Frameworks/R.framework/Versions/2.12/Resources/library/Matrix/libs/x86_64/Matrix.so'
# cholmod = cdll.LoadLibrary(matrix_so_path)

def precision_to_products(Q):
    return {'S': scikits.sparse.cholmod.cholesky(Q)}
    
def rmvn(M,S):
    "M is a vector, S is a CHOLMOD Factor from scikits.sparse"
    y = np.random.normal(size=len(M))
    return S.solve_L(y)
    
def mvn_logp(x,M,S):
    "M is a vector, x is a vector, S is a CHOLMOD Factor from scikits.sparse"
    d = x-M
    d_ = d[S.P()]
    L = S.L()
    ldet = np.sum(np.log(L.diagonal()))
    d_L = L.T * d_
    return -.5*(len(M)*np.log(2.*np.pi)) - .5*ldet -.5*np.dot(d_L,d_L)

def get_nonzero(m):
    nz = m.nonzero()
    nz_ = [nz_.astype('int') for nz_ in nz]
    return nz_

def scipy_to_cvxopt(m, nz=None):
    nz = nz or get_nonzero(m)
    return cvxopt.spmatrix(m[nz].todense(), *nz)


if __name__ == '__main__':
    
    A = scipy_to_cvxopt(sparse.lil_matrix(np.eye(100)*5))
    # Symbolic factorization
    F = cvxopt.cholmod.symbolic(A)
    cvxopt.cholmod.numeric(A,F)
    L = cvxopt.cholmod.getfactor(F)