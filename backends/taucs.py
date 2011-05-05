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
import ctypes
from ctypes import cdll, util

# This should be the path to the Matrix dll provided by R's Matrix library.
matrix_so_path = 'libtaucs.so'
taucs = cdll.LoadLibrary(matrix_so_path)

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
