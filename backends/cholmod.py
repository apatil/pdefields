"""
A linear algebra backend that uses sparse, cholesky-based operations.
"""
__all__ = ['into_matrix_type', 'precision_to_products', 'pattern_to_products', 'rmvn', 'mvn_logp', 'axpy', 'dm_solve_m', 'm_mul_m', 'm_xtyx']

import numpy as np
import scipy
from scipy import sparse
from sparse_iterative import into_matrix_type, axpy, dm_solve_m, m_mul_m, m_xtyx
import scikits.sparse.cholmod as cholmod

precision_product_keys = ['L','P','F']

def pattern_to_products(q):
    return {'symbolic': cholmod.analyze(q)}

def precision_to_products(q, symbolic):
    F = symbolic.cholesky(q)
    L = F.L()
    P = symbolic.P()
    return {'L':L, 'P': P, 'F': F}

def rmvn(M,L,F,P):
    return M+F.solve_Lt(np.random.normal(size=len(M)))

def mvn_logp(x,M,L,F,P):
    """
     M is the mean vector
     L is the sparse Cholesky triangle
     F is the CHOLMOD factor
     P is the permutation vector
     """
    d = L.T*((x-M)[P])
    return -.5*np.dot(d,d) +np.sum(np.log(L.diagonal()))-.5*len(M)*np.log(2.*np.pi)

if __name__ == '__main__':
    OK = False
    while OK==False:
        try:
            B = np.random.normal(size=(100,100))
            ind = np.arange(B.shape[0])
            for i in xrange(B.shape[0]):
                np.random.shuffle(ind)
                B[i,ind[:B.shape[0]-5]]=0
    
            A = sparse.csc_matrix(np.dot(B,B.T))
            # A = sparse.csc_matrix(np.eye(100)*np.random.random(size=100))
            pattern_products = pattern_to_products(A)
            precision_products = precision_to_products(A, **pattern_products)
            OK = True
        except:
            pass
    m = np.random.normal(size=A.shape[0])
    x = np.random.normal(size=A.shape[0])    
    l1  = mvn_logp(x,m,**precision_products)
    import pymc as pm
    l2 = pm.mv_normal_like(x,m,A.todense())
    print l1,l2
    # # z = [rand(m,precision_products,1) for i in xrange(1000)]
    # # C_empirical = np.cov(np.array(z).T)