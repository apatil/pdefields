import pymc
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

def prec_rmvn(m,q,lqdet,n=None):
    n = n or min(len(m), 2000)
    return m+krylov_product_Simpson(q, np.random.normal(size=len(m)), n)
    
def prec_mvn_logp(x,m,q,lqdet):
    # FIXME: sparse
    "Blah."
    return -q.shape[0]/2 * (np.log(2*np.pi)+lqdet) - v_xtyx((x-m),q)/2
    
def prec_gibbs(m,q,conditional_obs_prec):
    return prec_mvn(m,q+conditional_obs_prec)

SparsePrecMVN = pymc.distributions.stochastic_from_dist('sparse_prec_mvn', prec_mvn_logp, prec_rmvn, mv=True)

class SparsePrecMVNStep(pymc.StepMethod):

    def __init__(self, stochastic, observation, obs_precision):
        self.stochastic = stochastic
        self.observation = observation
        self.obs_precision = obs_precision
    
    def step(self):
        m = pymc.utils.value(stochastic.parents['m'])
        q = pymc.utils.value(stochastic.parents['q'])
        conditional_precision = q + pm.utils.value(self.obs_precision)
        delta = pm.utils.value(self.observation) - self.stochastic.value
        # Slow step.
        conditional_mean = m_solve_v(q,m_mul_v(conditional_precision, delta))
        self.stochastic.value = prec_rmvn(conditional_mean, conditional_precision, 0)
    
def prec_propose_element_from_conditional_prior(x,m,q,i):
    raise NotImplementedError
    
if __name__ == '__main__':
    import pymc as pm
    x = np.linspace(-1,1,101)
    A = pm.gp.cov_funs.matern.euclidean(x,x,amp=1,scale=1,diff_degree=.5)
    r = np.random.normal(size=A.shape[0])
    m = 101
    y = krylov_product_Simpson(np.linalg.inv(A),r,m)
    import pylab as pl
    pl.clf()
    pl.plot(y)
