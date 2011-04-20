"""
Functionality to perform common operations on large multivariate normal variables parameterized by precision.
Agnostic to linear algebra backend.
"""

import pymc
import numpy as np
import linalg

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
