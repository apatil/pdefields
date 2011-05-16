# -*- coding: UTF-8 -*-

"High-level interface to multivariate normal variables. Retargetable linear algebra backend."
import numpy as np
import pymc as pm
import algorithms
from scipy import sparse

# TODO: Conditional versions of all.

def mvn_logp(x,M,precision_products,backend):
    "Takes a candidate value x, a mean vector M, the products returned by backend.precision_products as a map, and the linear algebra backend module. Passes the arguments to backend.mvn_logp and returns the log-probability of x given M and the precision matrix represented in precision_products."
    return backend.mvn_logp(x,M,**precision_products)
    
def rmvn(M,precision_products,backend):
    "Takes a mean vector M, the products returned by backend.precision_products as a map, and the linear algebra backend module. Passes the arguments to backend.rmvn and returns a random draw x from the multivariate normal variable with mean M and the precision matrix represented in precision_products."
    return backend.rmvn(M,**precision_products)

def eta(M, precision_products, backend):
    "Takes a mean vector M, the products returned by backend.precision_products as a map, and the linear algebra backend module. Passes the arguments to backend.eta and returns the corresponding 'canonical mean' η=Q^{-1}M, where Q is the precision matrix."
    return backend.eta(M,**precision_products)

SparseMVN = pm.stochastic_from_dist('SparseMVN', mvn_logp, rmvn, mv=True)

class GMRFGibbs(pm.StepMethod):
    def __init__(self, backend, x, obs, M, Q,  Q_obs, L_obs=None, K_obs=None):
        self.x = x
        self.backend = backend
        self.obs = obs
        self.M = M
        self.Q = Q
        self.Q_obs = Q_obs
        self.L_obs = L_obs
        self.K_obs = K_obs
        
        @pm.deterministic(trace=False)
        def Qc(Q=Q,Q_obs=Q_obs,L_obs=L_obs):
            "The conditional precision matrix."
            return algorithms.conditional_precision(Q,Q_obs,L_obs)
        
        @pm.deterministic(trace=False)
        def pattern_products(Qc=Qc,backend=backend):
            "The backend-specific computations that can be got from just the sparsity pattern, e.g. the symbolic Cholesky factorization."
            return backend.pattern_to_products(Qc)
        
        @pm.deterministic(trace=False)
        def M_and_precision_products(obs=obs, M=M, Qc=Qc, Q_obs=Q_obs, L_obs=L_obs, K_obs=K_obs, pp=pattern_products, backend=backend):
            return backend.conditional_mean_and_precision_products(obs, M, Qc, Q_obs, L_obs, K_obs, **pp)
                
        self.Qc = Qc
        self.pattern_products = pattern_products
        self.M_and_precision_products = M_and_precision_products
        
        pm.StepMethod.__init__(self, [x])
        
    def step(self):
        v = self.M_and_precision_products.value
        self.x.value = self.backend.rmvn(v[0], **v[1])

class GMRFMetropolis(pm.StepMethod):
    def __init__(self, x, likelihood_code, M, Q, likelihood_variables, n_sweeps):
        """
        Takes the following arguments:
        - x: a SparseMVN instance.
        - likelihood_code: A Fortran code snippet for evaluating the likelihoods. 
          See the documentation of algorithms.compile_metropolis_sweep.
        - M: A mean vector or a PyMC variable valued as one.
        - Q: A precision matrix, in SciPy CSR or CSC format, or a PyMC variable valued as one.
        - likelihood_variables: All the vertex-specific variables needed to compute the likelihoods.
          Must be a (len(x), _) array or a PyMC variable valued as one.
        - n_sweeps: The number of compiled Metropolis sweeps to do per step.
        """

        self.x = x
        self.M = M
        self.Q = Q
        self.likelihood_variables = likelihood_variables
        self.n_sweeps = n_sweeps
        self.compiled_metropolis_sweep = algorithms.compile_metropolis_sweep(likelihood_code)
        
        pm.StepMethod.__init__(self, [x])
    
    def step(self):
            self.x.value = algorithms.fast_metropolis_sweep(pm.utils.value(self.M),
                                        pm.utils.value(self.Q),
                                        self.compiled_metropolis_sweep,
                                        self.x.value,
                                        pm.utils.value(self.likelihood_variables),
                                        n_sweeps=self.n_sweeps)