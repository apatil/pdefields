# -*- coding: UTF-8 -*-

"High-level interface to multivariate normal variables. Retargetable linear algebra backend."
import numpy as np
import pymc as pm
import mcmc
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

class GMRFMetropolis(pm.StepMethod):
    def __init__(self, x, likelihood_code, M, Q, likelihood_variables, n_sweeps):
        """
        Takes the following arguments:
        - x: a SparseMVN instance.
        - likelihood_code: A Fortran code snippet for evaluating the likelihoods. 
          See the documentation of mcmc.compile_metropolis_sweep.
        - M: A mean vector or a PyMC variable valued as one.
        - Q: A precision matrix, in SciPy CSR or CSC format, or a PyMC variable valued as one.
        - likelihood_variables: All the vertex-specific variables needed to compute the likelihoods.
          Must be a (len(x), _) array or a PyMC variable valued as one.
        - n_sweeps: The number of compiled Metropolis sweeps to do per step.
        """

        if len(x.extended_children-set(children_list))>0:
            raise ValueError, "Children_list must contain all of %s's extended children."%self.x
            
        if pm.utils.value(Q).__class__ not in [sparse.csc.csc_matrix, sparse.csc.csr_matrix]:
            raise ValueError, "The value of Q must be a SciPy CSC or CSR matrix."
        self.x = x
        self.M = M
        self.Q = Q
        self.likelihood_variables = likelihood_variables
        self.n_sweeps = n_sweeps
        self.compiled_metropolis_sweep = mcmc.compile_metropolis_sweep(likelihood_code)
    
    def step(self):
            self.S.value, _ = mcmc.fast_metropolis_sweep(pm.utils.value(self.M),
                                        pm.utils.value(self.Q),
                                        self.compiled_metropolis_sweep,
                                        self.x.value,
                                        pm.utils.value(self.likelihood_variables),
                                        n_sweeps=self.n_sweeps)