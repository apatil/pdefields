# -*- coding: UTF-8 -*-

"High-level interface to Gaussian Markov random fields and associated fitting algorithms. Retargetable linear algebra backend."

import numpy as np
import pymc as pm
import algorithms
from scipy import sparse

def mvn_logp(x,M,precision_products,backend):
    "Takes a candidate value x, a mean vector M, the products returned by backend.precision_products as a map, and the linear algebra backend module. Passes the arguments to backend.mvn_logp and returns the log-probability of x given M and the precision matrix represented in precision_products."
    if precision_products is None:
        return -np.inf
    else:
        return backend.mvn_logp(x,M,**precision_products)
    
def rmvn(M,precision_products,backend):
    "Takes a mean vector M, the products returned by backend.precision_products as a map, and the linear algebra backend module. Passes the arguments to backend.rmvn and returns a random draw x from the multivariate normal variable with mean M and the precision matrix represented in precision_products."
    return backend.rmvn(M,**precision_products)

SparseMVN = pm.stochastic_from_dist('SparseMVN', mvn_logp, rmvn, mv=True)

class GMRFGibbs(pm.StepMethod):
    def __init__(self, backend, x, obs, M, Q,  Q_obs, L_obs=None, K_obs=None, pattern_products=None):
        """
        Applies to the following conjugate submodel:
        x ~ N(M,Q^{-1})
        obs ~ N(L_obs x + K_obs, Q_obs^{-1})
        
        Takes the following arguments:
        - backend: A linear algebra backend, for example CHOLMOD
        - x: a SparseMVN instance
        - obs: A normal variable whose mean is a linear transformation of x.
        - M: The prior mean of M.
        - Q: The prior precision of M.
        - Q_obs: The precision of obs conditional on x.
        - L_obs: A sparse matrix, optional.
        - K_obs: a vector, optional.
        - pattern_products: If the sparsity pattern of the conditional precision of x on obs can be computed ahead of time, the backend's pattern-specific computations can be passed in to save some work.
        """
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
        
        if pattern_products is None:
            @pm.deterministic(trace=False)
            def pattern_products(Qc=Qc,backend=backend):
                "The backend-specific computations that can be got from just the sparsity pattern, e.g. the symbolic Cholesky factorization."
                return backend.pattern_to_products(Qc)
        
        @pm.deterministic(trace=False)
        def M_and_precision_products(obs=obs, M=M, Qc=Qc, Q_obs=Q_obs, L_obs=L_obs, K_obs=K_obs, pp=pattern_products, backend=backend):
            "The conditional mean and the backend-specific computations from the actual precision matrix."
            return backend.conditional_mean_and_precision_products(obs, M, Qc, Q_obs, L_obs=L_obs, K_obs=K_obs, **pp)
        
        self.Qc = Qc
        self.pattern_products = pattern_products
        self.M_and_precision_products = M_and_precision_products
        
        pm.StepMethod.__init__(self, [x])
        
    def step(self):
        v = self.M_and_precision_products.value
        self.x.value = self.backend.rmvn(v[0], **v[1])

class GMRFMetropolis(pm.StepMethod):
    def __init__(self, x, likelihood_code, M, Q, likelihood_variables, n_sweeps, double_check_likelihood=False):
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
        - double_check_likelihood: Whether to double-check that the log-likelihood is finite after making the jump.
        """

        self.x = x
        self.M = M
        self.Q = Q
        self.likelihood_variables = likelihood_variables
        self.n_sweeps = n_sweeps
        self.compiled_likelihood_evaluation = algorithms.compile_likelihood_evaluation(likelihood_code)
        self.compiled_metropolis_sweep = algorithms.compile_metropolis_sweep(likelihood_code)
        self.double_check_likelihood = double_check_likelihood
        
        pm.StepMethod.__init__(self, [x])
    
    def step(self):
            self.x.value = algorithms.fast_metropolis_sweep(pm.utils.value(self.M),
                                        pm.utils.value(self.Q),
                                        self.compiled_metropolis_sweep,
                                        self.compiled_likelihood_evaluation,
                                        self.x.value,
                                        pm.utils.value(self.likelihood_variables),
                                        n_sweeps=self.n_sweeps)
            
            if self.double_check_likelihood:
                # Guard against inconsistent jumps.                            
                try:
                    self.loglike
                except pm.ZeroProbability:
                    self.x.revert()
                    
def wrap_metropolis_for_INLA(metro_class):
    """
    Wraps Metropolis step methods so they can handle extended parents of
    Gaussian processes.
    """
    class wrapper(metro_class):
        def __init__(self, stochastic, likelihood_deriv1, likelihood_deriv2, tol, pattern_products, *args, **kwds):
            
            self.likelihood_deriv1 = likelihood_deriv1
            self.likelihood_deriv2 = likelihood_deriv2
            self.tol = tol
            self.pattern_products = pattern_products
            self.metro_class.__init__(self, stochastic, *args, **kwds)
            gmrfs = []
            
            for c in list(self.children):
                if isinstance(c, SparseMVN):
                    gmrfs.append(c)
            if len(gmrfs)>1:
                raise NotImplementedError, 'Only one GMRF allowed so far. Why not combine them?'
            
            self.mb_for_logp = set(self.markov_blanket) - set(gmrfs) - set(gmrfs[0].extended_children)
            
            self.gmrf = gmrfs[0]
            
            @pm.deterministic
            def approx_conditional(M = self.gmrf.parents['M'], precprod = self.gmrf.parents['precision_products'], tol=self.tol, backend=self.gmrf.parents['backend'], patprod=self.pattern_products, d1=self.likelihood_deriv1, d2=self.likelihood_deriv2):
                vals, vars, Mc, ppc = algorithms.scoring_gaussian_full_conditional(M,precprod['Q'],patprod,d1,d2,backend,tol)
                return Mc, ppc
            self.approx_conditional = approx_conditional
        
        def get_evidence(self):
            # Approximates log p(self.gmrf.extended_children | self.stochastics) p(self.stochastics)
            pygx = pm.logp_of_set(self.gmrf.extended_children)
            px = self.gmrf.logp
            Mc, ppc = self.approx_conditional.value
            pxgy = self.gmrf.parents['backend'].mvn_logp(self.gmrf.value, Mc, **ppc)
            return pygx + px - pxgy
        evidence = property(get_evidence)    
            
        def get_logp_plus_loglike(self):
            return pm.logp_of_set(self.mb_for_logp) + self.get_evidence()
        logp_plus_loglike = property(get_logp_plus_loglike)
    
        def step(self):
            self.metro_class.step(self)
            # Then draws a value for the gmrf from its approximate conditional posterior.
            Mc, ppc = self.approx_conditional.value
            self.gmrf.value = self.gmrf.parents['backend'].rmvn(Mc, **ppc)
                    
        @staticmethod
        def competence(stochastic, metro_class=metro_class):
            # Opt-in only.
            return 0
        
    wrapper.__name__ = 'InlaParent%s'%metro_class.__name__
    wrapper.metro_class = metro_class
    wrapper.__doc__ = """A modified version of class %s that handles parents of GMRFs, and GMRFs, using INLA.'
The extra arguments required may be flat values or PyMC variables:
    - likelihood_deriv1 and likelihood_deriv2: functions returning the first two derivatives of the likelihood given the GMRF's value.
    - tol: Tolerance for the scoring method.
    - pattern_products: Whatever the backend can precompute given only the sparsity pattern of the precision matrix.
    
    
Docstring of class %s: \n\n%s"""%(metro_class.__name__,metro_class.__name__,metro_class.__doc__)
            
    return wrapper