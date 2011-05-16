# -*- coding: UTF-8 -*-
"""
This module contains 'generic' MCMC algorithms, generic meaning that they work on any SciPy CSR or CSC precision matrix without requiring specialized linear algebra found in the backends. This class does not include Gibbs steps, which require whole matrix operations. However, it does include one-vertex-at-a-time Metropolis steps.
"""

import hashlib
from numpy import f2py
import pdefields
import os
import numpy as np
import scipy
from scipy import sparse

# Conditional precision.
def conditional_precision(Q,Q_obs,L_obs):
    """
    Returns the conditional precision of x in the conjugate submodel
    
    x ~ N(M,Q^{-1})
    y ~ N(L_obs x + K_obs, Q_obs^{-1})
     """
    if L_obs is None:
        Qc = Q+Q_obs
    else:
        # Do it this way round to avoid switching from CSC to CSR or vice versa.
        Qc = Q+(Q_obs*L_obs).__rmul__(L_obs.T)
    
    return Qc
        
def eta(M,Q):
    u"""
    Takes the following:
    - M: A mean vector
    - Q: A sparse precision matrix
    Returns the "canonical mean" η=Q M.
     """
    return Q*M

def compile_metropolis_sweep(fortran_likelihood_code):
    """
    Takes an unindented template of Fortran code with the following template parameters:
    - {X}: The value of the field at a vertex.
    - i: The index at the vertex
    - lv: Variables other than X that participate in the likelihood, as a len(x)-by-m array.
    - lkp: The log-likelihood at that vertex for value xp.
    The code should assign to lkp. A non-optimized logistic regression example:
    
    lX = dexp({X})/dexp(1+{X})
    n = lv(i,1)
    k = lv(i,2)
    lkp = n*dlog(lX) + k*dlog(1-lX)
    
    Returns a function for use by fast_metropolis_sweep. The generated code is included in the return object as the 'code' attribute.
    """
    def fortran_indent(s):
        return '\n'.join([' '*6+l for l in s.splitlines()])
    
    # Instantiate the template with the likelihood code snippet.
    fortran_likelihood_code = fortran_likelihood_code.replace('{X}','(xp+M(i))')
    fortran_code = file(os.path.join(os.path.split(pdefields.__file__)[0],'fast_metro.f')).read()
    fortran_code = fortran_code.replace('{LIKELIHOOD_CODE}', fortran_indent(fortran_likelihood_code))
    lpf_hash = hashlib.sha1(fortran_code).hexdigest()
    
    # Compile the generated code and import it.
    try:
        exec('import gmrf_metro_%s as gmrf_metro'%lpf_hash)
    except ImportError:
        for l in fortran_likelihood_code.splitlines():
            if len(l)>72-6:
                raise RuntimeError, 'Line "%s" in your log-likelihood code snippet is too long, Fortran will hurl.'%l
                
        f2py.compile(fortran_code, modulename='gmrf_metro_%s'%lpf_hash)
    exec('import gmrf_metro_%s as gmrf_metro'%lpf_hash)
    
    # Store the generated code on the function and return it.
    gmrf_metro.code = fortran_code
    return gmrf_metro

def fast_metropolis_sweep(M,Q,gmrf_metro,x,likelihood_variables=None,n_sweeps=10):
    """
    Takes:
    - Mean vector M
    - Sparse precision matrix Q, in SciPy CSC or CSR format
    - A Fortran module produced by compile_metropolis_sweep.
    - Current state vector x.
    - Optionally, any vertex-specific variables needed to compute the likelihoods as an m-by-(len(x)) array. For example, N and K in the binomial likelihood.
    
    Returns a new value for x, obtained by one-at-a-time Metropolis.
    """
    if Q.__class__ not in [sparse.csc.csc_matrix, sparse.csr.csr_matrix]:
        raise ValueError, "The value of Q must be a SciPy CSC or CSR matrix."
    
    # Unpack the sparse matrix.
    ind = Q.indices
    dat = Q.data
    ptr = Q.indptr
    diag = Q.diagonal()

    # Subtract off the mean.
    x_ = x-M
    
    # Square up the likelihood variables.
    if likelihood_variables is None:
        likelihood_variables = np.zeros(len(x))
    likelihood_variables = np.asarray(likelihood_variables, order='F')

    # Initialize the likelihoods
    log_likelihoods = gmrf_metro.lkinit(x_,M,likelihood_variables)

    # Call the Fortran code, add the mean back on and return.
    for i in xrange(n_sweeps): 
        # There's no need to do these vectorized operations in Fortran.
        acc = np.random.random(size=len(x))
        norms = np.random.normal(size=len(x))
        
        gmrf_metro.gmrfmetro(ind, dat, ptr, x_, log_likelihoods, diag, M, acc, norms, likelihood_variables)

    return x_ + M