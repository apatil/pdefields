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
import pymc as pm

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

def compile_fortran_routine(fortran_likelihood_code, modname, template):
    "Used by compile_metropolis_sweep and compile_EP_sweep."
    def fortran_indent(s):
        return '\n'.join([' '*6+l for l in s.splitlines()])
    
    # Instantiate the template with the likelihood code snippet.
    fortran_likelihood_code = fortran_likelihood_code.replace('{X}','(xp+M(i))')
    
    fortran_code = template.replace('{LIKELIHOOD_CODE}', fortran_indent(fortran_likelihood_code))
    lpf_hash = hashlib.sha1(fortran_code).hexdigest()
    
    # Compile the generated code and import it.
    try:
        exec('import %s_%s as %s'%(modname, lpf_hash, modname))
    except ImportError:
        for l in fortran_likelihood_code.splitlines():
            if len(l)>72-6:
                raise RuntimeError, 'Line "%s" in your log-likelihood code snippet is too long, Fortran will hurl.'%l
                
        f2py.compile(fortran_code, modulename='%s_%s'%(modname,lpf_hash))
    exec('import %s_%s as %s'%(modname, lpf_hash, modname))

    # Store the generated code on the function and return it.
    exec('%s.code = fortran_code'%modname)
    exec('mod = %s'%modname)
    return mod

def compile_metropolis_sweep(fortran_likelihood_code):
    """
    Takes an unindented template of Fortran code with the following template parameters:
    - {X}: The value of the field at a vertex.
    - i: The index at the vertex
    - lv: Variables other than X that participate in the likelihood, as a len(x)-by-m array.
    - lkp: The log-likelihood at that vertex for value xp.
    The code should assign to lkp. A non-optimized logistic regression example:
    
    lkp = dexp({X})/(1.0D0+dexp({X}))
    lkp = lv(i,2)*dlog(lkp) + (lv(i,1)-lv(i,2))*dlog(1.0D0-lkp)
    
    Returns a function for use by metropolis_sweep. The generated code is included in the return object as the 'code' attribute.
    """
    template = file(os.path.join(os.path.split(pdefields.__file__)[0],'metro.f')).read()
    return compile_fortran_routine(fortran_likelihood_code, 'gmrf_metro', template)
    
def compile_likelihood_evaluation(fortran_likelihood_code):
    template = file(os.path.join(os.path.split(pdefields.__file__)[0],'likelihood_eval.f')).read()
    return compile_fortran_routine(fortran_likelihood_code, 'likelihood', template)
    
def compile_ep_sweep(fortran_likelihood_code):
    template = file(os.path.join(os.path.split(pdefields.__file__)[0],'ep.f')).read()
    return compile_fortran_routine(fortran_likelihood_code, 'gmrf_ep', template)

def fast_metropolis_sweep(M,Q,gmrf_metro,fortran_likelihood,x,likelihood_variables=None,n_sweeps=10):
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
    log_likelihoods = fortran_likelihood.lkinit(x_,M,likelihood_variables)

    # Call the Fortran code, add the mean back on and return.
    for i in xrange(n_sweeps): 
        # There's no need to do these vectorized operations in Fortran.
        acc = np.random.random(size=len(x))
        norms = np.random.normal(size=len(x))
        gmrf_metro.gmrfmetro(ind, dat, ptr, x_, log_likelihoods, diag, M, acc, norms, likelihood_variables)

    return x_ + M

def spmatvec(m,v):
    return (m*v.reshape((-1,1))).view(np.ndarray).ravel()

def scoring_gaussian_full_conditional(M,Q,pattern_products,like_deriv1,like_deriv2,backend,tol):
    """
    This function produces an approximate Gaussian full conditional for a multivariate normal variable x with sparse precision Q. This can be used to produce an approximate MCMC scheme or an INLA-like scheme.
    
    The scoring algorithm is described in section 2.2 of Rue, Martino and Chopin.
    
    Takes:
    - Mean vector M
    - Sparse precision matrix Q, in SciPy CSC or CSR format
    - pattern_products: The backend's precomputations based on sparsity pattern alone.
    - like_deriv1: A function that takes a value for x and returns an array whose i'th element is the derivative of the likelihood of datapoint i with respect to x[i].
    - like_deriv2: Same, but the second derivative.
    - The linear algebra backend that will handle the matrix solves.
    
    Returns the approximate full conditional mean of x, the backend's analysis of the full conditional precision Q, and the approximate evidence (the probability of the data conditional on M and Q, not x).
    """    
    x = M
    delta = x*0+np.inf
    while np.abs(delta).max() > tol:
        d1 = like_deriv1(x)
        d2 = like_deriv2(x)
        grad2 = d1-spmatvec(Q,(x-M))
        delta, precision_products = backend.precision_solve_v(Q,d1-spmatvec(Q,(x-M)),-d2,**pattern_products)
        
        x = x + delta
    like_vals = -d1/d2+x-delta
    return like_vals,-1/d2,x,precision_products
    
def gaussian_evidence(vals, vars, M, pp, Mc, ppc, x, backend):
    """
    This function gives p(vals) in 
    
    x ~ N(M,Q^{-1})
    vals[i] ~ N(x[i], vars[i])
    
    where: 
    - pp are the backend's precision products of Q
    - Mc is E[x|y]
    - ppc is the backend's precision products of the conditional precision of x on y
    - backend is a linear algebra backend
    
    The answer should not depend on x, but Mc is probably a good choice.3
    """
    
    # p(x|y)p(y) = p(x,y)
    # p(y) = p(y|x)p(x)/p(x|y)
    pygx = pm.normal_like(vals, x, 1./vars)
    pxgy = backend.mvn_logp(x, Mc, **ppc)
    px = backend.mvn_logp(x, M, **pp)
    return px + pygx - pxgy
    
# def with_delta(x,new_x,delta):
#     delta = max(delta, np.abs(x-new_x))
#     return new_x, delta
#     
# def EP_gaussian_full_conditional(M,Q,fortran_likelihood_code,tol,backend,pattern_products,likelihood_variables=None,n_bins=100,sd_width=10):
#     """
#     Blah
#     """
#     import warnings
#     warnings.warn('The EP algorithm as currently implemented is slow and incorrect.')
#     
#     # Prepare gridpoints for numerical integration.
#     int_pts = np.linspace(-sd_width/2.,sd_width/2., n_bins)
#     dint_pts = int_pts[1]-int_pts[0]
#     diag = Q.diagonal()
#     prior_sds = 1./np.sqrt(diag)
#     
#     from scipy import integrate
#     
#     like_eval = compile_likelihood_evaluation(fortran_likelihood_code)
#     if likelihood_variables is None:
#         likelihood_variables = np.zeros(len(M))
#     likelihood_variables = np.asarray(likelihood_variables, order='F')
#     
#     Q_obs = sparse.csc_matrix((Q.shape))
#     nx = len(M)
#     delta = np.inf
#     effective_obsvals = 0*M
#     effective_obsvars = 0*M+np.inf
#     while delta > tol:
#         delta = 0
#         Q_obs.setdiag(1./effective_obsvars)
#         mc, precision_products = backend.conditional_mean_and_precision_products(effective_obsvals,M,Q+Q_obs,Q_obs,**pattern_products)
#         for i in xrange(nx):
#             # TODO: The loop is embarrassingly parallel and can be written in Fortran or PyCuda (note no 'if's).
#             
#             x_for_integral = int_pts*prior_sds[i] + mc[i]
#             dx = dint_pts * prior_sds[i]
#             
#             # Evaluate likelihood over x-axis of integral.
#             likes = np.array([like_eval.lkinit(np.atleast_1d(xi), np.atleast_1d(0), np.atleast_2d(likelihood_variables[i,:])) for xi in x_for_integral]).ravel()
#             # Posterior \propto prior * likelihood.
#             posteriors = -(x_for_integral-mc[i])**2/2*diag[i] + likes
#             # This is going to be exponentiated and the normalizing constant isn't known anyway, so make the numbers
#             # reasonable sized to avoid numerical problems.
#             posteriors -= posteriors.max()
#             
#             # Normalizing constant.
#             norm = integrate.simps(np.exp(posteriors),None, dx)
# 
#             # Full conditional mean
#             m = integrate.simps(np.exp(posteriors)*x_for_integral,None, dx)/norm
#             m2 = integrate.simps(np.exp(posteriors)*x_for_integral**2,None, dx)/norm
#             
#             # Full conditional variance
#             v = m2-m**2
#             
#             # Back out the 'observation' value and measurement variance
#             effective_obsvars[i], delta = with_delta(effective_obsvars[i], 1/(1/v-diag[i]), delta)
#             effective_obsvals[i], delta = with_delta(effective_obsvals[i], effective_obsvars[i]*(m/v-mc[i]*diag[i]), delta)
#     return effective_obsvals, effective_obsvars, mc, precision_products