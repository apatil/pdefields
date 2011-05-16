# -*- coding: UTF-8 -*-
"""
A linear algebra backend that uses sparse, cholesky-based operations.
"""
__all__ = ['into_matrix_type', 'precision_to_products', 'pattern_to_products', 'rmvn', 'mvn_logp', 'axpy', 'dm_solve_m', 'm_mul_m', 'm_xtyx', 'eta', 'fast_metropolis_sweep', 'compile_metropolis_sweep']

import numpy as np
import scipy
from scipy import sparse
import scikits.sparse.cholmod as cholmod

precision_product_keys = ['L','P','F']

def into_matrix_type(m):
    "Takes a matrix m and returns a representation of it as a SciPy compressed sparse column matrix. This is the matrix format used by the CHOLMOD wrapper in scikits.sparse."
    return sparse.csc.csc_matrix(m)
    
def axpy(a,x,y):
    "a,x,y -> ax+y"
    return a*x + y

def dm_solve_m(x,y):
    "A^{-1} B, where A is diagonal and B is CSC."
    x_i = x.copy()
    x_i.data = 1./x_i.data
    return x_i * y

def m_xtyx(x,y):
    "x,y -> x^T y x ."
    # Do it this way to stay in CSC format.
    return x.__rmul__(x.T*y)

def m_mul_m(x,y):
    "x,y -> xy"
    return x*y
    
def pattern_to_products(pattern):
    """Takes a sparse matrix with the correct sparsity pattern, but not necessarily meaningful values, and returns the symbolic Cholesky factorization of it, computed by CHOLMOD via scikits.sparse. The symbolic factorization is stored in a Factor object. Its method P can be called to obtain the permutation vector. I don't know if there's any way to get the actual sparsity pattern out, but it can surely be done.
    
    The return value is stored in a singleton dictionary with one key, 'symbolic'. It is stored in a dictionary to make it possible to have a uniform return type across all backends."""
    return {'symbolic': cholmod.analyze(pattern)}

def precision_to_products(Q, diag_pert, symbolic):
    """
    Takes a sparse precision matrix Q and a symbolic Cholesky factorization 'symbolic' and returns several derivative products in a dictionary:
    - Q: The input precision matrix, unaltered.
    - F: The Factor object generated by scikits.sparse. This object supports quick and easy triangular solves.
    - det: The determinant of Q.
    - P: A permutation vector. Cholmod computes the Cholesky factorization LDL^T = Q[P,:][:,P].
    - Pbak: The backward permutation vector. x[P][Pbax] = x, and (LDL^T)[Pbak,:][:,Pbak] = Q
    - sqrtD: sqrt(D).
    - """
    # FIXME: This should be symbolic.cholesky(Q)... but that doesn't work when alpha=2. Why?
    # F = cholmod.cholesky(Q)
    F = symbolic.cholesky(Q,beta=diag_pert)
    D = F.D()
    sqrtD = np.sqrt(D)
    det = np.sum(np.log(D))
    P = symbolic.P()
    Pbak = np.argsort(P)
    return {'det':det, 'P': P, 'F': F, 'Pbak': Pbak, 'Q': Q, 'sqrtD': sqrtD}

def rmvn(M,Q,det,F,P,Pbak,sqrtD):
    """
    Takes the following:
    - M: A mean vector
    - Q: A sparse precision matrix
    - det: The determinant of Q
    - F: A scikits.sparse Factor object representing the Cholesky factorization of Q
    - P: A permutation vector
    - Pbak: The permutation vector that inverts P.
    - sqrtD: The square root of the diagonal matrix D from Cholmod's Cholesky factorization.
    Returns a draw from the multivariate normal distribution ith mean M and precision P
    """
    return M + F.solve_Lt(np.random.normal(size=len(M)) / sqrtD)[Pbak]

def mvn_logp(x,M,Q,det,F,P,Pbak,sqrtD):
    """
    Takes the following:
    - x: A candidate value as a vector.
    - M: A mean vector
    - Q: A sparse precision matrix
    - det: The determinant of Q
    - F: A scikits.sparse Factor object representing the Cholesky factorization of Q
    - P: A permutation vector
    - Pbak: The permutation vector that inverts P.
    - sqrtD: The square root of the diagonal matrix D from Cholmod's Cholesky factorization.    
    Returns the log-probability of x given M and Q.
     """
    d = (x-M)[P]
    return -.5*np.dot(d,Q*d) + .5*det - .5*len(M)*np.log(2.*np.pi)

def eta(M,Q,det,F,P,Pbak,sqrtD):
    u"""
    Takes the following:
    - x: A candidate value as a vector.
    - M: A mean vector
    - Q: A sparse precision matrix
    - det: The determinant of Q
    - F: A scikits.sparse Factor object representing the Cholesky factorization of Q
    - P: A permutation vector
    - Pbak: The permutation vector that inverts P.
    - sqrtD: The square root of the diagonal matrix D from Cholmod's Cholesky factorization.    
    Returns the "canonical mean" η=Q M.
     """
    return Q*M        

def compile_metropolis_sweep(lpf_string):
    """
    Takes an unindented template of Fortran code with the following template parameters:
    - {X}: The value of the field at a vertex.
    - {I}: The index at the vertex
    - {LV}: Variables other than X that participate in the likelihood, as a len({X})-by-m array.
    - {LP}: The log-likelihood at that vertex.
    The code should assign to {LP}. A non-optimized example:
    
    lX = dexp({X})/dexp(1+{X})
    n = {LV}({I},1)
    k = {LV}({I},2)
    {LP} = n*dlog(lX) + k*dlog(1-lX)
    
    Returns a function for use by fast_metropolis_sweep. The generated code is included in the return object as the 'code' attribute.
    """
    def fortran_indent(s):
        return '\n'.join([' '*6+l for l in s.splitlines()])
    
    # Instantiate the template with the likelihood code snippet.
    import hashlib
    from numpy import f2py
    lpf_string = lpf_string.replace('{X}','(xp+M(i))')\
                            .replace('{LP}','lpp')\
                            .replace('{I}','i')\
                            .replace('{LV}', 'lv')
    import cholmod
    import os
    fortran_code = file(os.path.join(os.path.split(cholmod.__file__)[0],'fast_metro.f')).read()
    fortran_code = fortran_code.replace('{LPF}', fortran_indent(lpf_string))
    lpf_hash = hashlib.sha1(fortran_code).hexdigest()
    
    # Compile the generated code and import it.
    try:
        exec('import gmrf_metro_%s as gmrf_metro'%lpf_hash)
    except ImportError:
        f2py.compile(fortran_code, modulename='gmrf_metro_%s'%lpf_hash)
    exec('import gmrf_metro_%s as gmrf_metro'%lpf_hash)
    
    # Store the generated code on the function and return it.
    gmrf_metro.gmrfmetro.code = fortran_code
    return gmrf_metro.gmrfmetro

def fast_metropolis_sweep(M,Q,gmrfmetro,x,log_likelihoods,likelihood_variables=None,n_sweeps=1):
    """
    Takes:
    - Mean vector M
    - Sparse precision matrix Q, in SciPy CSC or CSR format
    - A function produced by compile_metropolis_sweep.
    - Current state vector x.
    - The current log-likelihoods at each point in x. Note: these are overwritten in-place.
    - Optionally, any vertex-specific variables needed to compute the likelihoods as an m-by-(len(x)) array. For example, N and K in the binomial likelihood.
    
    Returns a new value for x, obtained by one-at-a-time Metropolis, and the corresponding log-likelihoods.
    """
    # Unpack the sparse matrix.
    ind = Q.indices
    dat = Q.data
    ptr = Q.indptr
    diag = Q.diagonal()

    # Subtract off the mean.
    x_ = x-M
    
    cond_std = 1./np.sqrt(diag)
    
    # Square up the likelihood variables.
    if likelihood_variables is None:
        likelihood_variables = np.zeros(len(x))
    likelihood_variables = np.asarray(likelihood_variables, order='F')

    # Call the Fortran code, add the mean back on and return.
    for i in xrange(n_sweeps): 
        # There's no need to do these vectorized operations in Fortran.
        acc = np.random.random(size=len(x))
        norms = np.random.normal(size=len(x))*cond_std
        gmrfmetro(ind, dat, ptr, x_, log_likelihoods, diag, acc, norms, M, likelihood_variables)

    return x_ + M, log_likelihoods    

def id(x):
    return x    
    
def conditional_mean_and_precision(y,M,Q,Q_obs,L_obs=None,K_obs=None,symbolic=None):
    """    
    Returns the conditional mean and precision of x in the conjugate submodel
    
    x ~ N(M,Q^{-1})
    y ~ N(L_obs x + K_obs, Q_obs^{-1})
    
    Takes:
    - Observed value of y
    - Mean vector M
    - Sparse SPD matrices Q and Qobs
    - Sparse matrix, dense matrix or LinearOperator L_obs (optional)
    - Offset vector K_obs (optional)
    - Symbolic Cholesky factorization previously returned by conditional_mean_and_precision (optional)
    
    Returns:
    - Conditional precision as a CSC matrix
    - Symbolic Cholesky factorization of conditional precision as a Cholmod Factor
    - Numeric "
    - Conditional mean 
    """
    # Note that the joint precision is
    # [Q + L_obs' Q_obs L_obs      -L_obs' Q_obs]
    # [-Q_obs L_obs                 Q_obs]
    
    # L_obs defaults to the identity operator
    if L_obs is None:
        from scipy.sparse import linalg
        L_obs = linalg.LinearOperator((len(y),len(x)), id, id)
        
    # K_obs defaults to the zero vector.
    if K_obs is None:
        K_obs = 0*y
    
    # Conditional precision.
    Qc = Q+(Q_obs*L_obs).__rmul__(L_obs.T)
    
    # If no symbolic factorization was provided, compute it here.
    if symbolic is None:
        symbolic = cholmod.analyze(Qc)
    
    # Numeric factorization and conditional mean.
    delta = y-L_obs*M-K_obs
    numeric = symbolic.cholesky(Qc)
    Mc = M + numeric.solve_A(L_obs.T*Q_obs*delta).reshape(M.shape)

    return Qc, symbolic, numeric, Mc
    
# if __name__ == '__main__':
#     n = 100
#     nobs = 50
#     B = np.random.normal(size=(n,n)).view(np.matrix)
#     Q = B*B.T
#     L = np.random.normal(size=(n,nobs)).view(np.matrix)
#     M = np.random.normal(size=n).reshape((-1,1))
#     
#     B = np.random.normal(size=(nobs,nobs)).view(np.matrix)
#     Qobs = B*B.T
#     
#     Cx = Q.I
#     Cxy = L.T * Cx
#     Cy = L.T* Cx * L + Qobs.I
#     Ccombo = np.bmat([[Cx,Cxy.T],[Cxy,Cy]])
# 
#     Qcombo = Ccombo.I
#     Qx = Q + L*Qobs*L.T
#     Qxy = -L*Qobs
#     # Qcombo_ = np.bmat([[Qx,Qxy],[Qxy.T,Qobs]])
#     
#     # Equivalent.
#     Ccond = Cx-Cxy.T*Cy.I*Cxy
#     # Ccond_try = Qcombo_[:n,:n].I
#     
#     xobs = L.T*M + np.dot(np.linalg.cholesky(Cy), np.random.normal(size=nobs)).reshape((-1,1))
#     Mcond = M+Cxy.T*Cy.I*(xobs-L.T*M)
#     
#     # eta = Qcombo_*np.vstack((M,L.T*M))
#     # etacond = eta[:n]-Qxy*xobs
#     # Mcond_try = Qx.I*etacond
#     # Mcond_try = M+Qx.I*L*Qobs*(xobs-L.T*M)
#     
#     Qx_try, symbolic, numeric, Mcond_try = conditional_mean_and_precision(xobs, M, *map(sparse.csc_matrix, [Q, Qobs, L.T]))
#     
#     print np.abs(Mcond_try.view(np.ndarray).ravel()-Mcond.view(np.ndarray).ravel()).max()
#     