# -*- coding: UTF-8 -*-
"""
Functionality to assemble sparse matrices approximating differential operators on triangulated domains.
Agnostic to both the manifold and the linear algebra backend. All linear algebra goes via the backend.
"""

import numpy as np
from scipy.special import gamma

# TODO: More operator specifications.
# TODO: Interior boundaries.
# TODO: Imaginary.
# TODO: Multivariate.
# TODO: Put an H matrix in there.
# TODO: Spatially-varying kappa and amplitude.

def mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, backend):
    u"""
    This function takes the following, with ψ_i being the finite element basis function at vertex i: 

    - Matrix Ctilde, which is diagonal with diagonal element i equal to <ψ_i, 1>.
    - Matrix G, whose i,jth element is <∇ ψ_i, ∇ ψ_j>
    - Kappa, a positive float: the scale parameter
    - Alpha, a positive integer multiple of 1/2: the smoothness parameter
    - The linear algebra backend that should be used.

    It assumes that the sparse matrices Ctilde and G are already in the format required by the backend, ie they are the output of a call to backend.into_matrix_type. All matrix operations are delegated to the backend, so the returned precision matrix will be in the same format.
    """
    
    # Normalize.
    
    # axpy(a, x, y) returns ax + y.
    K = backend.axpy(kappa**2, Ctilde ,G)

    # dm_solve_m solves a diagonal matrix against a matrix.
    Ctilde_I_K = backend.dm_solve_m(Ctilde, K)
    
    def make_Q(alpha):
        if alpha == 1:
            return K
        elif alpha ==  2:
            # m_mul_m multiplies two matrices.
            return backend.m_mul_m(K,Ctilde_I_K)
        else:
            Q = make_Q(alpha-2)
            # m_xtyx returns X.T Y X
            return backend.m_xtyx(Ctilde_I_K, Q)
    
    return make_Q(alpha)
    
