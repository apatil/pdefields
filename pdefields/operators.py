"""
Functionality to assemble sparse matrices approximating differential operators on triangulated domains.
Agnostic to both the manifold and the linear algebra backend. All linear algebra goes via the backend.
"""

import numpy as np

# TODO: More operator specifications.
# TODO: Interior boundaries.
# TODO: Imaginary.
# TODO: Multivariate.

# TODO: Each operator should be able to return a sparsity pattern in addition to an actual precision.
def mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, backend):
    # TODO: Put an H matrix in there.
    # TODO: Spatially-varying kappa and amplitude.
    K = backend.axpy(kappa**2, Ctilde ,G)
    Ctilde_I_K = backend.dm_solve_m(Ctilde, K)
    
    def make_Q(alpha):
        if alpha == 1:
            return K
        elif alpha ==  2:
            return backend.m_mul_m(K,Ctilde_I_K)
        else:
            Q = make_Q(alpha-2)
            return backend.m_xtyx(Ctilde_I_K, Q)
    
    return make_Q(alpha)
    
