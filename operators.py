"""
Functionality to assemble sparse matrices approximating differential operators on triangulated domains.
Agnostic to both the manifold and the linear algebra backend.
"""

import numpy as np
import linalg

# TODO: More operator specifications.
# TODO: Interior boundaries.
# TODO: Imaginary.
# TODO: Multivariate.

def mod_frac_laplacian_precision_and_log_determinant(Ctilde, G, kappa, alpha):
    # TODO: Put an H matrix in there.
    # TODO: Spatially-varying kappa and amplitude.
    K = linalg.axpy(kappa**2, Ctilde ,G)
    Ctilde_I_K = linalg.dm_solve_m(Ctilde, K)
    detK = linalg.log_determinant(K)
    det_Ctilde = np.log(linalg.extract_diagonal(Ctilde)).sum()
    
    def make_Q_det(alpha):
        if alpha == 1:
            return K, detK
        elif alpha ==  2:
            return linalg.m_mul_m(K,Ctilde_I_K), 2*detK - det_Ctilde
        else:
            Q, det = make_Q_det(alpha-2)
            return linalg.m_xtyx(Ctilde_I_K, Q), det + 2*(detK-det_Ctilde)
    
    return make_Q_det(alpha)
    
