"""
Functionality to assemble sparse matrices approximating differential operators on triangulated domains.
Agnostic to both the manifold and the linear algebra backend.
"""

import numpy as np
import linalg

# Fixme: Genericize operator specification    
def mod_frac_laplacian_precision_and_log_determinant(Ctilde, G, kappa, alpha):
    K = linalg.axpy(kappa**2, Ctilde ,G)
    Ctilde_I_K = linalg.m_solve_m(Ctilde, K)
    detK = linalg.log_determinant(K)
    det_C_Diag = log_determinant(Ctilde)
    
    def make_Q_det(alpha):
        if alpha == 1:
            return K, detK
        elif alpha ==  2:
            return m_mul_m(K,Ctilde_I_K), 2*detK - det_Ctilde
        else:
            Q, det = make_Q_det(alpha-2)
            return xtyx(Ctilde_I_K, Q), det + 2*(detK-det_Ctilde)
    
    return make_Q_det(alpha)