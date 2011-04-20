"""
Functionality to assemble sparse matrices approximating differential operators on triangulated domains.
Agnostic to both the manifold and the linear algebra backend.
"""

import numpy as np

def C_diag_and_G(mesh, module, **params):
    C = compute_C(mesh, module, **params)
    C_diag = diagonalize_conserving_rowsums(C)
    G = compute_G(mesh, module, **params)
    return C_diag, G

# Fixme: Genericize operator specification    
def mod_frac_laplacian_precision_and_log_determinant(C_diag, G, kappa, alpha):
    K = kappa**2*C_diag + G
    C_diag_I_K = m_solve_m(C_diag, K)
    detK = log_determinant(K)
    det_C_Diag = log_determinant(C_diag)
    
    def make_Q_det(alpha):
        if alpha == 1:
            return K, detK
        elif alpha ==  2:
            return m_mul_m(K,C_diag_I_K), 2*detK - det_C_diag
        else:
            Q, det = make_Q_det(alpha-2)
            return xtyx(C_diag_I_K, Q), det + 2*(detK-det_C_diag)
    
    return make_Q_det(alpha)