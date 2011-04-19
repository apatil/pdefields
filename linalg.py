import numpy as np

def build_sparse_matrix(mesh, f, symm=False):
    # FIXME: Sparse
    out = np.empty((len(mesh),len(mesh)))

    for node, neighbors in mesh.iteritems():
        for neighbor in neighbors:
            if symm and neighbor > node:
                continue
            else:
                out[node, neighbor] = f(node, neighbor, mesh)
                if symm:
                    out[neighbor,node] = out[node,neighbor]
                
    return out

def manifold_innerprod(node, neighbor, mesh):
    # FIXME: mock
    return node==neighbor

def manifold_gradient_innerprod(node, neighbor, mesh):
    # FIXME: mock
    return node==neighbor

def compute_C(mesh):
    return build_sparse_matrix(mesh, manifold_innerprod, True)
    
def compute_G(mesh):
    return build_sparse_matrix(mesh, manifold_gradient_innerprod, True)

# FIXME: Everything above here may be done by wrapping fmesher.

def m_xtyx(x,y):
    # FIXME: Sparse
    return np.dot(np.dot(x.T,y),x)

def v_xtyx(x,y):
    # FIXME: Sparse
    return m_xtyx(x,y)
    
def m_mul_m(x,y):
    # FIXME: Sparse
    return np.dot(x,y)

def m_solve_m(x,y):
    # FIXME: Sparse
    return np.linalg.solve(x,y)
    
def m_solve_v(x,y):
    return np.linalg.solve(x,y)

def m_mul_v(x,y):
    # FIXME: Sparse
    return np.dot(x,y)

def matrix_from_diag(d):
    # FIXME: Sparse
    return np.eye(d.shape[0])*d
    
def diagonalize_conserving_rowsums(x):
    new_diag = m_mul_v(x, np.ones(x.shape[0]))
    return matrix_from_diag(new_diag)
    
def C_diag_and_G(mesh):
    C = compute_C(mesh)
    C_diag = diagonalize_conserving_rowsums(C)
    G = compute_G(mesh)
    return C_diag, G

def log_determinant(x):
    # FIXME: sparse
    # Can the method of Bai et al. be used?
    return np.log(np.linalg.det(x))
    
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