# Triangulation

import spherical
import numpy as np
import pymc as pm

n = 500
X = np.random.normal(size=(3,n))
X /= np.sqrt((X**2).sum(axis=0))
X = X.T

neighbors, triangles, trimap, b = spherical.triangulate_sphere(X)
# spherical.plot_triangulation(X,neighbors)

# Matrix generation
triangle_areas = [spherical.triangle_area(X, t) for t in triangles]
Ctilde = spherical.Ctilde(X, triangles, triangle_areas)
C = spherical.C(X, triangles, triangle_areas)
G = spherical.G(X, triangles, triangle_areas)

# Operator generation
import operators
import linalg
Ctilde = linalg.into_matrix_type(Ctilde)
G = linalg.into_matrix_type(G)
Q, det = operators.mod_frac_laplacian_precision_and_log_determinant(Ctilde, G, 1, 1)

# Field and plot.
f = linalg.rmvn(Q)