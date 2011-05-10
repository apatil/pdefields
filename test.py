# Triangulation

import spherical
import numpy as np
import pymc as pm
import interface
import operators
import backends
from backends import cholmod

n = 25000
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
Ctilde = backends.cholmod.into_matrix_type(Ctilde)
G = backends.cholmod.into_matrix_type(G)
Q = operators.mod_frac_laplacian_precision(Ctilde, G, 1, 1, backends.cholmod)

# Variable
M = np.zeros(n)
pattern_products = cholmod.pattern_to_products(Q)
precision_products = cholmod.precision_to_products(Q, **pattern_products)

kls = interface.sparse_MVN(cholmod)
S=kls('S',M, **precision_products)

import spherical
rast = spherical.mesh_to_map(X,S.value)