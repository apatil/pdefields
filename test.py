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
M = np.zeros(n)

kappa = pm.Exponential('kappa',1,value=1)
alpha = pm.DiscreteUniform('alpha',1,10,value=1.)

@pm.deterministic
def Q(kappa=kappa, alpha=alpha):
    return operators.mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, backends.cholmod)

# Nailing this ahead of time reduces time to compute logp from .18 to .13s for n=25000.
pattern_products = cholmod.pattern_to_products(Q.value)
# @pm.deterministic
# def pattern_products(Q=Q):
#     return cholmod.pattern_to_products(Q)

@pm.deterministic
def precision_products(Q=Q, p=pattern_products):
    return cholmod.precision_to_products(Q, **p)

S=interface.SparseMVN('S',M, precision_products, cholmod)

def map_S(S):
    # Make a map
    import spherical
    rast = spherical.mesh_to_map(X,S.value,501)
    import pylab as pl
    pl.clf()
    pl.imshow(rast,interpolation='nearest',vmin=-4,vmax=4,extent=(-2,2,-1,1))
    pl.colorbar()