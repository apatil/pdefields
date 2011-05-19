# Triangulation


import numpy as np
import pymc as pm
from pdefields import pymc_objects, operators, algorithms
from pdefields.backends import cholmod
from pdefields.manifolds import spherical
from scipy.special import gamma
from scipy import sparse

n = 250

X = spherical.well_spaced_mesh(n)

neighbors, triangles, trimap, b = spherical.triangulate_sphere(X)
# spherical.plot_triangulation(X,neighbors)

# Matrix generation
triangle_areas = [spherical.triangle_area(X, t) for t in triangles]
Ctilde = spherical.Ctilde(X, triangles, triangle_areas)
C = spherical.C(X, triangles, triangle_areas)
G = spherical.G(X, triangles, triangle_areas)

# Operator generation
Ctilde = cholmod.into_matrix_type(Ctilde)
G = cholmod.into_matrix_type(G)
M = np.zeros(n)

kappa = pm.Exponential('kappa',1,value=3)
alpha = pm.DiscreteUniform('alpha',1,10,value=2.)


@pm.deterministic
def Q(kappa=kappa, alpha=alpha):
    out = operators.mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, cholmod)
    return out

# Nailing this ahead of time reduces time to compute logp from .18 to .13s for n=25000.
pattern_products = cholmod.pattern_to_products(Q.value)
# @pm.deterministic
# def pattern_products(Q=Q):
#     return cholmod.pattern_to_products(Q)

@pm.deterministic
def precision_products(Q=Q, p=pattern_products):
    return cholmod.precision_to_products(Q, **p)

S=pymc_objects.SparseMVN('S',M, precision_products, cholmod)

def map_S(S):
    # Make a map
    rast = spherical.mesh_to_map(X,S.value,501)
    import pylab as pl
    pl.clf()
    pl.imshow(rast,interpolation='nearest')
    pl.colorbar()

S.rand()
lpf = [lambda x: 0 for i in xrange(n)]
lp = 0*S.value


vals = X[:,0]
vars = pm.rgamma(4,4,size=n)/1000

likelihood_vars = np.vstack((vals,vars)).T

Qobs = sparse.csc_matrix((n,n))

lpf_str = "lkp = -({X}-lv(i,1))**2/2.0D0/lv(i,2)"
Qobs.setdiag(1./vars)

# lpf_str = "lkp=0"
# Qobs.setdiag(0*vars+1e-8)

import pylab as pl

S.rand()
metro = pymc_objects.GMRFMetropolis(S,lpf_str,M,Q,likelihood_vars,n_sweeps=10000)
gibbs = pymc_objects.GMRFGibbs(cholmod, S, vals, M, Q, Qobs, pattern_products=pattern_products)

def metrostep():
    metro.step()
    pl.figure(1)
    map_S(S)
    pl.title('Metropolis')

def gibbsstep():
    gibbs.step()
    pl.figure(2)
    map_S(S)
    pl.title('Gibbs')
    
# devs = []
# for i in xrange(100):
#     metro.step()
#     devs.append(S.value-S.last_value)
# devs = np.array(devs)

metrostep()
gibbsstep()