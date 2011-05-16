# Triangulation


import numpy as np
import pymc as pm
from pdefields import spherical, pymc_objects, operators, backends, mcmc
from pdefields.backends import cholmod
from scipy.special import gamma
from scipy import sparse

n = 25000

X = spherical.well_spaced_mesh(n)

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
diag_pert = pm.Exponential('diag_pert',1,value=0.)

@pm.deterministic
def normconst(kappa=kappa,alpha=alpha):
    """normconst = function(parents)"""
    d = 2.
    nu = alpha - d/2
    normconst = gamma(nu)/(gamma(nu+d/2)*(4.*np.pi)**(d/2)*kappa**(2*nu))
    return normconst

@pm.deterministic
def Q(kappa=kappa, alpha=alpha):
    out = operators.mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, backends.cholmod)
    return out

# Nailing this ahead of time reduces time to compute logp from .18 to .13s for n=25000.
pattern_products = cholmod.pattern_to_products(Q.value)
# @pm.deterministic
# def pattern_products(Q=Q):
#     return cholmod.pattern_to_products(Q)

@pm.deterministic
def precision_products(Q=Q, p=pattern_products, diag_pert=diag_pert,normconst=normconst):
    return cholmod.precision_to_products(Q, diag_pert=diag_pert*normconst, **p)

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

likelihood_vars = np.vstack((vals,vars)).reshape((-1,2))

# TODO: Statistical test comparing Metropolis and Gibbs
lpf_str = """
lkp = ({X}-lv(i,1))**2/2/lv(i,2)
"""
Qobs = sparse.csc_matrix((n,n))
Qobs.setdiag(1./vars)

# gmrfmetro = mcmc.compile_metropolis_sweep(lpf_str)
# S.value = mcmc.fast_metropolis_sweep(M,Q.value,gmrfmetro,S.value,likelihood_vars,n_sweeps=100)

# metro = pymc_objects.GMRFMetropolis(S,lpf_str,M,Q,likelihood_vars,n_sweeps=100)
# metro.step()

gibbs = pymc_objects.GMRFGibbs(cholmod, S, vals, M, Q, Qobs, pattern_products=pattern_products)
gibbs.step()

map_S(S)

# print condm1-condm3