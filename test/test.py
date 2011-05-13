# Triangulation


import numpy as np
import pymc as pm
from pdefields import spherical, interface, operators, backends
from pdefields.backends import cholmod
from scipy.special import gamma

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

S=interface.SparseMVN('S',M, precision_products, cholmod)

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

lpf_str = """
if (dabs({LV}({I},1)).GT.0.3) then
{LP}={X}
else
{LP}=-{X}
end if
"""
lp = 0*S.value
lp[np.where(np.abs(X[:,0])>.3)]=S.value[np.where(X[:,0]>.3)]

gmrfmetro = cholmod.compile_metropolis_sweep(lpf_str)
S.value,lp = cholmod.fast_metropolis_sweep(M,Q.value,gmrfmetro,S.value,lp,X)
map_S(S)

# print condm1-condm3