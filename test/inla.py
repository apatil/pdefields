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
    out = operators.mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, cholmod)
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

vals = X[:,0]
vars = pm.rgamma(4,4,size=n)/10


# TODO: Statistical test comparing Metropolis and Gibbs
Qobs = sparse.csc_matrix((n,n))
Qobs.setdiag(1./vars)

# Stuff for the scoring algorithm-based full conditional
def first_likelihood_derivative(x, vals=vals, vars=vars):
    return -(x-vals)/vars
    
def second_likelihood_derivative(x, vals=vals, vars=vars):
    return -1./vars

# Stuff for the EP-based full conditional
lpf_str = "lkp = -({X}-lv(i,1))**2/2.0D0/lv(i,2)"
likelihood_vars = np.vstack((vals,vars)).T

# true_conditional_mean, cpp = cholmod.conditional_mean_and_precision_products(vals,M,Q.value+Qobs,Qobs,**pattern_products)
# M_conditional, precprod_conditional = algorithms.scoring_gaussian_full_conditional(M,Q.value,pattern_products,first_likelihood_derivative,second_likelihood_derivative,cholmod,1e-4)

prod = algorithms.EP_gaussian_full_conditional(M,Q.value,lpf_str,1.e-8,cholmod,pattern_products,likelihood_vars,n_bins=100)

# These should be zero
# print np.abs(true_conditional_mean - M_conditional).max()
# print np.abs(1./(precprod_conditional['Q']-Q.value).diagonal()-vars).max()
