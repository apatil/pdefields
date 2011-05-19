import numpy as np
import pymc as pm
from pdefields import pymc_objects, operators, algorithms
from pdefields.backends import cholmod
from pdefields.manifolds import spherical
from scipy.special import gamma
from scipy import sparse

def make_model(X):
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
    M = np.random.normal(size=n)

    kappa = pm.Exponential('kappa',1,value=3)
    alpha = pm.DiscreteUniform('alpha',1,10,value=2.)
    diag_pert = pm.Exponential('diag_pert',1,value=0.)

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
    def precision_products(Q=Q, p=pattern_products, diag_pert=diag_pert):
        return cholmod.precision_to_products(Q, diag_pert=diag_pert, **p)

    S=pymc_objects.SparseMVN('S',M, precision_products, cholmod)

    vars = pm.rgamma(4,4,size=n)
    vals = X[:,2]

    data = pm.Normal('data', S, 1./vars, value=vals, observed=True)

    # TODO: Statistical test comparing Metropolis and Gibbs
    Qobs = sparse.csc_matrix((n,n))
    Qobs.setdiag(1./vars)

    # Stuff for the scoring algorithm-based full conditional
    def first_likelihood_derivative(x, vals=vals, vars=vars):
        return -(x-vals)/vars
    
    def second_likelihood_derivative(x, vals=vals, vars=vars):
        return -1./vars

    return locals()

if __name__ == '__main__':
    n = 250
    X = spherical.well_spaced_mesh(n)
    
    INLAParentAdaptiveMetropolis = pymc_objects.wrap_metropolis_for_INLA(pm.AdaptiveMetropolis)

    M = pm.MCMC(make_model(X))
    M.use_step_method(INLAParentAdaptiveMetropolis, [M.kappa, M.alpha], M.first_likelihood_derivative, M.second_likelihood_derivative, 1e-5, M.pattern_products)