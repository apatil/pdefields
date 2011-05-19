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
    
    # amp is the overall amplitude. It's a free variable that will probably be highly confounded with kappa.
    amp = pm.Exponential('amp', .0001, value=100)

    # A constant mean.
    m = pm.Uninformative('m',value=0)
    
    @pm.deterministic(trace=False)
    def M(m=m,n=len(X)):
        """The mean vector"""
        return np.ones(n)*m

    kappa = pm.Exponential('kappa',1,value=3)
    alpha = pm.DiscreteUniform('alpha',1,10,value=2., observed=True)

    @pm.deterministic(trace=False)
    def Q(kappa=kappa, alpha=alpha, amp=amp):
        out = operators.mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, cholmod)/np.asscalar(amp)**2
        return out

    # Nailing this ahead of time reduces time to compute logp from .18 to .13s for n=25000.
    pattern_products = cholmod.pattern_to_products(Q.value)
    # @pm.deterministic
    # def pattern_products(Q=Q):
    #     return cholmod.pattern_to_products(Q)

    @pm.deterministic(trace=False)
    def precision_products(Q=Q, p=pattern_products):
        return cholmod.precision_to_products(Q, **p)

    S=pymc_objects.SparseMVN('S',M, precision_products, cholmod)

    vars = pm.rgamma(4,4,size=n)
    vals = X[:,2]

    data = pm.Normal('data', S, 1./vars, value=vals, observed=True)
    
    Qobs = sparse.csc_matrix((n,n))
    Qobs.setdiag(1./vars)
    
    @pm.deterministic(trace=False)
    def true_evidence(Q=Q, M=M, vals=vals, vars=vars):
        C = np.array(Q.todense().I+np.diag(vars))
        return pm.mv_normal_cov_like(vals, M, C)
    
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
    M.use_step_method(INLAParentAdaptiveMetropolis, [M.kappa, M.m, M.amp, M.S], M.first_likelihood_derivative, M.second_likelihood_derivative, 1e-5, M.pattern_products)
    sm = M.step_method_dict[M.S][0]
    M.isample(1000,0,10)
    [pm.Matplot.plot(s) for s in [M.amp, M.kappa, M.amp]]