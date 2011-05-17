# Triangulation

import numpy as np
import pymc as pm
from pdefields import spherical, pymc_objects, operators, backends, mcmc
from pdefields.backends import cholmod
from scipy.special import gamma
from scipy import sparse

def make_model(N,k,X,backend,manifold):
    """
    A standard spatial logistic regression.
    - N: Number sampled at each location
    - k: Number positive at each location
    - X: x,y,z coords of each location
    - Backend: The linear algebra backend. So far, this has to be 'cholmod'. 
    - manifold: The manifold to work on. So far, this has to be 'spherical'.
    """
    
    # Make the Delaunay triangulation.
    neighbors, triangles, trimap, b = manifold.triangulate_sphere(X)

    # Uncomment to visualize the triangulation.
    # manifold.plot_triangulation(X,neighbors)

    # Generate the C, Ctilde and G matrix in SciPy 'lil' format.
    triangle_areas = [manifold.triangle_area(X, t) for t in triangles]
    Ctilde = manifold.Ctilde(X, triangles, triangle_areas)
    C = manifold.C(X, triangles, triangle_areas)
    G = manifold.G(X, triangles, triangle_areas)

    # Convert to SciPy 'csc' format for efficient use by the CHOLMOD backend.
    C = backend.into_matrix_type(C)
    Ctilde = backend.into_matrix_type(Ctilde)
    G = backend.into_matrix_type(G)

    # Kappa is the scale parameter. It's a free variable.
    kappa = pm.Exponential('kappa',1,value=3)

    # Fix the value of alpha.
    alpha = 2.

    # amp is the overall amplitude. It's a free variable that will probably be highly confounded with kappa.
    amp = pm.Exponential('amp', .0001, value=1)

    # A constant mean.
    m = pm.Uninformative('m',value=0)
    
    @pm.deterministic(trace=False)
    def M(m=m,n=len(X)):
        """The mean vector"""
        return np.ones(n)*m
        
    @pm.deterministic(trace=False)
    def Q(kappa=kappa, alpha=alpha, Ctilde=Ctilde, G=G, backend=backend):
        "The precision matrix."
        out = operators.mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, backend)
        return out

    # Do all the precomputation you can based on the sparsity pattern alone.
    # Note that if alpha is made free, this needs to be free also, as the sparsity
    # pattern will be changeable.
    pattern_products = backend.pattern_to_products(Q.value)

    @pm.deterministic(trace=False)
    def precision_products(Q=Q, p=pattern_products):
        "All the analysis of the precision matrix that the backend needs to do MVN computations."
        try: 
            return backend.precision_to_products(Q, diag_pert=0, **p)
        except backend.NonPositiveDefiniteError:
            return None

    # The random field.
    S=pymc_objects.SparseMVN('S',M, precision_products, backend)
    
    @pm.deterministic
    def p(S=S):
        """The success probability."""
        return pm.invlogit(S)

    # The data.
    data = pm.Binomial('data', n=N, p=p, value=k, observed=True)
    
    # A Fortran representation of the likelihood, to allow for fast Metropolis steps without querying data.logp.
    likelihood_variables = np.vstack((np.resize(N,k.shape),k)).T
    likelihood_string = """
    lkp = dexp({X})/dexp(1.0D0+{X})
    lkp = lv(i,2)*dlog(lkp) + (lv(i,1)-lv(i,2))*dlog(1.0D0-lkp)
    """
    
    return locals()

if __name__ == '__main__':
    ###############################
    # Simulate data.
    ###############################

    # How many datapoints?
    n = 250

    # Put down a random scattering of data locations on the unit sphere.
    X = spherical.well_spaced_mesh(n)

    # Generate some binomial data. Prevalence is going to be high at the equator, low at the poles.
    p_true = np.exp(-X[:,2]**2*5)

    # Number sampled and number positive.
    N = 100
    k = pm.rbinomial(N, p_true)
    
    ################################
    # Fit the model.
    ################################
    M = pm.MCMC(make_model(N,k,X,cholmod,spherical))
    M.use_step_method(pm.AdaptiveMetropolis, [M.kappa, M.amp, M.m])
    M.use_step_method(pymc_objects.GMRFMetropolis, M.S, M.likelihood_string, M.M, M.Q, M.likelihood_variables, n_sweeps=100)
    M.isample(2000,0,10)
    
    ################################
    # Visualize the results
    ################################
    
    # Traces of the scalar variables.
    import pylab as pl
    pl.close('all')
    scalar_variables = [M.m, M.amp, M.kappa]
    for v in scalar_variables:
        pm.Matplot.plot(v)
        
    # Make mean and variance maps
    burn = 100
    thin = 1
    resolution = 501
    m1 = np.zeros((resolution/2+1,resolution))
    m2 = np.zeros((resolution/2+1,resolution))
    nmaps = 0
    for i in xrange(burn, len(M.trace('kappa')[:]), thin):
        M.remember(0,i)
        # Note, this rasterization procedure is not great. It's using SciPy, which is assuming that the data are on the plane. It would be better to either:
        # - Take the finite element representation literally, and evaluate it on the grid
        # - Sample to the interior conditional on the finite element representation.
        gridded_p = spherical.mesh_to_map(X, M.p.value, resolution)
        m1 += gridded_p
        m2 += gridded_p**2
        nmaps += 1
    
    mean_map = np.ma.masked_array(m1 / nmaps, mask=gridded_p==np.nan)
    variance_map = np.ma.masked_array(m2 / nmaps - mean_map**2, mask=gridded_p==np.nan)
    
    pl.figure(len(scalar_variables)+1)
    pl.imshow(mean_map[::-1,:], interpolation='nearest')
    pl.colorbar()
    pl.title('Mean map')
    
    pl.figure(len(scalar_variables)+2)
    pl.imshow(variance_map[::-1,:], interpolation='nearest')
    pl.colorbar()
    pl.title('Variance map')
    
    pl.figure(len(scalar_variables)+3)
    true_p_map = spherical.mesh_to_map(X, p_true, resolution)
    pl.imshow(true_p_map[::-1,:], interpolation='nearest')
    pl.colorbar()
    pl.title('True map')