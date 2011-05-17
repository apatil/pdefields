# Triangulation

import numpy as np
import pymc as pm
from pdefields import pymc_objects, operators, algorithms
from pdefields.backends import cholmod
from pdefields.manifolds import spherical
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
    amp = pm.Exponential('amp', .0001, value=100)

    # A constant mean.
    m = pm.Uninformative('m',value=0)
    
    @pm.deterministic(trace=False)
    def M(m=m,n=len(X)):
        """The mean vector"""
        return np.ones(n)*m
        
    @pm.deterministic(trace=False)
    def Q(kappa=kappa, alpha=alpha, amp=amp, Ctilde=Ctilde, G=G, backend=backend):
        "The precision matrix."
        out = operators.mod_frac_laplacian_precision(Ctilde, G, kappa, alpha, backend)/np.asscalar(amp)**2
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
    empirical_S = pm.logit((k+1)/(N+2.))
    S=pymc_objects.SparseMVN('S',M, precision_products, backend, value=empirical_S)
    
    @pm.deterministic(trace=False)
    def p(S=S):
        """The success probability."""
        return pm.invlogit(S)

    # The data.
    data = pm.Binomial('data', n=N, p=p, value=k, observed=True)
    
    # A Fortran representation of the likelihood, to allow for fast Metropolis steps without querying data.logp.
    likelihood_variables = np.vstack((np.resize(N,k.shape),k)).T
    likelihood_string = """
    lkp = dexp({X})/(1.0D0+dexp({X}))
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
    M = pm.MCMC(make_model(N,k,X,cholmod,spherical),db='hdf5')
    scalar_variables = filter(lambda x:not x.observed, [M.m, M.amp, M.kappa])
    if len(scalar_variables)>0:    
        M.use_step_method(pm.AdaptiveMetropolis, scalar_variables)
    # Comment to use the default AdaptiveMetropolis step method.
    # GMRFMetropolis kind of scales better to high dimensions, but may mix worse in low.
    M.use_step_method(pymc_objects.GMRFMetropolis, M.S, M.likelihood_string, M.M, M.Q, M.likelihood_variables, n_sweeps=100)

    M.isample(1000,0,10)
    
    ################################
    # Visualize the results
    ################################
    
    # Traces of the scalar variables.
    import pylab as pl
    pl.close('all')
    for v in scalar_variables:
        pm.Matplot.plot(v)
        
    # Make mean and variance maps
    burn = 0
    thin = 1
    resolution = 501
    m1 = np.zeros((resolution/2+1,resolution))
    m2 = np.zeros((resolution/2+1,resolution))
    nmaps = 0
    for i in xrange(burn, M._cur_trace_index, thin):
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
    pl.imshow(mean_map[::-1,:], interpolation='nearest',vmin=0,vmax=1)
    pl.colorbar()
    pl.title('Mean map')
    
    pl.figure(len(scalar_variables)+2)
    pl.imshow(variance_map[::-1,:], interpolation='nearest',vmin=0,vmax=1)
    pl.colorbar()
    pl.title('Variance map')
    
    pl.figure(len(scalar_variables)+3)
    true_p_map = spherical.mesh_to_map(X, p_true, resolution)
    pl.imshow(true_p_map[::-1,:], interpolation='nearest',vmin=0,vmax=1)
    pl.colorbar()
    pl.title('True map')
    
    # Performance profile, 10 iterations.
    # n = 250:
    # ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    #     10    0.049    0.005    0.071    0.007 algorithms.py:77(fast_metropolis_sweep)
    #   1010    0.015    0.000    0.015    0.000 {method 'normal' of 'mtrand.RandomState' objects}
    #   1010    0.006    0.000    0.006    0.000 {method 'random_sample' of 'mtrand.RandomState' objects}
    #     11    0.006    0.001    0.009    0.001 {method 'cholesky' of 'scikits.sparse.cholmod.Factor' objects}
    #     99    0.003    0.000    0.007    0.000 compressed.py:101(check_format)
    #     99    0.002    0.000    0.003    0.000 compressed.py:622(prune)
    #     11    0.002    0.000    0.002    0.000 {_csr.csr_sort_indices}
    #    753    0.002    0.000    0.002    0.000 {numpy.core.multiarray.array}
    #     22    0.002    0.000    0.002    0.000 {_csc.csc_matmat_pass2}
    # n = 2500:
    # ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    #     10    0.524    0.052    0.702    0.070 algorithms.py:77(fast_metropolis_sweep)
    #     11    0.216    0.020    0.240    0.022 {method 'cholesky' of 'scikits.sparse.cholmod.Factor' objects}
    #   1010    0.136    0.000    0.136    0.000 {method 'normal' of 'mtrand.RandomState' objects}
    #   1010    0.039    0.000    0.039    0.000 {method 'random_sample' of 'mtrand.RandomState' objects}
    #     22    0.024    0.001    0.024    0.001 {_csc.csc_matmat_pass2}
    #     11    0.023    0.002    0.023    0.002 {_csr.csr_sort_indices}
    #     22    0.010    0.000    0.010    0.000 {_csc.csc_matmat_pass1}
    # n = 10000:
    #  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    #      10    1.927    0.193    2.603    0.260 algorithms.py:77(fast_metropolis_sweep)
    #      10    1.078    0.108    1.163    0.116 {method 'cholesky' of 'scikits.sparse.cholmod.Factor' objects}
    #    1010    0.524    0.001    0.524    0.001 {method 'normal' of 'mtrand.RandomState' objects}
    #    1010    0.143    0.000    0.143    0.000 {method 'random_sample' of 'mtrand.RandomState' objects}
    #      20    0.098    0.005    0.098    0.005 {_csc.csc_matmat_pass2}
    #      10    0.080    0.008    0.080    0.008 {_csr.csr_sort_indices}
    #      20    0.044    0.002    0.044    0.002 {_csc.csc_matmat_pass1}
    # 155/106    0.029    0.000    1.487    0.014 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    #      30    0.022    0.001    0.022    0.001 {method 'astype' of 'numpy.ndarray' objects}
    #      10    0.021    0.002    0.021    0.002 {method 'D' of 'scikits.sparse.cholmod.Factor' objects}
    #      19    0.020    0.001    0.020    0.001 {_csc.csc_matvec}
    # n = 25000 doesn't work with a RAM db, and also doesn't work with hdf5 because of https://github.com/pymc-devs/pymc/issues/41. However, here are the results from just stepping sequentially.
    #  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    #      10    6.133    0.613    7.735    0.774 algorithms.py:77(fast_metropolis_sweep)
    #      10    4.077    0.408    4.293    0.429 {method 'cholesky' of 'scikits.sparse.cholmod.Factor' objects}
    #    1010    1.205    0.001    1.205    0.001 {method 'normal' of 'mtrand.RandomState' objects}
    #    1010    0.366    0.000    0.366    0.000 {method 'random_sample' of 'mtrand.RandomState' objects}
    #      20    0.247    0.012    0.247    0.012 {_csc.csc_matmat_pass2}
    #      10    0.200    0.020    0.200    0.020 {_csr.csr_sort_indices}
    #      20    0.114    0.006    0.114    0.006 {_csc.csc_matmat_pass1}
    # 148/100    0.083    0.001    5.117    0.051 {method 'get' of 'pymc.LazyFunction.LazyFunction' objects}
    #      30    0.075    0.003    0.075    0.003 {method 'astype' of 'numpy.ndarray' objects}
    #      19    0.055    0.003    0.055    0.003 {_csc.csc_matvec}