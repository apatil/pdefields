"High-level interface to multivariate normal variables. Retargetable linear algebra backend."
import numpy as np
import pymc as pm

# TODO: Conditional versions of all.

def mvn_logp(x,M,precision_products,backend):
    "Takes a candidate value x, a mean vector M, the products returned by backend.precision_products as a map, and the linear algebra backend module. Passes the arguments to backend.mvn_logp and returns the log-probability of x given M and the precision matrix represented in precision_products."
    return backend.mvn_logp(x,M,**precision_products)
    
def rmvn(M,precision_products,backend):
    "Takes a mean vector M, the products returned by backend.precision_products as a map, and the linear algebra backend module. Passes the arguments to backend.rmvn and returns a random draw x from the multivariate normal variable with mean M and the precision matrix represented in precision_products."
    return backend.rmvn(M,**precision_products)

SparseMVN = pm.stochastic_from_dist('SparseMVN', mvn_logp, rmvn, mv=True)

