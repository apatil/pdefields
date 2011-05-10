"High-level interface to multivariate normal variables. Retargetable linear algebra backend."
import numpy as np
import pymc as pm

# TODO: Conditional versions of all.

def mvn_logp(x,M,precision_products,backend):
    ""
    return backend.mvn_logp(x,M,**precision_products)
    
def rmvn(M,precision_products,backend):
    return backend.rmvn(M,**precision_products)

SparseMVN = pm.stochastic_from_dist('SparseMVN', mvn_logp, rmvn, mv=True)

