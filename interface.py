"High-level interface to multivariate normal variables. Retargetable linear algebra backend."
import numpy as np
import pymc as pm

# TODO: Conditional versions of all.

def sparse_MVN(backend):
    return pm.stochastic_from_dist('SparseMVN_%s'%backend.__name__, backend.mvn_logp, backend.rmvn, mv=True)