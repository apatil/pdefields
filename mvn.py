"High-level interface to multivariate normal variables. Retargetable linear algebra backend."

import numpy as np

def rmvn(M,precision_products,backend):
    return backend.rmvn(M,**precision_products)
    
def mvn_logp(x,M,precision_products,backend):
    return backend.mvn_logp(x,M,**precision_products)