# -*- coding: UTF-8 -*-
"Includes functions to compute the needed inner product matrices on triangulated 2d manifolds."

# TODO: This all needs to be changed to allow for spatially varying parameters, 
# and for matrix coefficients stuck into the differential operators.
# Also, need to figure out how to handle 'internal boundaries.'

import numpy as np
from scipy.sparse import lil_matrix

__all__ = ['triangle_area', 'Ctilde', 'C', 'G']

def get_edges(vertices, triangle):
    v = [vertices[i] for i in triangle]
    edges = np.array([v[2]-v[1], v[0]-v[2], v[1]-v[0]])
    return edges

def triangle_area(vertices, triangle):
    edges = get_edges(vertices, triangle)
    return (np.cross(edges[0], edges[1])**2).sum()/2.

def Ctilde(vertices, triangles, triangle_areas):
    u"Returns the diagonal of the <ψ_i, 1> matrix as a vector."
    out = np.empty(len(vertices))
    for t,a in zip(triangles, triangle_areas):
        for i in t:
            out[i] += a/3
    return out

def C(vertices, triangles, triangle_areas):
    u"Returns the <ψ_i, ψ_j> matrix as a SciPy csr matrix."
    out = lil_matrix((len(vertices), len(vertices)))
    for t,a in zip(triangles, triangle_areas):
        for i in t:
            for j in t:
                out[i,j] += a/12
            out[i,i] += a/12
    return out.tocsr()

def G(vertices, triangles, triangle_areas):
    u"Returns the <∇ ψ_i, ∇ ψ_j> matrix as a SciPy CSR matrix."
    out = lil_matrix((len(vertices), len(vertices)))
    for t,a in zip(triangles, triangle_areas):
        e = get_edges(vertices, t)
        m = np.dot(e,e.T)/4/a
        for mi,i in enumerate(t):
            for mj,j in enumerate(t):
                out[i,j] += m[mi,mj]
    return out.tocsr()