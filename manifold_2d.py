# -*- coding: UTF-8 -*-
"Includes functions to compute the needed inner product matrices on triangulated 2d manifolds."

# FIXME: This all needs to be changed to allow for spatially varying parameters, 
# and for matrix coefficients stuck into the differential operators.

import numpy as np
from scipy.sparse import lil_matrix

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
        e = get_edges[vertices, t]
        m = np.dot(e,e.T)/4/a
        for i in t:
            for j in t:
                out[i,j] += m[i,j]
    return out.tocsr()
    
def B(vertices, triangles, triangle_areas, boundary_nodes):
    u"Returns the <ψ_i, ∂_n ψ_j> matrix as a SciPy CSR matrix. Is only nonzero when ψ_j has boundary edges, I think."
    out = lil_matrix((len(vertices), len(vertices)))
    for t,a in zip(triangles, triangle_areas):
        e = get_edges[vertices, t]
        m1 = np.bmat([[0,e[0],e[0]],[e[1],0,e[1]],[e[2],e[2],0]])
        
        i3 = np.eye(3)
        b_ = [i in boundary_nodes for i in t]
        b = [b_[2] and b_[1], b_[0] and b_[2], b_[1] and b_[0]]
        m2 = np.bmat([[i3*b[0]],[i3*b[1]],[i3*b[2]]])
        
        m = -1/(4*a)*m1.T*m2*e.T
        for i in t:
            for j in t:
                out[i,j] += m[i,j]
    return out.tocsr()

def B_val(vertices, triangles, triangle_areas, boundary_nodes):
    "???"
    out = lil_matrix((len(vertices), len(vertices)))
    for t,a in zip(triangles, triangle_areas):
        e = get_edges[vertices, t]
        isbound = []
        raise NotImplementedError
    return out.tocsr()