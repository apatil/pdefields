# -*- coding: UTF-8 -*-
"""
Includes functions to compute the needed inner product matrices on triangulated 2d manifolds. The SciPy sparse matrix package contains several storage types, with easy. conversion between. The compressed sparse row and compressed sparse column support relatively efficient matrix operations and are closest to the representation used by packages like CHOLMOD, but the lil_matrix format supports efficient random access and most importantly efficient changes to sparsity pattern. So I use the LIL version when assembling the C and G matrices, then downstream code can 'freeze' them as appropriate.
"""

# TODO: This all needs to be changed to allow for spatially varying parameters, 
# and for matrix coefficients stuck into the differential operators.
# Also, need to figure out how to handle 'internal boundaries.'

import numpy as np

from scipy.sparse import lil_matrix

__all__ = ['triangle_area', 'Ctilde', 'C', 'G']

def get_edges(vertices, triangle):
    """
    Takes the vertices as an n X 3 array of coordinates, and a triangle as a 3-array of indices (each in [0,n-1]) and returns the edges of the triangle as vectors from zero stored in a 3 X 3 array.
    """
    v = [vertices[i] for i in triangle]
    edges = np.array([v[2]-v[1], v[0]-v[2], v[1]-v[0]])
    return edges

def triangle_area(vertices, triangle):
    """
    Takes the vertices as an n X 3 array of coordinates, and a triangle as a 3-array of indices (each in [0,n-1]) and returns the area of the triangle.
    """
    edges = get_edges(vertices, triangle)
    return np.sqrt((np.cross(edges[0], edges[1])**2).sum())/2.

def Ctilde(vertices, triangles, triangle_areas):
    u"Takes the vertices as an n X 3 array of coordinates, the triangles as an m X 3 array of indices (each in [0,n-1]) and the triangle areas as an m-array and returns the diagonal of the <ψ_i, 1> matrix in sparse lil_matrix format, where ψ_i is the finite element basis function at vertex i."
    out = lil_matrix((len(vertices), len(vertices)))
    for t,a in zip(triangles, triangle_areas):
        for i in t:
            out[i,i] += a/3
    return out

def C(vertices, triangles, triangle_areas):
    u"Takes the vertices as an n X 3 array of coordinates, the triangles as an m X 3 array of indices (each in [0,n-1]) and the triangle areas as an m-array and returns the <ψ_i, ψ_j> matrix in sparse lil_matrix format, where ψ_i is the finite element basis function at vertex i."
    out = lil_matrix((len(vertices), len(vertices)))
    for t,a in zip(triangles, triangle_areas):
        for i in t:
            for j in t:
                out[i,j] += a/12
            out[i,i] += a/12
    return out

def G(vertices, triangles, triangle_areas):
    u"Takes the vertices as an n X 3 array of coordinates, the triangles as an m X 3 array of indices (each in [0,n-1]) and the triangle areas as an m-array and returns the <∇ ψ_i, ∇ ψ_j> matrix in sparse lil_matrix format, where ψ_i is the finite element basis function at vertex i."
    out = lil_matrix((len(vertices), len(vertices)))
    for t,a in zip(triangles, triangle_areas):
        e = get_edges(vertices, t)
        m = np.dot(e,e.T)/4/a
        for mi,i in enumerate(t):
            for mj,j in enumerate(t):
                out[i,j] += m[mi,mj]
    return out