"""
Produces triangulated meshes on spheres using stripackd, 
http://orion.math.iastate.edu/burkardt/f_src/stripack/stripackd.f90
"""

from stripackd import trmesh, bnodes
from manifold_2d import *
import numpy as np

def fortan_index(a, i):
    return a[i-1]
    
def trilist2trimap(n, trilist):
    "Converts the list of triangles to a convenient node-to-triangle map."
    trimap = [[] for i in xrange(n)]
    for t in trilist:
        for i in t:
            trimap[i].append(t)
    return trimap

def neighbors2tri(n, neighbors):
    "Neighbors should be a list of n arrays. The ith element contains the neighbors of node i, whose index is > i."
    triangles = []
    for i in xrange(n):
        i_edges = neighbors[i]
        for j in neighbors[i]:
            if j > i:
                shared_endpts = np.array(list(i_edges & neighbors[j]))
                triangles = triangles + [(i,j,s) for s in shared_endpts[np.where(shared_endpts>j)]]

    return triangles, trilist2trimap(n, triangles)

def triangulate_sphere(X):
    """
    X are the x,y,z coordinates of the points to triangulate on the unit sphere.

    Returns: 
        - List of neighbors of each vertex
        - List of triangles represented as tuples of three indices
        - List of all triangles adjacent to each vertex.
        - List of indices of boundary vertices.
    """
    
    x,y,z = X.T
    lst, lptr, lend, lnew, ier = trmesh(x,y,z)
    if ier != 0:
        raise RuntimeError, 'stripackd.trmesh returned error code %i.'%ier
    
    # stripackd.trprnt(x,y,z,-1,lst,lptr,lend)
    
    # Unwind the fortran linked list
    neighbormap = []
    boundary = []
    for i in xrange(X.shape[0]):
        j = lend[i]
        block = []
        while True:
            j = fortan_index(lptr, j)
            block.append(np.abs(fortan_index(lst, j))-1)
            if j==lend[i]:
                neighbormap.append(set(block))
                if fortan_index(lst, j)<0:
                    boundary.append(i)
                break
    
    triangles, trimap = neighbors2tri(X.shape[0], neighbormap)

    return neighbormap, triangles, trimap, boundary
    
def plot_triangulation(X,neighbors):
    """
    Generates and visualizes the triangulation using mplot3d.
    """
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n = X.shape[0]
    
    x,y,z = X.T
    
    for frm,n in enumerate(neighbors):
        for to in n:
            ax.plot([x[frm], x[to]], [y[frm], y[to]], [z[frm], z[to]], 'k-')
            
    ax.plot(x,y,z,'r.')
    
def mesh_to_map(X, values, n=101):
    from scipy.interpolate import griddata
    theta = np.arctan2(X[:,1],X[:,0])
    phi = np.arctan2(X[:,2], np.sqrt(X[:,0]**2+X[:,1]**2))
    # define grid.
    xi = np.linspace(-np.pi,np.pi,n)
    yi = np.linspace(-np.pi/2., np.pi/2., n)
    # grid the data.
    zi = griddata((theta, phi), values, (xi[None,:], yi[:,None]), method='cubic')
    return np.ma.masked_array(zi,mask=np.isnan(zi))