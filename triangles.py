from stripackd import trmesh, trlist
import stripackd

__all__ = ['triangulate_sphere', 'plot_triangulation']

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
    """
    
    x,y,z = X.T
    lst, lptr, lend, lnew, ier = trmesh(x,y,z)
    if ier != 0:
        raise RuntimeError, 'stripackd.trmesh returned error code %i.'%ier
    
    # stripackd.trprnt(x,y,z,-1,lst,lptr,lend)
    
    # Unwind the fortran linked list
    neighbormap = []
    for i in xrange(n):
        j = lend[i]
        block = []
        while True:
            j = fortan_index(lptr, j)
            block.append(np.abs(fortan_index(lst, j))-1)
            if j==lend[i]:
                neighbormap.append(set(block))
                break
    
    return (neighbormap,) + neighbors2tri(X.shape[0], neighbormap)
    
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
    
    
if __name__ == '__main__':
    import numpy as np
    import pymc as pm
    
    n = 500
    X = np.random.normal(size=(3,n))
    # X[0,:] = X[0,:]**2
    X /= np.sqrt((X**2).sum(axis=0))
    
    neighbors, triangles, trimap = triangulate_sphere(X.T)
    # plot_triangulation(X.T,neighbors)
    
    # plot_triangulation(x,y,z)