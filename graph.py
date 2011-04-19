from stripackd import trmesh, trlist
import stripackd

def trilist2trimap(n, trilist):
    "Converts the list of triangles to a convenient node-to-triangle map."
    trimap = dict([(i,[]) for i in xrange(n)])
    for t in trilist:
        for i in t:
            trimap[i].append(t)
    return trimap

def neighborlist2neighbormap(n, neighborlist):
    return neighborlist

def neighbors2tri(n, neighbors):
    "Neighbors should be a list of n arrays. The ith element contains the neighbors of node i, whose index is > i."
    triangles = []
    for i in xrange(n):
        i_edges = neighbors[i]
        for j in xrange(i+1, n):
            if j in neighbors[i]:
                shared_endpts = i_edges & neighbors[j]
                triangles = triangles + [(i,j,s) for s in shared_endpts]
                
    return neigborlist2neighbormap(neighbors), triangles, trilist2trimap(n, triangles)

def fortan_index(a, i):
    return a[i-1]
def triangulate_sphere(X):
    """
    X are the x,y,z coordinates of the points to triangulate on the unit sphere.

    Returns: Map from vertex to neigbors, list of triangles, map from vertex to all adjacent triangles.
    """
    x,y,z=X[:,0],X[:,1],X[:,2]
    lst, lptr, lend, lnew, ier = trmesh(x,y,z)
    if ier != 0:
        raise RuntimeError, 'stripackd.trmesh returned error code %i.'%ier
    
    stripackd.trprnt(x,y,z,-1,lst,lptr,lend)
    
    print lend
    print lptr
    # print lnew
        
    # lptr = np.abs(lptr) - 1
    # lend = np.abs(lend) - 1
    # lst = np.abs(lst) - 1
    # lnew = lnew-1
    
    j = 1
    blocks = []
    block = [j]
    i=0
    for i in xrange(n):
        j = lend[i]
        block = []
        while True:
            j = fortan_index(lptr, j)
            block.append(np.abs(fortan_index(lst, j)))
            if j==lend[i]:
                blocks.append(block)
                break

    print blocks
    # neighbors = []
    # j = 0
    # for i in xrange(n):
    #     while True:
    #         j = lptr[j]
    #         neighbors.append([i,lst[j]])
    #         print j, lend[i]
    #         if j==lend[i]:
    #             print i, lend[i]
    #             break
    
    neighbors = np.array(neighbors)
    
    return neighbors2tri(X.shape[0], neighbors)
    
def plot_triangulation(neighbors):
    """
    Generates and visualizes the triangulation using mplot3d.
    """
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n = X.shape[0]
    
    x,y,z = X.T
    
    for frm, to in map(tuple, neighbors):
        ax.plot([x[frm], x[to]], [y[frm], y[to]], [z[frm], z[to]], 'k-')
            
    ax.plot(x,y,z,'r.')

def get_neighboring_triangles(index, vert):
    pass
    
    
if __name__ == '__main__':
    import numpy as np
    import pymc as pm
    
    n = 5
    X = np.random.normal(size=(3,n))
    # X[0,:] = X[0,:]**2
    X /= np.sqrt((X**2).sum(axis=0))
    
    vert, tri = triangulate_sphere(X.T)
    neighbors, triangles, trimap = vert2tri(vert)
    
    # plot_triangulation(x,y,z)