import numpy as np
import scipy.spatial as spatial
radius = 50.0

def computeDensity2D(xs, ys):
    points = np.vstack((xs, ys)).T
    return computeDensity(points)

def computeDensity3D(xs, ys, zs):
    points = np.vstack((xs, ys, zs)).T
    return computeDensity(points)
    
def computeDensity(points):
    tree = spatial.KDTree(np.array(points))
    neighbors = tree.query_ball_tree(tree, radius)
    frequency = np.array([len(i) for i in neighbors],dtype=np.float64)
    return frequency