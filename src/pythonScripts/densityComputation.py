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
    tree = spatial.KDTree(np.asarray(points))
    neighbors = tree.query_ball_tree(tree, radius)
    return np.array([len(i) for i in neighbors], dtype=np.float64)


def run(poca):
    """New PoCA external-worker API.

    The worker passes a PocaData object exposing named shared-memory NumPy arrays.
    This function uses x/y when only 2D data is available and x/y/z when z is present.
    It returns an action telling PoCA to add a 'density' feature to DetectionSet.
    """
    component_name = "DetectionSet"
    detections = poca.component(component_name)

    x = detections["x"]
    y = detections["y"]

    if "z" in detections:
        z = detections["z"]
        density = computeDensity3D(x, y, z)
        dimension = "3D"
    else:
        density = computeDensity2D(x, y)
        dimension = "2D"

    poca.add_feature(component_name, "density", density)
    poca.display(f"Computed {dimension} density for {density.size} detections with radius={radius}.")
