import numpy as np
from math import sqrt
#	fit a sphere to X,Y, and Z data points
#	returns the radius and center points of
#	the best fit sphere
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = np.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = sqrt(t)

    C = C.ravel()
    C[3] = radius
    return C
    
#   project the point (x, y, z) to the sphere (x0, y0, z0) of radius r1
def get_spherical_pcd(x,y,z,x0,y0,z0,r1):
    x2 = x
    y2 = y
    z2 = z
    x0 = x0 * np.ones(x2.size)
    y0 = y0 * np.ones(y2.size)
    z0 = z0 * np.ones(z2.size)
    r1 = r1 * np.ones(x2.size)
    r2 = np.power(np.square(x2-x0)+np.square(y2-y0)+np.square(z2-z0),0.5)
    xr = x0 + (r1/r2) * (x2-x0)
    yr = y0 + (r1/r2) * (y2-y0)
    zr = z0 + (r1/r2) * (z2-z0)
    return xr, yr, zr
    
def sphericalProj(params, xs, ys, zs):
    x0 = params[0]
    y0 = params[1]
    z0 = params[2]
    r1 = params[3]
    points = np.array([])
    for x,y,z in np.nditer([xs,ys,zs]):
        xr,yr,zr = get_spherical_pcd(x,y,z,x0,y0,z0,r1)
        #print(x, ", ", y, ", ", z, " -> ", xr, ", ", yr, ", ", zr, end='\n')
        points = np.append(points, [xr,yr,zr])
    return points