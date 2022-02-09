import numpy as np
from scipy.linalg import eig, inv
from scipy.spatial.transform import Rotation as Rot

#least squares fit to a 3D-ellipsoid
#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
#
# Note that sometimes it is expressed as a solution to
#  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
# where the last six terms have a factor of 2 in them
# This is in anticipation of forming a matrix with the polynomial coefficients.
# Those terms with factors of 2 are all off diagonal elements.  These contribute
# two terms when multiplied out (symmetric) so would need to be divided by two

# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes

def data_regularize(data, type="spherical", divs=10):
    limits = np.array([
        [min(data[:, 0]), max(data[:, 0])],
        [min(data[:, 1]), max(data[:, 1])],
        [min(data[:, 2]), max(data[:, 2])]])
        
    regularized = []

    if type == "cubic": # take mean from points in the cube
        
        X = np.linspace(*limits[0], num=divs)
        Y = np.linspace(*limits[1], num=divs)
        Z = np.linspace(*limits[2], num=divs)

        for i in range(divs-1):
            for j in range(divs-1):
                for k in range(divs-1):
                    points_in_sector = []
                    for point in data:
                        if (point[0] >= X[i] and point[0] < X[i+1] and
                                point[1] >= Y[j] and point[1] < Y[j+1] and
                                point[2] >= Z[k] and point[2] < Z[k+1]):
                            points_in_sector.append(point)
                    if len(points_in_sector) > 0:
                        regularized.append(np.mean(np.array(points_in_sector), axis=0))

    elif type == "spherical": #take mean from points in the sector
        divs_u = divs 
        divs_v = divs * 2

        center = np.array([
            0.5 * (limits[0, 0] + limits[0, 1]),
            0.5 * (limits[1, 0] + limits[1, 1]),
            0.5 * (limits[2, 0] + limits[2, 1])])
        d_c = data - center
    
        #spherical coordinates around center
        r_s = np.sqrt(d_c[:, 0]**2. + d_c[:, 1]**2. + d_c[:, 2]**2.)
        d_s = np.array([
            r_s,
            np.arccos(d_c[:, 2] / r_s),
            np.arctan2(d_c[:, 1], d_c[:, 0])]).T

        u = np.linspace(0, np.pi, num=divs_u)
        v = np.linspace(-np.pi, np.pi, num=divs_v)

        for i in range(divs_u - 1):
            for j in range(divs_v - 1):
                points_in_sector = []
                for k, point in enumerate(d_s):
                    if (point[1] >= u[i] and point[1] < u[i + 1] and
                            point[2] >= v[j] and point[2] < v[j + 1]):
                        points_in_sector.append(data[k])

                if len(points_in_sector) > 0:
                    regularized.append(np.mean(np.array(points_in_sector), axis=0))
# Other strategy of finding mean values in sectors
#                    p_sec = np.array(points_in_sector)
#                    R = np.mean(p_sec[:,0])
#                    U = (u[i] + u[i+1])*0.5
#                    V = (v[j] + v[j+1])*0.5
#                    x = R*math.sin(U)*math.cos(V)
#                    y = R*math.sin(U)*math.sin(V)
#                    z = R*math.cos(U)
#                    regularized.append(center + np.array([x,y,z]))
    return np.array(regularized)



def ellipsoid_fit2(x, y, z):
    data = np.array([x, y, z])
    data2 = data_regularize(data)
    return ellipsoid_fit2(data)
    
def ellipsoid_fit(x,y,z):
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)
    
    print(radii)
    print(center)
    print(evecs)
    
    r = Rot.from_dcm([[evecs[0,0],evecs[0,1],evecs[0,2]],
                   [evecs[1,0],evecs[1,1],evecs[1,2]],
                   [evecs[2,0],evecs[2,1],evecs[2,2]]])
    #print(r.as_euler('zyx', degrees=True))

    coeffs = np.asarray(center)
    coeffs = np.append(coeffs, radii)
    coeffs = np.append(coeffs, np.asarray([evecs[0,0],evecs[0,1],evecs[0,2]]))
    coeffs = np.append(coeffs, np.asarray([evecs[1,0],evecs[1,1],evecs[1,2]]))
    coeffs = np.append(coeffs, np.asarray([evecs[2,0],evecs[2,1],evecs[2,2]]))
    return coeffs

def ls_ellipsoid(xx,yy,zz):

   #ellipsoid_fit(xx,yy,zz)
   
   # change xx from vector of length N to Nx1 matrix so we can use hstack
   x = xx[:,np.newaxis]
   y = yy[:,np.newaxis]
   z = zz[:,np.newaxis]

   #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
   J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
   K = np.ones_like(x) #column of ones

   #np.hstack performs a loop over all samples and creates
   #a row in J for each x,y,z sample:
   # J[ix,0] = x[ix]*x[ix]
   # J[ix,1] = y[ix]*y[ix]
   # etc.

   JT=J.transpose()
   JTJ = np.dot(JT,J)
   InvJTJ=np.linalg.inv(JTJ);
   ABC= np.dot(InvJTJ, np.dot(JT,K))

# Rearrange, move the 1 to the other side
#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
#    or
#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
#  where J = -1
   eansa=np.append(ABC,-1)
   
   #print("Polynomial coeffs")
   #print(eansa)
   center,axes,inve = polyToParams3D(eansa, False)
   printAns3D(center, axes, inve, xx,yy,zz, True)

#   return (eansa)
   print(axes)
   coeffs = np.asarray(center)
   coeffs = np.append(coeffs, axes)
   coeffs = np.append(coeffs, inve)
   return coeffs
   
def polyToParams3D(vec,printMe):

   # convert the polynomial form of the 3D-ellipsoid to parameters
   # center, axes, and transformation matrix
   # vec is the vector whose elements are the polynomial
   # coefficients A..J
   # returns (center, axes, rotation matrix)

   #Algebraic form: X.T * Amat * X --> polynomial form

   if printMe: print ('\npolynomial\n',vec)

   Amat=np.array(
   [
   [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
   [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
   [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
   [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
   ])

   if printMe: print ('\nAlgebraic form of polynomial\n',Amat)

   #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
   # equation 20 for the following method for finding the center
   A3=Amat[0:3,0:3]
   A3inv=inv(A3)
   ofs=vec[6:9]/2.0
   center=-np.dot(A3inv,ofs)
   if printMe: print ('\nCenter at:',center)

   # Center the ellipsoid at the origin
   Tofs=np.eye(4)
   Tofs[3,0:3]=center
   R = np.dot(Tofs,np.dot(Amat,Tofs.T))
   if printMe: print ('\nAlgebraic form translated to center\n',R,'\n')

   R3=R[0:3,0:3]
   R3test=R3/R3[0,0]
   if printMe: print ('normed \n',R3test)
   s1=-R[3, 3]
   R3S=R3/s1
   (el,ec)=eig(R3S)

   recip=1.0/np.abs(el)
   axes=np.sqrt(recip)
   if printMe: print ('\nAxes are\n',axes  ,'\n')

   inve=inv(ec) #inverse is actually the transpose here
   if printMe: print ('\nRotation matrix\n',inve)
   return (center,axes,inve)
   
def printAns3D(center,axes,R,xin,yin,zin,verbose):

      if verbose: print ("\nCenter at  %10.4f,%10.4f,%10.4f" % (center[0],center[1],center[2]))
      if verbose: print ("Axes gains %10.4f,%10.4f,%10.4f " % (axes[0],axes[1],axes[2]))
      if verbose: print ("Rotation Matrix\n%10.5f,%10.5f,%10.5f\n%10.5f,%10.5f,%10.5f\n%10.5f,%10.5f,%10.5f" % (
      R[0,0],R[0,1],R[0,2],R[1,0],R[1,1],R[1,2],R[2,0],R[2,1],R[2,2]))
      
      r = Rot.from_dcm([[R[0,0],R[0,1],R[0,2]],
                   [R[1,0],R[1,1],R[1,2]],
                   [R[2,0],R[2,1],R[2,2]]])


      # Check solution
      # Convert to unit sphere centered at origin
      #  1) Subtract off center
      #  2) Rotate points so bulges are aligned with axes (no xy,xz,yz terms)
      #  3) Scale the points by the inverse of the axes gains
      #  4) Back rotate
      # Rotations and gains are collected into single transformation matrix M

      # subtract the offset so ellipsoid is centered at origin
      xc=xin-center[0]
      yc=yin-center[1]
      zc=zin-center[2]

      # create transformation matrix
      L = np.diag([1/axes[0],1/axes[1],1/axes[2]])
      M=np.dot(R.T,np.dot(L,R))
      #if verbose: print ('\nTransformation Matrix\n',M)
      
       # apply the transformation matrix
      [xm,ym,zm]=np.dot(M,[xc,yc,zc])
      # Calculate distance from origin for each point (ideal = 1.0)
      rm = np.sqrt(xm*xm + ym*ym + zm*zm)

      #if verbose: print ("\nAverage Radius  %10.4f (truth is 1.0)" % (np.mean(rm)))
      #if verbose: print ("Stdev of Radius %10.4f\n " % (np.std(rm)))