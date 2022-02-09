#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates Euclidean distances for points up to the specified number of nearby
neighbours.

Uses local sub-regions, memory-mapping, and chunking of data to avoid memory
blowouts.

Can be parallelized to speed up processing.

"""
import numpy as np
from scipy.spatial.distance import cdist


def RegionCropper(InputTable, RegionBounds, DataCols=[3, 4, 5]):
    """
    Crops x, xy, or xyz data in 1, 2, or 3 dimensions

    @params:
        InputTable    - Required  : coordinate data (numpy array)
        RegionBounds  - Required  : limits (List)
        DataCols      - Required  : location of x, y, z columns in the input
                                    array (List)
    """
    RegBndsSize = len(RegionBounds)

    if RegBndsSize == 2 and len(DataCols) == 1:
        # crop by 1 dim
        OutputTable = InputTable[
                                 (RegionBounds[1] > InputTable[:, DataCols[0]]) &
                                 (InputTable[:, DataCols[0]] > RegionBounds[0])
                                 ]
    elif RegBndsSize == 4 and len(DataCols) == 2:
        # crop by 2 dim
        OutputTable = InputTable[
                                 (RegionBounds[1] > InputTable[:, DataCols[0]]) &
                                 (InputTable[:, DataCols[0]] > RegionBounds[0]) &
                                 (RegionBounds[3] > InputTable[:, DataCols[1]]) &
                                 (InputTable[:, DataCols[1]] > RegionBounds[2])
                                 ]
    elif RegBndsSize == 6 and len(DataCols) == 3:
        # crop by 3 dim
        OutputTable = InputTable[
                                 (RegionBounds[1] > InputTable[:, DataCols[0]]) &
                                 (InputTable[:, DataCols[0]] > RegionBounds[0]) &
                                 (RegionBounds[3] > InputTable[:, DataCols[1]]) &
                                 (InputTable[:, DataCols[1]] > RegionBounds[2]) &
                                 (RegionBounds[5] > InputTable[:, DataCols[2]]) &
                                 (InputTable[:, DataCols[2]] > RegionBounds[4])
                                 ]
    else:
        # something amiss! error!
        print('Problem with size of RegionBounds or DataCols!')
        OutputTable = []
    return OutputTable


def dnns_v3(inputdata1, inputdata2, ps, output_filename, i):
    """
    Compute the distances for the i-th inputdata1 point to all nearest-neighbour 
    points in inputdata2. Nearest neighbours are assesed from ps['ClosestFriend'] 
    up to ps['FurthestFriend'], inclusive.
    
    This functions returns an array N (points) by M (columns of the original input 
    data) by Z (total neighbours).
    
    Reading along Axis 0, e.g. output[n,:,:] gives the data for nth point. 
    
    Each NN is in each row as distance-to-NNth, NNth's row from the original data 
    table (with it's UID in the last column).
    
    """
    
    # Expects data as [Xcoord, Ycoord, UID]
    TotalColumns = np.shape(inputdata1)[1]
    inputdata1 = inputdata1[i, :].reshape(1, TotalColumns)

    if inputdata2.shape[0] < ps['FurthestFriend']:
#        print('Input data has fewer points (' + str(inputdata2.shape[0]) + ') than the furthest requested distance measurement (' + str(ps['FurthestFriend']) + ')')
        maxNeighbours = inputdata2.shape[0] - 1
        padNeighbours = ps['FurthestFriend'] - inputdata2.shape[0] + 1
        TestRegionSize = ps['ImageSize'][0]
    else:
        maxNeighbours = ps['FurthestFriend']
        padNeighbours = 0
        TestRegionSize = np.ceil(np.cbrt(maxNeighbours / (inputdata2.shape[0] / ( 0.3 * ps['ImageSize'][0] * ps['ImageSize'][1] * ps['ImageSize'][2] )))) # Guesstimate a starting region size based on the density and assuming the points are only in about 1/3 of the available area
        
    UseableRegion = False

    while not UseableRegion:
        TestRegBounds = [np.floor(inputdata1[0, ps['xCol']] - TestRegionSize),
                         np.ceil(inputdata1[0, ps['xCol']] + TestRegionSize),
                         np.floor(inputdata1[0, ps['yCol']] - TestRegionSize),
                         np.ceil(inputdata1[0, ps['yCol']] + TestRegionSize),
                         np.floor(inputdata1[0, ps['zCol']] - TestRegionSize),
                         np.ceil(inputdata1[0, ps['zCol']] + TestRegionSize)]
        TestRegion = RegionCropper(inputdata2,
                                   TestRegBounds,
                                   [ps['xCol'], ps['yCol'], ps['zCol']])
        if np.shape(TestRegion)[0] >= maxNeighbours + 1:
            UseableRegion = True
            # Have a useable region but enlarge it by >= cbrt(3) to make sure
            # we are capturing all the true NNs and not just corner-dense
            # events.
            TestRegionSize = 1.5 * TestRegionSize
            TestRegBounds = [inputdata1[0, ps['xCol']] - TestRegionSize,
                             inputdata1[0, ps['xCol']] + TestRegionSize,
                             inputdata1[0, ps['yCol']] - TestRegionSize,
                             inputdata1[0, ps['yCol']] + TestRegionSize,
                             inputdata1[0, ps['zCol']] - TestRegionSize,
                             inputdata1[0, ps['zCol']] + TestRegionSize]
            TestRegion = RegionCropper(inputdata2,
                                       TestRegBounds,
                                       [ps['xCol'], ps['yCol'], ps['zCol']])
        else:
            # Enlarging TestRegionSize to get more points
            TestRegionSize = 2 * TestRegionSize
    
    # calculate the distances
    dists = cdist(inputdata1[:, [ps['xCol'], ps['yCol'], ps['zCol']]],TestRegion[:, [ps['xCol'], ps['yCol'], ps['zCol']]]).T
    
    # append the UID for each NN point
    dists = np.concatenate((dists,TestRegion), axis=1)
    
    # sort by the distance column (column 0)
    dists = dists[dists[:, 0].argsort()]
    
    if padNeighbours > 0:
        # add the missing NN distances as nans.s
        dists = np.concatenate((dists, np.full((padNeighbours, dists.shape[1]), np.nan)), axis=0)
        
    # export this point's results for insertion into the main output
    output_filename[i, :, :] = dists[ps['ClosestFriend']:ps['FurthestFriend'] + 1, :]