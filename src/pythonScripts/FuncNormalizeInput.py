#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nearest-neighbour distances need to be prepared before being used as input to
train or use models.

Normally this is some kind of normalization and various options are handled by 
this script, referred to as PreparationType:

'Raw-Distances'              The absolute NN distances
'Norm-Distances-Only'        l2 normalisation of the raw distances

'Raw-Differences'            The difference between each consective NN distance
'Diff-Norm'                  Consecutive differences are calculated and then normalised (using l2 norm). Values are then rescaled by 100000
'Norm-Diff'                  Default normalisation (l2 norm, i.e. sqrt(sum(abs(X).^2)) ) is applied. Values are rescaled by 100000 and the consecutive differences taken.
'Norm-Diffs-Only-NoRescale'  As above but without rescaling the norm values.

'Norm-Self'                  Normalize each point's distance-differences to the min/max for that point (rather than normalize to the entire set)
'Norm-FirstDiff'             The first (nearest) distance is used to normalise the other distance-differences
'Norm-MinDiff'               The minimum distance-hop is used to normalise the other distance-differences
'Norm-MaxDiff'               The maximum distance-hop is used to normalise the other distance-differences

'CoordsOnly'                 Data are xy(z) coordinates, normalize to 0-1 by largest dimension (we assume each is of the same scale)

'd2NN_SelfNormalized'        Not yet implemented.

@author: dave
"""

import numpy as np
from sklearn import preprocessing
import os
import tempfile
import shutil

def normalize_dists(X_distances_raw, total_X_distances, PreparationType):
    
    # ========================================================================
    #  Model Preparation: Turn raw input (distances) into model input-data
    # ========================================================================
    
    # points to process in one go, only implemented for Norm-Self at this stage
    ProcAsChunks = False
    chunk_width = 100000 
    
    if X_distances_raw.shape[0] > 5 * chunk_width:
        ProcAsChunks = True
        chunk_idx = list(np.arange(0,X_distances_raw.shape[0], chunk_width))
        print('Large matrix - using MemMap on disk (may be slower to process).')
        
    if ProcAsChunks:
        temp_folder_name = tempfile.mkdtemp()
        temp_data_fname = os.path.join(temp_folder_name, 'temp.mmap')
        if os.path.exists(temp_data_fname): 
            os.unlink(temp_data_fname) # Check for existing temp file and delete it if it exists

        X_data_normalized = np.memmap(
                                     temp_data_fname, 
                                     dtype='float32', 
                                     shape=X_distances_raw.shape, 
                                     mode='w+')
    else:
        X_data_normalized = np.zeros(X_distances_raw.shape)

    if PreparationType == 'Norm-Self':
        
        # ========================================================================
        #  Model Prep: Difference --> Normalise to the first NN-Distance
        # ========================================================================
        
        # copy the first-NN distances as they are ...
        X_data_normalized[:, 0] = X_distances_raw[:,0]
        
        # ... and take the distance-differences for the rest
        if ProcAsChunks:
            for chunk in chunk_idx:
                X_data_normalized[chunk:chunk + chunk_width, 1:] = np.diff(X_distances_raw[chunk:chunk + chunk_width,:], axis=1)
        else:
            X_data_normalized[:, 1:] = np.diff(X_distances_raw, axis=1)

        # Normalize each event to its min/max
        X_train_min = X_data_normalized.min(axis=1)
        X_train_max = X_data_normalized.max(axis=1)
        X_data_range = X_train_max - X_train_min 

        if ProcAsChunks:
            for chunk in chunk_idx:
                X_data_normalized[chunk:chunk + chunk_width, :] = (X_data_normalized[chunk:chunk + chunk_width] - X_train_min[chunk:chunk + chunk_width, None]) / X_data_range[chunk:chunk + chunk_width, None]
        else:
            X_data_normalized = (X_data_normalized - X_train_min[:,None]) / X_data_range[:,None]
        
        del(X_train_min, X_train_max, X_data_range)

    elif PreparationType == 'Norm-FirstDiff':
        
        # ========================================================================
        #  Model Prep: Difference --> Normalise to the first NN-Distance
        # ========================================================================

        X_distance_differences = np.diff(X_distances_raw, axis=1)
        
        # Add back the first column (distance from self to 1st neighbour)
        X_data_concat = np.concatenate((X_distances_raw[:,0].reshape(total_X_distances,1),X_distance_differences), axis=1)

        # Normalize to the first non-zero distance value NB: there should be no zero values!
        X_firstNN_dists = X_data_concat[:,0]
        X_data_normalized = X_data_concat / X_firstNN_dists[:,None]
    
    elif PreparationType == 'Norm-MaxDiff':
        
        # ========================================================================
        #  Model Prep: Difference --> Normalise to the maximum Distance-Difference (largest hop)
        # ========================================================================

        X_distance_differences = np.diff(X_distances_raw, axis=1)

        # Add back the first column (distance from self to 1st neighbour)
        X_data_normalized = np.concatenate((X_distances_raw[:,0].reshape(total_X_distances,1),X_distance_differences), axis=1)

        # Normalize to the maximum distance value
        X_data_normalized = X_data_normalized / X_data_normalized.max(axis=0)

    elif PreparationType == 'Norm-MinDiff':
        
        # ========================================================================
        #  Model Prep: Difference --> Normalise to the minimum Distance-Difference (smallest hop)
        # ========================================================================

        X_distance_differences = np.diff(X_distances_raw, axis=1)
        
        # Add back the first column (distance from self to 1st neighbour)
        X_data_normalized = np.concatenate((X_distances_raw[:,0].reshape(total_X_distances,1),X_distance_differences), axis=1)
    
        # Normalize to the minimum distance value NB: there should be no zero values!
        X_data_normalized = X_data_normalized / np.min(X_data_normalized[np.nonzero(X_data_normalized)], axis=0) # TODO
    
    elif PreparationType == 'Norm-Diff':
        
        # ========================================================================
        #  Model Prep: Normalise --> Rescale --> Difference --> Addback 1st dNN
        # ========================================================================

        # Normalise the raw distances
        X_dists_normalized = preprocessing.normalize(X_distances_raw)

        # Rescale the normalised values
        X_dists_normd_rescaled = np.array(X_dists_normalized*100000, np.int32)
        
        # Take the differences of the rescaled, normalised distances
        X_dists_normd_scaled_diffs = np.diff(X_dists_normd_rescaled, axis=1)

        # Add back the first column (normalised & rescaled distance from self to 1st neighbour)
        X_data_normalized = np.concatenate((X_dists_normd_rescaled[:,0].reshape(total_X_distances,1),X_dists_normd_scaled_diffs), axis=1)   

    elif PreparationType == 'Diff-Norm':
    
        # ========================================================================
        #  Model Prep: Difference --> Normalise --> Rescale --> Addback 1st dNN
        # ========================================================================

        # Take the differences of the raw distances
        X_distance_differences = np.diff(X_distances_raw, axis=1)

        # Add back the first column (distance from self to 1st neighbour)
        X_dists_diffs_addback = np.concatenate((X_distances_raw[:,0].reshape(total_X_distances,1),X_distance_differences), axis=1)

        # Normalise the distance-differences
        X_dists_diffs_normalized = preprocessing.normalize(X_dists_diffs_addback)

        # Rescale the normalised values
        X_data_normalized = np.array(X_dists_diffs_normalized*100000, np.int32)

    elif PreparationType == 'Raw-Differences':
        # ========================================================================
        #  Model Prep: Difference --> Addback 1st dNN
        # ========================================================================

        # Get the distance-differences
        X_distance_differences = np.diff(X_distances_raw, axis=1)

        # Add back the first column (distance from self to 1st neighbour)
        X_data_normalized = np.concatenate((X_distances_raw[:,0].reshape(total_X_distances,1),X_distance_differences), axis=1)

    elif PreparationType == 'Norm-Diffs-Only-NoRescale':
        # ========================================================================
        #  Model Prep: Difference --> Addback 1st --> Normalize (No rescaling)
        # ========================================================================

        X_distance_differences = np.diff(X_distances_raw, axis=1)

        # Add back the first column (distance from self to 1st neighbour)
        X_dists_diffs_normalized = np.concatenate((X_distances_raw[:,0].reshape(total_X_distances,1),X_distance_differences), axis=1)

        # Normalize all dist-diffs to the whole set
        X_data_normalized = preprocessing.normalize(X_dists_diffs_normalized)

    elif PreparationType == 'Raw-Distances':
        # ========================================================================
        #  Model Prep: Raw Distances to neighbours --> Nothing else
        # ========================================================================

        # Use the raw distances as they are, i.e. return the input un modified
        X_data_normalized = X_distances_raw

    elif PreparationType == 'Norm-Distances-Only':
        # ========================================================================
        #  Model Prep: Raw Distances to neighbours --> Normalize
        # ========================================================================

        # Normalize just the raw distances to the entire set
        X_data_normalized = preprocessing.normalize(X_distances_raw)

    elif PreparationType == 'CoordsOnly':
        # ========================================================================
        #  Model Prep: XY coords --> Nothing else
        # ========================================================================

        # We already normalized the xycoords during step 2 so we don't do anything
        # further to the data; just send it back as it is.
        X_data_normalized = X_distances_raw

#    elif PreparationType == 'd2NN_SelfNormalized':
#        # ========================================================================
#        #  Model Prep: Sum Raw distance to NN10, normalise to local density
#        # ========================================================================
#        #
#        # Prepare the D2NN(10) values
#        # TODO - finish and fix this
##        sumNNs = 10
##        SumNNdists = np.sum(X_distances_raw[ps['ClosesestFriend']:sumNNs,:])

    else:
        raise ValueError('Unsure how to prepare the input data. Check PreparationType setting...')

    if ProcAsChunks:  
        shutil.rmtree(temp_folder_name)

    return X_data_normalized
