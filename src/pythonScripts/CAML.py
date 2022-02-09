# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:04:46 2020

@author: Florian
"""

import os
import numpy as np
import gc
import time
import pickle
import multiprocessing
from joblib import Parallel, delayed
import datetime
import json
import FuncEtc as fn_etc
import FuncDistCalcs as fn_distcalc
import FuncDistCalcs_3D as fn_distcalc_3D
import FuncNormalizeInput as fn_normalize
from keras.models import load_model

# ========================================================================
# Parallel Processing
# ========================================================================
# How many of the system's processors to devote to processing?
# You will want to keep 1 or 2 CPUs free for background tasks to run.
total_cpus = multiprocessing.cpu_count() - 2
# N.B.: If you want to disable parallel processing, set total_cpus = 1

# Processing time estimation.
# The times given for processing a data table here are from my machine, using 
# 10 cores of a Core i7-4930K CPU at 3.40 GHz and Debian Stretch.
# If you have evaluated your own  machine on various sizes of input data, you 
# should be able to fit a quadratic function as a rough estimator of the processing
# time (which is what is used below to give time estimates).
# You can edit your own values to get a better estimage of the times:
# ProcTime = (ProcTimeEst_A * Points^2) + (ProcTimeEst_B * Points) + ProcTimeEst_C
# ProcTime is processing time in seconds, Points is total points in a data file.
# ProcTime includes time taken to load the data, calculate and save distances, and
# save high-resolution preview images.
ProcTimeEst_A = 4.61E-9
ProcTimeEst_B = -2.92E-4
ProcTimeEst_C = 0.0

# ========================================================================
# Cluster forming
# ========================================================================
pred_threshold = 0.5 # points need to be above this score to qualify as clustered (normally 0.5).
# This is only for binary classification models (e.g. clustered or not-clustered)
# For multiple classification, the label with the highest score is assigned to the point.

input_PrepJSON = 'C:/Git/caml-master/CAMLtutorial/AAA Data Descriptions.json'
inputpath = 'C:/Git/caml-master/CAMLtutorial'
current_file = '1_PpMS(100.0)_PpC(20)_PC(50.0)_r(0.0-40.0)_CpMS(2.5)_cellID(1).tsv'
s1_prep_outputpath = 'C:\\Git\\caml-master\\CAMLtutorial\\testFlorian'
model_fname = 'C:\\Git\\caml-master\\CAMLtutorial\\testFlorian\\model\\07VEJJ - Norm-Self Train(500.0k,0.5×Clus) Val(100.0k,0.5×Clus).h5'

def testCAML(infos, xs, ys):
    print("We are here")
    with open(input_PrepJSON, 'r') as file:
        print (xs)
        dataTest = np.array([xs, ys])
        print(dataTest)
            
        ps = {
            "AutoAxes": False,
            "AutoAxesNearest": 1000,
            "ChanIDCol": None,
            "ClosestFriend": 1,
            "ClusMembershipIDCol": 2,
            "FurthestFriend": 100,
            "ImageSize": [
                infos[0],
                infos[1],
                infos[2]
            ],
            "InputFileDelimiter": "\t",
            "InputFileExt": ".tsv",
            "LabelIDCol": 3,
            "SaveImagesForRepeat": 0,
            "UIDCol": None,
            "xCol": 0,
            "xMax": infos[3],
            "xMin": infos[4],
            "yCol": 1,
            "yMax": infos[5],
            "yMin": infos[6]
        }
        
        if(infos[7] == 0):
            ps['zCol'] = None
            ps['zMax'] = None
            ps['zMin'] = None
        else:
            ps['zCol'] = 2
            ps['zMax'] = infos[7]
            ps['zMin'] = infos[8]
        
        print(type(ps))
        
        # check here that the ImageSize is valid. Older versions used a single int
        # and assumed a square 2D field. We can convert those older values here.
        if type(ps['ImageSize']) == int:
            ps['ImageSize'] = [ps['ImageSize'], ps['ImageSize'], 0]
        
        # we'll need to know if we are using 2D or 3D images
        if ps['ImageSize'][2] == 0:
            ps['three_dee'] = False
        else:
            ps['three_dee'] = True
        
        fn_etc.info_msg('Imported JSON variables:')
        print(' │')
        print(' ├─InputFileDelimiter:\t' + ps['InputFileDelimiter'])
        print(' ├─InputFileExt:\t' + ps['InputFileExt'])
        print(' │')
        print(' ├─xCol:\t\t' + str(ps['xCol']))
        print(' ├─yCol:\t\t' + str(ps['yCol']))
        if ps['three_dee']:
            print(' ├─zCol:\t\t' + str(ps['zCol']))
        print(' ├─ClusMembershipIDCol:\t' + str(ps['ClusMembershipIDCol']))
        print(' ├─ChanIDCol:\t\t' + str(ps['ChanIDCol']))
        print(' ├─UIDCol:\t\t' + str(ps['UIDCol']))
        if 'LabelIDCol' in ps:
            print(' ├─LabelIDCol:\t\t' + str(ps['LabelIDCol']))
        print(' │')
        print(' ├─AutoAxes:\t\t' + str(ps['AutoAxes']))
        if ps['AutoAxes']:
            print(' ├─AutoAxesNearest:\t' + str(ps['AutoAxesNearest']))
            print(' ├─ImageSize:\t\tTo be determined')
            print(' ├─xMin:\t\tTo be determined')
            print(' ├─xMax:\t\tTo be determined')
            print(' ├─yMin:\t\tTo be determined')
            print(' ├─yMax:\t\tTo be determined')
            if ps['three_dee']:
                print(' ├─zMax:\t\tTo be determined')
                print(' ├─zMax:\t\tTo be determined')
        else:
            print(' ├─AutoAxesNearest:\tNot applicable')
            print(' ├─ImageSize:\t\t' + str(ps['ImageSize']))
            print(' ├─xMin:\t\t' + str(ps['xMin']))
            print(' ├─xMax:\t\t' + str(ps['xMax']))
            print(' ├─yMin:\t\t' + str(ps['yMin']))
            print(' ├─yMax:\t\t' + str(ps['yMax']))
            if ps['three_dee']:
                print(' ├─zMax:\t\t' + str(ps['zMin']))
                print(' ├─zMax:\t\t' + str(ps['zMax']))
        print(' │')
        print(' ├─ClosestFriend:\t' + str(ps['ClosestFriend']))
        print(' └─FurthestFriend:\t' + str(ps['FurthestFriend']))
        
        # we will add a UID for each point in a file so we can track things across arrays.
        if not ps['UIDCol']:
            doAddUID = True
        else:
            doAddUID = False
            
        # If your data are not consistently located within the same field of view then the
        # image axes can be adjusted to accomodate based on the range of each 
        # image's xy data.
        if ps['AutoAxes']:
            doSetUpAxes = True
        else:
            doSetUpAxes = False
        
        prepType = 'novel'
        ps['ClusMembershipIDCol'] = False
        ps['LabelIDCol'] = False
        
        output_prefix = os.path.splitext(current_file)[0]
        print('Loading data...', end='', flush=True)
        #datatable = np.genfromtxt(os.path.join(inputpath, current_file),
        #                          delimiter=ps['InputFileDelimiter'],
        #                          skip_header=1)
        
        datatable = dataTest.transpose()
        print(datatable)
        
        print(datatable.shape)
        
         # scale for data which is in daft units
        if 'DataScale' in ps:
            datatable[:,ps['xCol']] = ps['DataScale'] * datatable[:,ps['xCol']]
            datatable[:,ps['yCol']] = ps['DataScale'] * datatable[:,ps['yCol']]
        
        if ps['three_dee']:
            datatable[:,ps['zCol']] = ps['DataScale'] * datatable[:,ps['zCol']]
    
        # will be exporting as tab-delimited from here, so swap out original delimiters in the header for tabs
        #with open(os.path.join(inputpath, current_file), 'r') as f:
        #    ps['TableHeaders'] = f.readline().strip()
        #    if ps['InputFileDelimiter'] != '\t':
        #        ps['TableHeaders'] = ps['TableHeaders'].replace(ps['InputFileDelimiter'], '\t')
                
        ps['TableHeaders'] = 'x (nm)\ty (nm)'
    
        TotalPointsThisImage = datatable.shape[0]
        print('Done (' + str(TotalPointsThisImage) + ' points)')
        
        # warn about insufficient points in this image
        if TotalPointsThisImage < ps['FurthestFriend'] - ps['ClosestFriend']:
            fn_etc.warn_msg('This image has ' + str(TotalPointsThisImage) + ' points. Minimum of ' + str(ps['FurthestFriend'] - ps['ClosestFriend'] + 1) + ' points is required, according to the supplied JSON file. Missing neighbours will be padded with nans!')
            
        # auto image boundaries; avoid plotting cropped regions on a full-sized field.
        if doSetUpAxes == True:
            
            # get xy range of the data
            xmin = np.min(datatable[:,ps['xCol']])
            xmax = np.max(datatable[:,ps['xCol']])
            ymin = np.min(datatable[:,ps['yCol']])
            ymax = np.max(datatable[:,ps['yCol']])
            if ps['three_dee']:
                zmin = np.min(datatable[:,ps['zCol']])
                zmax = np.max(datatable[:,ps['zCol']])
    
    #        if xmin < 0:
    #            datatable[:,ps['xCol']] += abs(xmin)
    #            
    #        if ymin < 0:
    #            datatable[:,ps['yCol']] += abs(ymin)
            
            ps['xMin'] = np.floor(xmin / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            ps['xMax'] = np.ceil(xmax / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            ps['yMin'] = np.floor(ymin / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            ps['yMax'] = np.ceil(ymax / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
            if ps['three_dee']:
                ps['zMin'] = np.floor(zmin / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
                ps['zMax'] = np.ceil(zmax / ps['AutoAxesNearest'] ) * ps['AutoAxesNearest']
                ps['ImageSize'] = [ps['xMax'] - ps['xMin'], ps['yMax'] - ps['yMin'], ps['zMax'] - ps['zMin']]
                print('AutoAxes: \t Set image boundaries to [ ' + str(ps['xMin']) + ', ' + str(ps['xMax']) + ', ' + str(ps['yMin']) + ', ' + str(ps['yMax']) + ', ' + str(ps['zMin']) + ', ' + str(ps['zMax']) + ' ]')
            else:
                ps['ImageSize'] = [(ps['xMax'] - ps['xMin'], ps['yMax'] - ps['yMin'])]
                print('AutoAxes: \t Set image boundaries to [ ' + str(ps['xMin']) + ', ' + str(ps['xMax']) + ', ' + str(ps['yMin']) + ', ' + str(ps['yMax']) + ' ]')
            ps['AutoAxes'] = False
        
        #duplicate xy screening
        if ps['three_dee']:
            data_coordsonly = np.concatenate((datatable[:, ps['xCol'], None], datatable[:, ps['yCol'], None], datatable[:, ps['zCol'], None]), axis=1)
        else:
            data_coordsonly = np.concatenate((datatable[:, ps['xCol'], None], datatable[:, ps['yCol'], None]), axis=1)
        
        _, uniq_idx = np.unique(data_coordsonly, axis=0, return_index=True) # get the unique rows
        
        if uniq_idx.shape[0] < datatable.shape[0]:
            uniq_idx = np.sort(uniq_idx)
            datatable = datatable[uniq_idx,:]
            
            oldTotalPoints = TotalPointsThisImage
            TotalPointsThisImage = datatable.shape[0]
            DuplicatePointsRemoved = oldTotalPoints - TotalPointsThisImage
            
            if not doAddUID:
                doAddUID = True
                
            fn_etc.info_msg('Checked for duplicate points and removed ' + str(DuplicatePointsRemoved) + ' identical points.')
        else:
            DuplicatePointsRemoved = 0
            print('Checked for duplicate points and none were found')
        del data_coordsonly
        
        if doAddUID:
            ps['UIDCol'] = np.shape(datatable)[1]  # will be with zero ref
            datatable = np.insert(datatable,
                                  ps['UIDCol'],
                                  list(range(1, len(datatable)+1)),
                                  axis=1)
            ps['TableHeaders'] = ps['TableHeaders'] + '\tpointUID'
            # print('UIDCol:\tAdded Unique ID to column ' + str(ps['UIDCol']))
            
        # Very rough estimate of processing time -- this will vary with every machine
        # and depend on how many CPU cores you employ.
        # once you have a few files done you can work out some rough params and put
        # them in at the top as ProcTimeEst_ABC for your own edification...
        GuessProcTime = ((ProcTimeEst_A * TotalPointsThisImage * TotalPointsThisImage) + (ProcTimeEst_B * TotalPointsThisImage) + ProcTimeEst_C) 
        if GuessProcTime > 30:
            FancyProcTime = datetime.timedelta(seconds=np.ceil(GuessProcTime))
            print('Total of ' + str(TotalPointsThisImage) + ' points is expected to take ' + str(FancyProcTime) + ' to calculate distances.') # + GuessFinishTime.strftime('%H:%M:%S on %d %B'))
        else:
            print('Total of ' + str(TotalPointsThisImage) + ' points.')
        free_space = fn_etc.get_free_space(s1_prep_outputpath)
        GuessMemoryRequirements = (( (TotalPointsThisImage * (ps['FurthestFriend'] - ps['ClosestFriend'] + 1) * (datatable.shape[1] + 1)) * len(ps['ImageSize']) * 64) + (datatable.size * 64)) / 8  # float64 bytes required
        print(fn_etc.convert_size(GuessMemoryRequirements) + ' is required to store this image\'s xy data and distance measurements (' + fn_etc.convert_size(free_space) + ' available space)')       
        if GuessMemoryRequirements > free_space:
            print('\x1b[1;35;43m' + '\tPROBLEM!\t' + '\x1b[1;37;45m' + '\tInsufficient storage space...\t' + '\x1b[0m', flush=True)
            # raise ValueError('This images requires ' + fn_etc.convert_size(GuessMemoryRequirements) + ' to store distance measurements and there is ' + fn_etc.convert_size(free_space) + ' remaining on the drive containing the output folder:\n' + s1_prep_outputpath)
            print('\x1b[1;33;40m' + 'Free up at least ' + fn_etc.convert_size(GuessMemoryRequirements - free_space) + ' to continue processing this image.\nBe aware that subsequent images in the queue may also require additional storage space.' + '\x1b[0m', flush=True)
            ask_prepType = fn_etc.askforinput(
                            message = 'Enter Y when you have confirmed there is more space...',
                            errormessage= 'Press Y and press enter to continue\nor type Ctrl C to cancel processing here.',
                            defaultval= '',
                            isvalid = lambda v : v in ['Y','y','Yes','yes'])
    
        # Set up a memory mapped file to hold a copy of datatable. You can switch this
        # out to a regular in-memory variable if you wish but things can bog down
        # when large datatables are loaded. If you work with MemMaps then you at least
        # have a good excuse to buy some fast PCIe or SSD storage, right?
        datatable_mmap_partname = output_prefix + '_Data.MemMap'
        ps['datatable_mmap_fname'] = datatable_mmap_partname
        datatable_mmap_fname = os.path.join(s1_prep_outputpath, datatable_mmap_partname)
        
        datatable_mmapped = np.memmap(
                                     datatable_mmap_fname, 
                                     dtype='float64', 
                                     shape=datatable.shape, 
                                     mode='w+')
        datatable_mmapped[:] = datatable
        
        del datatable    # free up memory by dropping the original datatable
        _ = gc.collect() # and forcing garbage collection
    
        # Pre-allocate another writeable shared memory map files to contain 
        # the ouput of the parallel distance-to-NNs computation
        dists_mmap_partname = output_prefix + '_D[NN' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + ']_Dists.MemMap'
        ps['dists_mmap_fname'] = dists_mmap_partname
        dists_mmap_fname = os.path.join(s1_prep_outputpath, dists_mmap_partname)
        dists_mmapped = np.memmap(
                                 dists_mmap_fname, 
                                 dtype=datatable_mmapped.dtype, 
                                 shape=(TotalPointsThisImage, ps['FurthestFriend'] - ps['ClosestFriend'] + 1, datatable_mmapped.shape[1] + 1), 
                                 mode='w+')
        #
        # perform the distance measurements.
        #
        # Output is 3D array with:
        #    each point (dim 0)
        #    it's FurthestFriend NN UIDs (dim 1)
        #    and the measurements (distances, matching x,y,etc from original data) (dim 2)
        # distances_only = dists_mmapped[:,:,0]
    
        start_time = time.time()        
        # If you want to do single-thread processing do this:
        for i in range(TotalPointsThisImage):
            fn_distcalc.dnns_v3(datatable_mmapped, datatable_mmapped, ps, dists_mmapped, i)
            if np.mod(i, 1000) == 0:
                elapsed_time = time.time() - start_time
                print(str(i) + ' done - ' + str(round(elapsed_time,2)) + ' sec')
        # Otherwise do this line for parallel processing
        #if ps['three_dee']:
        #    Parallel(n_jobs=total_cpus, verbose=3)(delayed(fn_distcalc_3D.dnns_v3)(datatable_mmapped, datatable_mmapped, ps, dists_mmapped, i) for i in range(TotalPointsThisImage))
        #else:
        #    Parallel(n_jobs=total_cpus, verbose=3)(delayed(fn_distcalc.dnns_v3)(datatable_mmapped, datatable_mmapped, ps, dists_mmapped, i) for i in range(TotalPointsThisImage))
        # Update and Dump ProcSettings as JSON
        ps['DistsDumpShape'] = dists_mmapped.shape
        ps['DataDumpShape'] = datatable_mmapped.shape
        ps['FilePrefix'] = output_prefix
        
        #json_fname = os.path.join(s1_prep_outputpath, output_prefix + '_dists[NN' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + ']-ProcSettings.json')
        #with open(json_fname, 'w') as file:
        #     file.write(json.dumps(ps, indent=4))
        #elapsed_time = time.time() - start_time
        print('Time for ' + str(TotalPointsThisImage) + ' points was ' + str(round(elapsed_time,2)) + ' seconds (' + str(round(TotalPointsThisImage / elapsed_time,2)) + ' points per second.)')
        
        model = load_model(model_fname)
        model_ID = os.path.basename(model_fname).split()[0]
        print('Loaded Model ' + model_ID + '\t(You can safely ignore warnings above about \'No training configuration found\')')
        model_config = model.get_config()
        RequiredInputSize = model_config['layers'][0]['config']['batch_input_shape'][1]
        ModelLabelsTotal = model_config['layers'][-1]['config']['units']
        if ModelLabelsTotal == 1:
            ModelLabelsTotal = 2
        print('This model expects input from ' + str(RequiredInputSize) + ' near-neighbours.')
        
        # extract the model ID
        model_name = os.path.splitext(os.path.basename(os.path.normpath(model_fname)))[0]
        inputpath_novel = s1_prep_outputpath
        
        # extract the preparation type required for this model from the model_name
        DistancesProcessedAs = model_name.split()[2]
        ModelShortName = model_name.split()[0]
        outputpath_novel = os.path.abspath(os.path.join(inputpath_novel, '4_evaluated_by_' + ModelShortName))
        datatable_called_fname = os.path.join(outputpath_novel, ps['FilePrefix'] + '_DataCalled' + ps['InputFileExt'])
        
        #make the folder for the output data
        if not os.path.exists(outputpath_novel):
            os.makedirs(outputpath_novel)
        
        Dists_all_New = dists_mmapped
        TotalPointsThisImage = Dists_all_New.shape[0]
        print('Total of ' + str(TotalPointsThisImage) + ' points to evaluate.')
        
        # check that we are using the right model for this data
        TotalNeighboursThisImage = Dists_all_New.shape[1]
        if TotalNeighboursThisImage != RequiredInputSize:
            raise ValueError('Your input data uses ' + str(TotalNeighboursThisImage) + ' near-neighbour values but Model ' + model_ID + ' is expecting ' + str(RequiredInputSize) + ' near-neighbour values.\r\nPlease use a model which matches the input data size.')
    
        # ========================================================================
        #  Model Preparation: Turn raw input (distances) into input for the model
        # ========================================================================
        
        X_novel_distances_raw = np.array(Dists_all_New[:,:,0], dtype='float32') # convert from float64 to float32 to ease memory requirements            
        X_novel = fn_normalize.normalize_dists(X_novel_distances_raw, TotalPointsThisImage, DistancesProcessedAs)
        if ModelLabelsTotal == 2:
            X_novel = X_novel.reshape((X_novel.shape[0],X_novel.shape[1], 1)) # Reshape the data to be repeats/measures/features for LSTM
        
        _ = gc.collect()
        
        ### make predictions
        modeleval_time_init = time.time()
        
        novel_probabilities = model.predict(X_novel, batch_size=64, verbose=1)       # probability assessment of event being in cluster(~1) or not (~0)
        if ModelLabelsTotal == 2:
            novel_predictions = [float(np.round(x - (pred_threshold - 0.5))) for x in novel_probabilities]   # convert probability into boolean by pessimistic threshold ... must be >pred_threshold to qualify as 1
        else:
            novel_predictions = novel_probabilities.argmax(axis=-1) # convert model's probabilities to labels
        
        # at this stage we should delete and salted points which have been labelled as clustered
        # idx_salt = 
        # idx_clustered = 
        # idx_salty_clustered = 
        # novel_probabilities = not salty_clustered
        # novel_predictions = not salty_clustered
        
        modeleval_time = time.time() - modeleval_time_init
        print('Time with model: ' + str(round(modeleval_time,2)) + ' seconds for ' + str(TotalPointsThisImage) + ' points.')
        
        pickle.dump((novel_probabilities, novel_predictions),open(os.path.join(outputpath_novel, ps['FilePrefix'] + '_ML_calls[NN' + str(ps['ClosestFriend']) + '-' + str(ps['FurthestFriend']) + '].pkl'), 'wb'))
        
        _ = gc.collect()
        
        # load from memmap file
        import_datatable_f = os.path.join(inputpath_novel, ps['datatable_mmap_fname'])
        if os.path.isfile(import_datatable_f):
            datatable_mmapped = np.memmap(import_datatable_f, dtype='float64', shape=tuple(ps['DataDumpShape']), mode='r')
        
        datatable_called = np.concatenate((datatable_mmapped, novel_probabilities,np.array(novel_predictions).reshape(TotalPointsThisImage,1)),axis=1)
        ps['TableHeaders'] = ps['TableHeaders'] + ps['InputFileDelimiter'] + 'score' + ps['InputFileDelimiter'] + 'label (' + model_ID + ')'
        
        scores_col = datatable_called.shape[1] - 2
        labels_col = datatable_called.shape[1] - 1
        
        # some stats about this thing
        TotalType0Points = sum([idx == 0 for idx in novel_predictions])
        TotalType1Points = sum([idx == 1 for idx in novel_predictions])
        if ModelLabelsTotal == 3:
            TotalType2Points = sum([idx == 2 for idx in novel_predictions])
            PercentClustered = ((TotalType1Points + TotalType2Points) / TotalPointsThisImage) * 100
        else:
            PercentClustered = ((TotalType1Points) / TotalPointsThisImage) * 100
        
        print('Model ' + model_ID + ' thinks this file has ' + str(round(PercentClustered, 2)) + ' percent of points in clusters')
        if ModelLabelsTotal == 3:
            print('\t' + str(round(TotalType1Points/TotalPointsThisImage*100, 2)) + ' percent of points are Type 1')
            print('\t' + str(round(TotalType2Points/TotalPointsThisImage*100, 2)) + ' percent of points are Type 2')
        
        print (datatable_called)
        print (datatable_called.ndim)
        print (datatable_called.shape)
        print (novel_predictions)
            
        print(datatable_called_fname)
        np.savetxt(datatable_called_fname, datatable_called, delimiter=ps['InputFileDelimiter'], fmt='%10.5f', header=ps['TableHeaders'], comments='')
        print('Done.')
        
        return np.array(novel_predictions)


#infos = np.array([1000, 1000, 0, 1000, 0, 1000, 0, 0, 0])        
#xs = np.random.rand(4000)
#xs = xs * 1000
#print (xs)
#ys = np.random.rand(4000)
#ys = ys * 1000
#print (ys)
#print(testCAML(infos, xs, ys))