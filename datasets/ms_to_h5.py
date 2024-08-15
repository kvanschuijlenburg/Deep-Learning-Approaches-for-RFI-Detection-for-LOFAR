import math
import os

import casacore.tables as tables
import numpy as np
import tables as tb
from tqdm import tqdm

measurementSetLocation = './data/LOFAR_L2014581 (recording)/ms'
datasetLocation = './data/LOFAR_L2014581 (recording)/h5'

polarizations = ["XX", "YY", "XY", "YX"]
batchSize = 500

def get_directory_size(directory_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

def msToh5(msFileName, h5FileName):
    # open measurement set
    ms = tables.table(msFileName, ack=False)
    #print(ms.colnames())

    # Get some information about the observation
    antenna_name = []
    antenna_pos = []
    for antenna in ms.ANTENNA:
        antenna_name.append(antenna['NAME'])
        antenna_pos.append(antenna['POSITION'])
    channel_frequencies = ms.SPECTRAL_WINDOW[0]['CHAN_FREQ']

    time = ms.getcol('TIME')
    antennas1 = ms.getcol("ANTENNA1")
    antennas2 = ms.getcol("ANTENNA2")
    uvw = ms.getcol('UVW')

    # Create a new h5 file after loading data to reduce the probability of an empty file will be created by memory issues
    h5file = tb.open_file(h5FileName, 'w',verbose=False)

    # Create a group for the metadata and observations
    metadata_group = h5file.create_group('/', 'Metadata', 'Metadata Information')
    observations_group = h5file.create_group('/', 'Observations', 'Obervations Data')

    # Store the metadata
    h5file.create_array(metadata_group, 'CHAN_FREQ', channel_frequencies)
    h5file.create_array(metadata_group, 'ANTENNA_NAME', antenna_name)
    h5file.create_array(metadata_group, 'POSITION', antenna_pos)

    dataShape = ms.getcol("DATA",0,1)[0].shape
    flagShape = ms.getcol('FLAG',0,1)[0].shape
    
    elementsCount = np.zeros((len(antenna_name),len(antenna_name),5),dtype=np.uint32)
    for antenna1, name1 in enumerate(antenna_name):
        antenna1Group = h5file.create_group(observations_group, f'{name1}', f'Observations for {name1}')
        for antenna2, name2 in enumerate(antenna_name):
            if antenna1 >= antenna2: continue
            correlation_group = h5file.create_group(antenna1Group, f'{name2}', f'Observations for {name2}')

            # Retrieve indices of the rows corresponding to this antenna correlation           
            correlation_idx = (antennas1==antenna1) & (antennas2==antenna2)
            
            # Since time is a small array, it can alreay be loaded not in batches
            correlation_time = time[correlation_idx]
            h5file.create_array(correlation_group, 'Time', correlation_time)

            correlationUvw = uvw[correlation_idx]
            h5file.create_array(correlation_group, 'uvw', correlationUvw)
            
            # Get recording height and width
            nTimeSteps = len(correlation_time)
            nChannels = flagShape[0]
            
            # create arrays for the flags and correlations
            h5file.create_array(correlation_group, 'Flags',shape=(nChannels,nTimeSteps),atom=tb.BoolAtom())#, np.transpose(correlationFlags))
            for polarization_index in range(len(polarizations)): 
                h5file.create_array(correlation_group,polarizations[polarization_index],shape=(nChannels,nTimeSteps),atom=tb.ComplexAtom(8))#.Complex64Atom())#,polarization)
    
    rowsInDataset = len(antennas1) #int((len(antennas1)-1)*len(antennas1)/2)
    batchSize = 2000000
    if batchSize ==-1:
        nSteps = 1
        batchSize = rowsInDataset
    else:
        nSteps = math.ceil(rowsInDataset/batchSize)

    for batchIndex in tqdm(range(nSteps), desc="Batch progress", leave=False):
        batchStart = batchIndex*batchSize
        batchStop = min(batchStart+batchSize, rowsInDataset)

        # Find which antenna pairs must be processed
        batchSum = 0
        antennaPairsInBatch = []
        for antenna1, name1 in enumerate(antenna_name):
            for antenna2, name2 in enumerate(antenna_name):
                if antenna1 >= antenna2: continue
                antennasMatch = (antennas1[batchStart:batchStop]==antenna1) & (antennas2[batchStart:batchStop]==antenna2)
                if np.any(antennasMatch):
                    antennaPairsInBatch.append([antenna1, antenna2])

        if len(antennaPairsInBatch)>0:
            # Load observations from the measurement sets, for this batch
            data = ms.getcol("DATA",batchStart,batchSize)
            flags = ms.getcol('FLAG',batchStart,batchSize)    

        # Only process antenna pairs occuring in the batch
        for [antenna1, antenna2] in tqdm(antennaPairsInBatch, desc="Antenna progress", leave=False):
            name1 = antenna_name[antenna1]
            name2 = antenna_name[antenna2]

            correlation_group = h5file.get_node('/Observations/'+name1+'/'+name2)
            correlation_idx = (antennas1[batchStart:batchStop]==antenna1) & (antennas2[batchStart:batchStop]==antenna2)                
            
            # Verify that the flags are equal for all dimensions, because they should. Then copy flags
            for flagDimension in range(3):                    
                equalArrays = flags[:,:,flagDimension] == flags[:,:,flagDimension+1]
                allEqual = np.all(equalArrays)
                if not allEqual:
                    raise Exception("Flags in different channels are different!")   
            
            # Retrieve the flags of the correlation of the batch, and the corresponding h5 dataset
            correlationFlags = flags[correlation_idx,:,0]
            flagsDatset= h5file.get_node(correlation_group,'/Flags')
            
            # Get the current number of elements in this flag set, and write the flag values
            flagCounts = elementsCount[antenna1, antenna2, 4]
            flagsDatset[:, flagCounts:flagCounts+correlationFlags.shape[0]]=np.transpose(correlationFlags)
            elementsCount[antenna1, antenna2, 4] += correlationFlags.shape[0]

            # For the correlation, calculate from all polarizations the amplitude and phase
            for polarization_index in range(len(polarizations)): # 0 for XX, 1 for XY, 2 for YX, 3 for YY
                polarizationCount = elementsCount[antenna1, antenna2, polarization_index]
                
                polarization = np.transpose(data[correlation_idx, :, polarization_index])
                polarizationDataset = h5file.get_node(correlation_group, '/'+polarizations[polarization_index])
                polarizationDataset[:,polarizationCount:polarizationCount+polarization.shape[1]]=polarization
                elementsCount[antenna1, antenna2, polarization_index] += polarization.shape[1]


    # Close the PyTables HDF5 file
    h5file.close()

def convertDirectory(msDir=measurementSetLocation, h5Dir=datasetLocation):
    os.makedirs(h5Dir, exist_ok=True)
    measurementSets = [folder for folder in os.listdir(msDir) if folder.endswith('.ms')]
    h5Sets = [filename for filename in os.listdir(h5Dir) if filename.endswith('.h5')]

    total_size = 0
    convertSets = []
    for measurementSet in measurementSets:
        h5Name = measurementSet.split('.ms')[0] + '.h5'
        if h5Name in h5Sets:
            print("H5 set for measurement set %s already exist, skip it." % measurementSet)
            continue
        convertSets.append(measurementSet)
        total_size += get_directory_size(msDir + measurementSet)

    mainProgressBar = tqdm(total=total_size, desc="Main progress")
    for measurementSet in convertSets:
        h5Name = measurementSet.split('.ms')[0] + '.h5'
        msToh5(msDir + measurementSet, h5Dir + h5Name)
        setSize = get_directory_size(msDir + measurementSet)
        mainProgressBar.update(setSize)

if __name__ == '__main__':
    convertDirectory(measurementSetLocation, datasetLocation)