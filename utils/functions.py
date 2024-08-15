import os
import pickle
import random
import warnings

import h5py
import numpy as np
from tqdm import tqdm

import utils as utils

plotsLocation = "./plots"
modelsLocation = "./models"
datasetsLocation = "./data"


def getMeasurementSetLocation(datasetName):
    datasetLocation = os.path.join(datasetsLocation,datasetName,'ms')
    if not os.path.exists(datasetLocation): 
        raise Exception("dataset directory %s does not exist" %datasetLocation)
    return datasetLocation

def getH5SetLocation(datasetName):
    datasetLocation = os.path.join(datasetsLocation,datasetName,'h5')
    if not os.path.exists(datasetLocation): os.makedirs(datasetLocation)
    return datasetLocation
    
def getDatasetLocation(observationName, fileName = None, subdir=None):
    if subdir is None:
        datasetLocation = os.path.join(datasetsLocation,observationName,'cache')
    else:
        datasetLocation = os.path.join(datasetsLocation,observationName,'cache', subdir)
    if not os.path.exists(datasetLocation): os.makedirs(datasetLocation)
    
    if not fileName is None:
        datasetLocation = os.path.join(datasetLocation,fileName)
        
    return datasetLocation

def getPlotLocation(datasetName=None, subfolderName = None, plotRoot = plotsLocation):
    if datasetName is None:
        plotLocation = plotRoot
    else:
        plotLocation = os.path.join(plotRoot,datasetName)
    if subfolderName is not None:
        plotLocation = os.path.join(plotLocation,subfolderName)
    os.makedirs(plotLocation, exist_ok=True)
    return plotLocation

def getModelLocation(modelName, subfolderName = None, modelRoot = modelsLocation):       
    os.makedirs(modelRoot, exist_ok=True)
    if subfolderName is None:
        modelSaveDir = os.path.join(modelRoot,modelName)
        logDir = os.path.join(modelRoot,modelName)
    else:
        modelSaveDir = os.path.join(modelRoot,modelName,subfolderName)
        logDir = os.path.join(modelRoot,modelName,subfolderName)

    os.makedirs(modelSaveDir, exist_ok=True)
    os.makedirs(logDir, exist_ok=True)
    return modelSaveDir, logDir

# Embedding
def GetEmbeddingFilename(modelDir, modelType, nFeaturesOrEpoch, valData = False):
    # Find the file location of the embeddings
    if valData: 
        embeddingPath = os.path.join(modelDir, 'embedding_val')
    else:
        embeddingPath = os.path.join(modelDir, 'embedding')

    if modelType == 'dlModel':
        embeddingFile = os.path.join(embeddingPath, 'embedding_epoch={}.pkl'.format(nFeaturesOrEpoch))
    else:
        embeddingFile = os.path.join(embeddingPath, 'embedding_nFeatures={}.pkl'.format(nFeaturesOrEpoch))
        
    return embeddingFile
 
def SaveEmbedding(filename:str, embedding, nFeaturesOrEpoch:int, samplesHash:str):
    embeddingData = [nFeaturesOrEpoch, embedding, samplesHash]
    with open(filename, 'wb') as file:
        pickle.dump(embeddingData, file)

def LoadEmbedding(filename:str, validateHash=None):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        savedHash = None
        if len(data) == 2:
            if isinstance(data[0], int) and isinstance(data[1], np.ndarray):
                nFeaturesOrEpoch, embedding = data
                savedHash = 0
        if savedHash is None:
            nFeaturesOrEpoch, embedding, savedHash = data
    if validateHash is not None:
        assert validateHash == savedHash, "Samples hash does not match. Embedding is probably from different data."
    return nFeaturesOrEpoch, embedding, savedHash

def EmbeddingVerifyAddHash(samplesHash:str, embeddingDir=None,embeddingFile = None):
    if embeddingDir is None and embeddingFile is None:
        raise Exception("No embedding directory or file provided")
    if embeddingDir is not None and embeddingFile is not None:
        raise Exception("Both embedding directory and file provided")

    if embeddingDir is not None:
        embeddingFiles = os.listdir(embeddingDir)
    else:
        embeddingFiles = [embeddingFile]

    if len(embeddingFiles) == 0:
        raise Exception("No embedding files found in {}".format(embeddingDir))
    
    # Loop over all embedding files and perform clustering search
    for embeddingFile in embeddingFiles:
        if embeddingFile is None:
            embeddingFilename = os.path.join(embeddingDir, embeddingFile)
        else:
            embeddingFilename = embeddingFile

        with open(embeddingFilename, 'rb') as file:
            embeddingData = pickle.load(file)
        if len(embeddingData) == 2:
            nFeaturesOrEpoch, embedding = embeddingData
            print("Update embedding file {}".format(embeddingFilename))
            utils.functions.SaveEmbedding(embeddingFilename, embedding, nFeaturesOrEpoch, samplesHash)
        else:
            print("Hash existed for embedding file {}".format(embeddingFilename))
            nFeaturesOrEpoch, embedding, savedHash = embeddingData
            assert samplesHash == savedHash

# Cluster search metrics
def SaveClusterScores(filename:str, algorithm:str, nFeaturesOrEpoch:int, kClusters, metricOne, metricTwo, samplesHash:str):
    with open(filename, 'wb') as file:
        pickle.dump([algorithm, nFeaturesOrEpoch, kClusters, metricOne, metricTwo, samplesHash], file)

def LoadClusterScores(filename, validateHash=None):
    with open(filename, 'rb') as file:
        loadedData = pickle.load(file)
        algorithm, nFeaturesOrEpoch, kClusters, metricOne, metricTwo, savedHash = loadedData
    if validateHash is not None:
        assert validateHash == savedHash, "Samples hash does not match. Metrics are probably from different data."
    return algorithm, nFeaturesOrEpoch, kClusters, metricOne, metricTwo, savedHash

def ClusterScoresVerifyAdd(filesDir, addAlgorithm=None, addHash=None):
    files = os.listdir(filesDir)

    raise Exception("dangerous function to unintentionally overwrite files")

    if len(files) == 0:
        raise Exception("No files found in {}".format(filesDir))
    
    # Loop over all embedding files and perform clustering search
    for file in files:
        filename = os.path.join(filesDir, file)

        with open(filename, 'rb') as file:
            loadedData = pickle.load(file)

        # faultyEpoch = loadedData[4]
        # loadedData[4] = loadedData[3]
        # loadedData[3] = loadedData[2]
        # loadedData[2] = loadedData[1]
        # loadedData[1] = faultyEpoch

        if len(loadedData) == 6:
            print("File already correct, continue. File {}".format(filename))
            continue
        
        offsetIndex = 0
        if addAlgorithm is None:
            algorithm = loadedData[0]
            offsetIndex += 1
        else:
            algorithm = addAlgorithm
        nFeaturesOrEpoch = loadedData[offsetIndex]
        kClusters = loadedData[offsetIndex+1]
        metricOne = loadedData[offsetIndex+2]
        metricTwo = loadedData[offsetIndex+3]

        if addHash is None:
            samplesHash = loadedData[offsetIndex+4]
            offsetIndex += 1
        else:
            samplesHash = addHash

        if len(loadedData) != 4+offsetIndex:
            raise Exception("Invalid number of elements in file {}".format(filename))
        
        if not isinstance(algorithm, str):
            raise Exception("algorithm is not a string")
        
        if not isinstance(nFeaturesOrEpoch, int):
            raise Exception("nFeaturesOrEpoch is not an integer")
        
        if not isinstance(kClusters, list) and not isinstance(kClusters, np.ndarray):
            raise Exception("kClusters is not a list")
        
        if not isinstance(metricOne, list)and not isinstance(kClusters, np.ndarray):
            raise Exception("metricOne is not a list")
        
        if not isinstance(metricTwo, list)and not isinstance(kClusters, np.ndarray):
            raise Exception("metricTwo is not a list")
        
        if not isinstance(samplesHash, str):
            raise Exception("samplesHash is not a string")
        

        SaveClusterScores(filename, algorithm, nFeaturesOrEpoch, kClusters, metricOne, metricTwo, samplesHash)

# Clusters assigned to samples
def GetClustersFilename(modelDir, modelType, nFeaturesOrEpoch, kClusters, algorithm, valData=False):
    # Create the directory for the cluster results
    if algorithm == 'kmeans':
        clusterSubDirName = 'kmeans_clusters'
        clusterName = 'clusters'
    elif algorithm == 'gmm':
        clusterSubDirName = 'gmm_clusters'
        clusterName = 'clusters'
    else:
        raise Exception("Unknown algorithm {}".format(algorithm))
    
    if modelType == 'dlModel':
        clusterName += '_epoch={}_k={}'.format(nFeaturesOrEpoch,kClusters)
    elif modelType == 'traditional':
        clusterName += '_nFeatures={}_k={}'.format(nFeaturesOrEpoch, kClusters)
    else:
        raise Exception("Unknown model type {}".format(modelType))
    
    if valData: clusterSubDirName += '_val'
    clusterDir = os.path.join(modelDir, clusterSubDirName)
    os.makedirs(clusterDir, exist_ok=True)
    clusterFilename = os.path.join(clusterDir, '{}.pkl'.format(clusterName))
    return clusterFilename

def SaveClusters(filename:str, nFeaturesOrEpoch:int, clusters, centroids, samplesHash:str):
    with open(filename, 'wb') as file:
        pickle.dump([nFeaturesOrEpoch, clusters, centroids, samplesHash], file)

def LoadClusters(filename, validateHash=None):
    with open(filename, 'rb') as file:
        data= pickle.load(file)
        nFeaturesOrEpoch, clusters, centroids, savedHash =data
    if validateHash is not None:
        assert validateHash == savedHash, "Samples hash does not match. Metrics are probably from different data."
    return nFeaturesOrEpoch, clusters, centroids, savedHash

def GetOddEvenClustersFilename(modelDir, modelType, nFeaturesOrEpoch, kClusters, algorithm, valData=False):
    # Create the directory for the cluster results
    if algorithm == 'kmeans':
        clusterSubDirName = 'kmeans_clusters'
    elif algorithm == 'gmm':
        clusterSubDirName = 'gmm_clusters'
    else:
        raise Exception("Unknown algorithm {}".format(algorithm))
    
    clusterName = 'oddEven_clusters'
    if modelType == 'dlModel':
        clusterName += '_epoch={}_k={}'.format(nFeaturesOrEpoch,kClusters)
    elif modelType == 'traditional':
        clusterName += '_nFeatures={}_k={}'.format(nFeaturesOrEpoch, kClusters)
    else:
        raise Exception("Unknown model type {}".format(modelType))
    
    if valData: clusterSubDirName += '_val'
    clusterDir = os.path.join(modelDir, clusterSubDirName)
    os.makedirs(clusterDir, exist_ok=True)
    clusterFilename = os.path.join(clusterDir, '{}.pkl'.format(clusterName))
    return clusterFilename

def SaveOddEvenClusters(filename:str, nFeaturesOrEpoch:int, oddClusters, oddCentroids, evenClusters, evenCentroids, samplesHash:str):
    with open(filename, 'wb') as file:
        pickle.dump([nFeaturesOrEpoch, oddClusters, oddCentroids, evenClusters, evenCentroids, samplesHash], file)

def LoadOddEvenClusters(filename, validateHash=None):
    with open(filename, 'rb') as file:
        data= pickle.load(file)
        nFeaturesOrEpoch, oddClusters, oddCentroids, evenClusters, evenCentroids, savedHash =data
    if validateHash is not None:
        assert validateHash == savedHash, "Samples hash does not match. Metrics are probably from different data."
    return nFeaturesOrEpoch, oddClusters, oddCentroids, evenClusters, evenCentroids, savedHash


# T-sne features
def GetTsneFilename(modelDir, modelType, nFeaturesOrEpoch, perplexity, valData=False):
    if valData: 
        tsnePath = os.path.join(modelDir, 'tsne_features_val')
    else:
        tsnePath = os.path.join(modelDir, 'tsne_features')
    
    if modelType == 'dlModel':
        tsneName = 'tsne_epoch={}_perplexity={}.pkl'.format(nFeaturesOrEpoch,perplexity)
    elif modelType == 'traditional':
        tsneName = 'tsne_nFeatures={}_perplexity={}.pkl'.format(nFeaturesOrEpoch, perplexity)
    else:
        raise Exception("Unknown model type {}".format(modelType))

    os.makedirs(tsnePath, exist_ok=True)
    tsneFilename = os.path.join(tsnePath,tsneName)
    return tsneFilename

def SaveTsneFeatures(filename:str, tsneFeatures, nFeaturesOrEpoch:int, samplesHash:str):
    with open(filename, 'wb') as file:
        pickle.dump([nFeaturesOrEpoch, tsneFeatures, samplesHash], file)

def LoadTsneFeatures(filename, validateHash=None):
    with open(filename, 'rb') as file:
        nFeaturesOrEpoch, tsneFeatures, savedHash = pickle.load(file)
    if validateHash is not None:
        assert validateHash == savedHash, "Samples hash does not match. Metrics are probably from different data."
    return nFeaturesOrEpoch, tsneFeatures, savedHash

def TsneAddHash(filename, addHash):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    if len(data)==2:
        with open(filename, 'wb') as file:
            pickle.dump([data[0], data[1], addHash], file)

def TsneVerifyAdd(filename, addNfeaturesOrEpochs = None, addHash = None):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    offset = 0
    if addNfeaturesOrEpochs is None:
        nFeaturesOrEpoch = data[0]
        offset += 1	
    else:
        nFeaturesOrEpoch = addNfeaturesOrEpochs
    
    if addHash is None:
        samplesHash = data[offset+1]
        offset += 1
    else:
        samplesHash = addHash

    if offset == 0:
        tsneFeatures = data
      
    SaveTsneFeatures(filename, tsneFeatures, nFeaturesOrEpoch, samplesHash)

# def linearToStokes(correlation_AB):
#     complexXX = correlation_AB['XX']
#     complexXY = correlation_AB['XY']
#     complexYX = correlation_AB['YX']
#     complexYY = correlation_AB['YY']

#     xxA = np.asarray(np.abs(complexXX))
#     xyA = np.asarray(np.abs(complexXY))
#     yxA = np.asarray(np.abs(complexYX))
#     yyA = np.asarray(np.abs(complexYY))
    
#     xxT = np.asarray(np.angle(complexXX))
#     xyT = np.asarray(np.angle(complexXY))
#     yxT = np.asarray(np.angle(complexYX))
#     yyT = np.asarray(np.angle(complexYY))

#     # Total Intensity (I):
#     stokesI = np.square(xxA)+np.square(xyA)+np.square(yxA)+np.square(yyA)
#     #stokesI = xxA^2 + xyA^2 + yxA^2 + yyA^2

#     # Linear Polarization Along X (Q):
#     stokesQ = np.square(xxA) - np.square(yyA)

#     # Linear Polarization Along Y (U):
#     stokesU = 2 * (xyA * yxA) * np.cos(xyT - yxT)

#     # Circular Polarization (V):
#     stokesV = 2 * (xyA * yxA) * np.sin(xyT - yxT)
    
#     stokesParameters = {'I':stokesI,'Q': stokesQ, 'U': stokesU, 'V':stokesV}
#     return stokesParameters

# def normCustomSoft(image):
#     mean = np.mean(image)
#     std = np.std(image)
    
#     threshold_factor=.5
#     maxUnchanged = mean + threshold_factor * std
#     maxAbsolute = mean + 6 * std
#     image = np.clip(image,0,maxAbsolute)
#     maxValue = np.max(image)

#     # Apply the soft thresholding
#     aboveThreshold = image[image > maxUnchanged]-maxUnchanged
#     factorAbove = (1-(aboveThreshold)/(maxValue-maxUnchanged))
#     addAbove = (aboveThreshold)*factorAbove*0.5
#     image[image > maxUnchanged] = maxUnchanged + addAbove
#     return image

def normalizeComplex(observationComplexValues, ignorePadding=False,standardize=True):
    if ignorePadding:
        complexValues = observationComplexValues
    else:
        complexValues = observationComplexValues[1:,:]

    magnitudes = np.abs(complexValues)

    if standardize:
        magnitudesMin = magnitudes-np.min(magnitudes)
        magnitudesLog = np.log10(magnitudesMin+1)
        clipSigma = 2
        minClip = np.mean(magnitudesLog) - np.std(magnitudesLog)*clipSigma
        maxClip = np.mean(magnitudesLog) + np.std(magnitudesLog)*clipSigma
        magnitudesClipped = np.clip(magnitudesLog,minClip,maxClip) 
        magnitudesNormed = (magnitudesClipped-np.min(magnitudesClipped))/(np.max(magnitudesClipped)-np.min(magnitudesClipped))
    else:
        magnitudesLog = np.log10(magnitudes+1)
        magnitudesNormed = magnitudesLog

    if np.isnan(magnitudes).any():
        print("NaN detected in magnitudes")
    if np.isnan(magnitudesNormed).any():
        print("NaN detected in magnitudesNormed")

    # Scale the norm of the vector according to the normalization of the magnitude
    normalizeFactor = np.divide(magnitudesNormed, magnitudes, out=np.zeros_like(magnitudesNormed), where=magnitudes!=0)
    complexNormalized = complexValues*normalizeFactor*0.99999

    # Write back normalized values to observation
    if ignorePadding:
        observationComplexValues=complexNormalized
    else:
        observationComplexValues[1:,:]=complexNormalized
    
    return observationComplexValues

def randomlyDivideOverArray(totalCounts,nElements):
    minPerElement = int(totalCounts/nElements)
    toBeDivided = totalCounts-(minPerElement*nElements)
    countsPerElement = minPerElement*np.ones(nElements,dtype=np.uint8)
    countsPerElement[np.random.choice(nElements, toBeDivided, replace=False)] += 1
    return countsPerElement

def sampleObservations(h5SetsLocation, nSamples, timeWindow=None, strategy = 'equalSubbands_equalCorrelation_randomTime'):
    """
    When the observations are split into time windows, the existing number of x-y pairs are
    nCorrelations * nSubbands * floor(duration/nTimeSteps)

    When n samples have to be sampled, they should represent the complete dataset as good as possible
    Many correlations contain the same RFI patterns when they are in the same time window.

    Restrictions:
    - Validation data must be sampled as well in a different time window

    """
    h5Sets = [filename for filename in os.listdir(h5SetsLocation) if filename.endswith('.h5')]

    if len(h5Sets) == 0:
        raise Exception("No h5 files found in {}".format(h5SetsLocation))

    datasetFrequencies = []
    for h5Set in h5Sets:
        antennaData, subbandFrequencies, time = getMetadata(os.path.join(h5SetsLocation, h5Set))
        datasetFrequencies.append(subbandFrequencies)
    
    antennaNames = antennaData[0]
    nSubbands = len(h5Sets)
    nAntennas = len(antennaNames)
    
    if timeWindow is None:
        timeStart = 0
        timeStop = len(time)
    else: 
        [timeStart, timeStop] = timeWindow
        if timeStop is None:
            timeStop = len(time)
        if timeStart is None:
            timeStart = 0
    nTimeSteps = timeStop-timeStart
    nDifferentTimePatches = int(nTimeSteps/utils.constants.nInputTimeSteps)
    nDifferentTimePatches = round(nDifferentTimePatches/2)
    
    nLeftOverTimeStamps = nTimeSteps-nDifferentTimePatches*utils.constants.nInputTimeSteps
    
    allCorrelations = []
    for antennaA in range(nAntennas):
        for antennaB in range(nAntennas):
            if antennaB <= antennaA:continue
            allCorrelations.append([antennaA,antennaB])
    nBaselines = len(allCorrelations)
    
    samples = []
    if strategy == 'equalSubbands_equalTime':
        # First, calculate how many samples should be taken from each subband
        samplesPerSubband = randomlyDivideOverArray(nSamples,nSubbands)

        for subbandIndex, nSubbandSamples in enumerate(samplesPerSubband):           
            # Within the time window, patches can be sampled from the subband
            # First the most uniform way is to use as much of the time window as possible
            # Then, to sample from antennas such that from each two pairs, not both are close to each other
            offsetPatchIndices = np.random.choice(nDifferentTimePatches+1, nLeftOverTimeStamps, replace=True)
            timePatchStartIndices = [timeStart + np.count_nonzero(offsetPatchIndices == 0)]
            for timepatchIndex in range(1, nDifferentTimePatches):
                patchStartIndex = timePatchStartIndices[-1] + utils.constants.nInputTimeSteps + np.count_nonzero(offsetPatchIndices == timepatchIndex)
                timePatchStartIndices.append(patchStartIndex)
            
            # Since there are more correlations than time samples, divide over the time patches
            samplesPerTimePatch = randomlyDivideOverArray(nSubbandSamples,nDifferentTimePatches)
            for timePatchIndex, nSamplesTimePatch in enumerate(samplesPerTimePatch):
                correlationIndices = np.random.choice(len(allCorrelations), nSamplesTimePatch, replace=False)
                for correlationIndex in correlationIndices:
                    [antennaA, antennaB] = allCorrelations[correlationIndex]
                    samples.append([h5Sets[subbandIndex],antennaA,antennaB,timePatchStartIndices[timePatchIndex]])

    elif strategy == 'equalSubbands_equalCorrelation':
        # First, calculate how many samples should be taken from each subband
        samplesPerSubband = randomlyDivideOverArray(nSamples,nSubbands)

        for subbandIndex, nSubbandSamples in enumerate(samplesPerSubband):   
            samplesPerBaseline = randomlyDivideOverArray(nSubbandSamples,nBaselines)
            for baselineIndex, nSamplesBaseline in enumerate(samplesPerBaseline):
                if nSamplesBaseline == 0: continue

                # For each subband/baseline, sample new time patches to make sure time is random.
                timePatchStartIndices = []
                timeBlocks = [np.arange(nTimeSteps-utils.constants.nInputTimeSteps)]
                for baselineSample in range(nDifferentTimePatches):
                    availableTimeSteps = np.concatenate([arr.ravel() for arr in timeBlocks])

                    # choose a random sample
                    randomPatchStart = np.random.choice(availableTimeSteps, 1)[0]
                    timePatchStartIndices.append(randomPatchStart)
                    
                    for blockIndex, block in enumerate(timeBlocks):
                        if randomPatchStart in block:
                            deleteBlock = blockIndex

                            rangeStartIndex = np.where(availableTimeSteps == randomPatchStart)[0][0]
                            rangeStopIndex = rangeStartIndex + utils.constants.nInputTimeSteps
                            newBlockOne = block[0:rangeStartIndex]
                            if len(newBlockOne)>=utils.constants.nInputTimeSteps:
                                newBlockOne = newBlockOne[0:len(newBlockOne)-utils.constants.nInputTimeSteps+1]
                                timeBlocks.append(newBlockOne)
                            newBlockTwo = block[rangeStopIndex:]
                            if len(newBlockTwo)>0:
                                timeBlocks.append(newBlockTwo)
                    del timeBlocks[deleteBlock]
                
                # Choose random time patches
                timePatchIndices = np.random.choice(nDifferentTimePatches, nSamplesBaseline, replace=False)

                for timepatchIndex in timePatchIndices:
                    [antennaA, antennaB] = allCorrelations[baselineIndex]
                    startTime = timePatchStartIndices[timepatchIndex]
                    samples.append([h5Sets[subbandIndex],antennaA,antennaB,startTime])
    
    elif strategy == 'equalSubbands_equalCorrelation_randomTime':       
        # First, calculate how many samples should be taken from each subband
        samplesPerSubband = randomlyDivideOverArray(nSamples,nSubbands)

        for subbandIndex, nSubbandSamples in enumerate(samplesPerSubband):   
            samplesPerBaseline = randomlyDivideOverArray(nSubbandSamples,nBaselines)
            for baselineIndex, nSamplesBaseline in enumerate(samplesPerBaseline):
                if nSamplesBaseline == 0: continue

                if nSamplesBaseline > nDifferentTimePatches:
                    warnings.warn("nSamplesBaseline > nDifferentTimePatches. Possibility of unable to sample all time patches, which will raise an exception.")

                # For each subband/baseline, sample new time patches to make sure time is random.
                timePatchStartIndices = []
                timeBlocks = [np.arange(nTimeSteps-utils.constants.nInputTimeSteps)]
                for baselineSample in range(nSamplesBaseline):
                    if len(timeBlocks)==0: raise Exception("No available time steps left")
                    availableTimeSteps = np.concatenate([arr.ravel() for arr in timeBlocks])

                    # choose a random sample
                    randomPatchStart = np.random.choice(availableTimeSteps, 1)[0]
                    timePatchStartIndices.append(randomPatchStart)
                    
                    for blockIndex, block in enumerate(timeBlocks):
                        if randomPatchStart in block:
                            deleteBlock = blockIndex

                            rangeStartIndex = np.where(availableTimeSteps == randomPatchStart)[0][0]
                            rangeStopIndex = rangeStartIndex + utils.constants.nInputTimeSteps
                            newBlockOne = block[0:rangeStartIndex]
                            if len(newBlockOne)>=utils.constants.nInputTimeSteps:
                                newBlockOne = newBlockOne[0:len(newBlockOne)-utils.constants.nInputTimeSteps+1]
                                timeBlocks.append(newBlockOne)
                            newBlockTwo = block[rangeStopIndex:]
                            if len(newBlockTwo)>0:
                                timeBlocks.append(newBlockTwo)
                    del timeBlocks[deleteBlock]

                for timepatchIndex in timePatchStartIndices:
                    [antennaA, antennaB] = allCorrelations[baselineIndex]
                    startTime = timepatchIndex #timePatchStartIndices[timepatchIndex]
                    samples.append([h5Sets[subbandIndex],antennaA,antennaB,startTime])    

    elif strategy == 'randomFrequency_randomTime':
        # First, calculate how many samples should be taken from each subband
        samplesPerBaseline = randomlyDivideOverArray(nSamples,nBaselines)

        datasetFrequencies = np.asarray(datasetFrequencies)
        freqStart = np.min(datasetFrequencies)
        freqStop = np.max(datasetFrequencies)
        freqStepSize = datasetFrequencies[0][1]-datasetFrequencies[0][0]
        
        possibleFrequencies = np.arange(freqStart, freqStop, freqStepSize)
        emptyMapping = -1*np.ones((possibleFrequencies.shape[0]),dtype=np.int8)
        frequencySubbandMapping = np.column_stack((possibleFrequencies, emptyMapping))
        
        baselineTimeFrequencyMask = np.zeros((possibleFrequencies.shape[0], nTimeSteps),dtype=np.uint8)

        for oneFreq in possibleFrequencies:
            for subbandIndex in range(datasetFrequencies.shape[0]):
                if oneFreq in datasetFrequencies[subbandIndex]:
                    matchingIndex = np.where(possibleFrequencies==oneFreq)[0][0]
                    frequencySubbandMapping[matchingIndex,1] = subbandIndex
                    baselineTimeFrequencyMask[matchingIndex,:] = 1

        for baselineIndex, nSamplesBaseline in enumerate(samplesPerBaseline):
            if nSamplesBaseline == 0: continue
            [antennaA, antennaB] = allCorrelations[baselineIndex]

            currentBaselineMask = baselineTimeFrequencyMask.copy()
            
            iterationCounter = 0
            while nSamplesBaseline > 0:
                iterationCounter += 1
                if iterationCounter > 10000:
                    raise Exception("Too many iterations")
                
                # Choose random frequency and time
                randomF = random.randint(0,currentBaselineMask.shape[0]-utils.constants.inputHeight-1)
                randomT = random.randint(0,currentBaselineMask.shape[1]-utils.constants.nInputTimeSteps-1)

                # Check if the mask is still valid
                maskPatch = currentBaselineMask[randomF:randomF+utils.constants.inputHeight,randomT:randomT+utils.constants.nInputTimeSteps]
                if not 0 in maskPatch:
                    # Accept sample
                    nSamplesBaseline -= 1
                    currentBaselineMask[randomF:randomF+utils.constants.inputHeight,randomT:randomT+utils.constants.nInputTimeSteps] = np.zeros_like(maskPatch)
                     
                    # Add sample to list
                    correspondingFrequency = frequencySubbandMapping[randomF,0]
                    samples.append([correspondingFrequency,antennaA,antennaB,randomT])    

        return samples, frequencySubbandMapping  
    
    else:
        raise Exception("Invalid strategy. Please choose an implemented one.")
    return samples

def sampleFromH5(h5SetsLocation, sampleList, frequencySubbandMapping=None,timeWindow=None, verbose = False, standardize=True, normalizing = True, loadComponents = None):
    if normalizing==False:
        warnings.warn("Normalizing is disabled. This is not recommended and only used for calculating dataset statistics.")
    
    if loadComponents is None:
        loadNames = True
        loadPositions = True
        loadFrequencies = True
        loadTimes = True
        loadUvws = True
        loadTimeStartSteps = True
        loadObservations = True
        loadLabels = True
        loadMetadata = True
    else:    
        loadNames = 'names' in loadComponents
        loadPositions = 'positions' in loadComponents
        loadFrequencies = 'frequencies' in loadComponents
        loadTimes = 'times' in loadComponents
        loadUvws = 'uvws' in loadComponents
        loadTimeStartSteps = 'timeStartStep' in loadComponents
        loadObservations = 'observations' in loadComponents
        loadLabels = 'labels' in loadComponents
        loadMetadata = 'setMetadata' in loadComponents
    
    nSamples = len(sampleList)


    observations = np.zeros((nSamples,utils.constants.inputHeight, utils.constants.inputWidth, utils.constants.nRepresentations),dtype=np.complex64)
    if loadLabels: labels = np.zeros((nSamples,utils.constants.inputHeight, utils.constants.inputWidth),dtype=np.uint8)
    if loadPositions: positions = np.zeros((nSamples,2, 3),dtype=np.float32)
    if loadFrequencies: frequencies = np.zeros((nSamples,64),dtype=np.float64)
    names = []
    times = []
    timeStartStep = []
    uvws = []

    for sampleIndex, sample in tqdm(enumerate(sampleList),disable=not verbose):
        [h5Set, indexA, indexB, timeStart] = sample
        timeStop = timeStart + utils.constants.nInputTimeSteps
        
        # Check if antennas are the same
        if indexA == indexB:
            raise Exception("Antenna A = %s and Antenna B = %s. Antennas cannot be the same")
        
        h5Filename = os.path.join(h5SetsLocation,h5Set)
        if not os.path.exists(h5Filename):
            print("File %s does not exist" % h5Filename)
            return None

        with h5py.File(h5Filename, 'r') as f5file:
            # Get metadata only from the first subband
            metadata_group = f5file['Metadata']
            antenna_name = [tempName.astype(str) for tempName in metadata_group['ANTENNA_NAME'][:]]
            antennaA = antenna_name[indexA]
            antennaB = antenna_name[indexB]
            if loadPositions: antenna_positions = metadata_group['POSITION'][:]
            
            # Get observation data
            correlationOneAB = f5file['Observations'][antennaA][antennaB]

            if loadLabels:
                label = correlationOneAB['Flags'][:]
                if np.sum(label) == 0:
                    raise Exception("No labels found in observation. She subband is not flagged")
            
            # Get time and uvw from the first subband
            if loadUvws: uvw = correlationOneAB['uvw'][timeStart:timeStop]
            if loadFrequencies: chanfreq = metadata_group['CHAN_FREQ'][:]/1e6
            if loadMetadata or loadTimes:
                time = correlationOneAB['Time'][timeStart:timeStop]
                if timeWindow is None:
                    timeWindow = [0,len(time)]
                else:
                    if timeWindow[0] is None:
                        timeWindow[0] = 0
                    if timeWindow[1] is None:
                        timeWindow[1] = len(time)
                setTimeStart = correlationOneAB['Time'][timeWindow[0]]
                setTimeStop = correlationOneAB['Time'][timeWindow[1]]
   
            for repIndex, representation in enumerate(utils.datasets.linearRepresentation): 
                complexValues = correlationOneAB[representation][:,timeStart:timeStop]
                if normalizing:
                    normalized = utils.functions.normalizeComplex(complexValues,ignorePadding = False, standardize=standardize)
                    normalized[0,:] = normalized[1,:]
                    observations[sampleIndex,:,:,repIndex] = normalized
                else:
                    observations[sampleIndex,:,:,repIndex] = complexValues
  
        if loadLabels: labels[sampleIndex,:,:] = label[:,timeStart:timeStop]
        if loadPositions: 
            positions[sampleIndex,0,:]=antenna_positions[indexA]
            positions[sampleIndex,1,:]=antenna_positions[indexB]
        if loadFrequencies: frequencies[sampleIndex,:]=chanfreq
        if loadTimes: times.append(time)
        if loadTimeStartSteps: timeStartStep.append(timeStart)
        if loadNames: names.append([antennaA, antennaB])
        if loadUvws: uvws.append(uvw)

        if timeStop-timeStart != utils.constants.nInputTimeSteps : raise Exception("time series not 256. This is caused by a bug, or observations with unequal lengts")

    if loadMetadata: setMetadata = [[setTimeStart,setTimeStop]]

    results = ()
    if loadNames: results += (names,)
    if loadPositions: results += (positions,)
    if loadFrequencies: results += (frequencies,)
    if loadTimes: results += (times,)
    if loadUvws: results += (uvws,)
    if loadTimeStartSteps: results += (timeStartStep,)
    if loadObservations: results += (observations,)
    if loadLabels: results += (labels,)
    if loadMetadata: results += (setMetadata,)
    return results

def getMetadata(h5Set):
    with h5py.File(h5Set, 'r') as f5file:
        metadata_group = f5file['Metadata']
        channelFrequencies = metadata_group['CHAN_FREQ'][:]/1e6
        antennaNames = [tempName.astype(str) for tempName in metadata_group['ANTENNA_NAME'][:]]
        antennaPositions = metadata_group['POSITION'][:]
        correlation_AB = f5file['Observations'][antennaNames[0]][antennaNames[1]]    
        time = correlation_AB['Time'][:]  
    antennaData = [antennaNames,antennaPositions]
    return antennaData, channelFrequencies, time

def getObservation(h5Set, antennaA=0, antennaB=1, time=None):
    with h5py.File(h5Set, 'r') as f5file:
        warning = False
        metadata_group = f5file['Metadata']
        chan_freq = metadata_group['CHAN_FREQ'][:]/1e6
        antenna_name = [tempName.astype(str) for tempName in metadata_group['ANTENNA_NAME'][:]]
        antenna_positions = metadata_group['POSITION'][:]
        
        # Convert antenna name to its corresponding index
        if isinstance(antennaA, str):
            indexA = antenna_name.index(antennaA)
        else: indexA = antennaA
        if isinstance(antennaB, str):
            indexB = antenna_name.index(antennaB)
        else: indexB = antennaB

        # Check if the combination is allowed
        if indexA == indexB:
            raise Exception("Antenna A = %s and Antenna B = %s. Antennas cannot be the same")
        if indexB < indexA:
            warning=True
            warnings.warn("antenna B is lower than antenna A, antennas are swapped")
            tempIndexA = indexA
            indexA = indexB
            indexB = tempIndexA

        # If index was provided, find corresponding antenna name
        antennaA = antenna_name[indexA].astype(str)
        antennaB = antenna_name[indexB].astype(str)

        posA = antenna_positions[indexA]
        posB = antenna_positions[indexB]

        correlation_AB = f5file['Observations'][antennaA][antennaB]
        
        obsShape = correlation_AB['XX'].shape
        observation = np.zeros((obsShape[0],obsShape[1], utils.constants.nRepresentations),dtype=np.complex128)
        time = correlation_AB['Time'][:]
        label = correlation_AB['Flags'][:]

        for repIndex, representation in enumerate(utils.constants.linearRepresentation):  
            observation[:,:,repIndex] = correlation_AB[representation] 
    return warning,[antennaA, antennaB, posA, posB],chan_freq, time, observation, label