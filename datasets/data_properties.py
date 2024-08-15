import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import utils as utils

datasetName = "LOFAR_L2014581 (recording)"
datasetSubdir = 'dataset250k'
h5SetsLocation = utils.functions.getH5SetLocation(datasetName)
trainSamplesFilename = utils.functions.getDatasetLocation(datasetName, 'trainSamples',subdir=datasetSubdir)

def PlotAoflaggerExamples(plotsLocation):
    nSamples = 100
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dimensionReduction', 'train', -1, trainSamplesFilename,dataSettings={'batchSize':10, 'normalizationMethod':12}, bufferAll=True, cacheDataset=True, nSamples = 500) # NOTE: for 500 samples there is a cache file
    normalizedObservations, labels = dataGenerator.getItems(stopIdx = nSamples)
    colorImages = dataGenerator.ConvertToImage(normalizedObservations)
    
    aoflaggerExamplesPlotsLocation = os.path.join(plotsLocation, 'AOFlagger examples')
    os.makedirs(aoflaggerExamplesPlotsLocation, exist_ok=True)

    for sampleIdx, (colorImage, label) in enumerate(zip(colorImages, labels)):
        fig, ax = plt.subplots(nrows=2,figsize=(10, 6))
        colorFlagged = colorImage.copy()
        colorFlagged[label==1] = [253/255,231/255,37/255]

        ax[0].set_title('Original')
        ax[0].imshow(colorImage)
        ax[0].invert_yaxis()
        ax[0].set_ylabel('Channel')
        ax[0].set_xticks([])
        
        ax[1].set_title('Flagged by AOFlagger')
        ax[1].imshow(colorFlagged)
        ax[1].invert_yaxis()
        ax[1].set_ylabel('Channel')
        ax[1].set_xlabel('Time (s)')
        
        dpi=300
        plt.savefig(os.path.join(aoflaggerExamplesPlotsLocation, "{}".format(sampleIdx)), dpi=dpi, bbox_inches='tight')
        plt.close()

def PlotRfiPercentagePerSubband(plotsLocation):
    nSamples = 6000 # NOTE: for 6000 samples there is a cache file
    sampleList = pickle.load(open(trainSamplesFilename, "rb"))
    sampleList = sampleList[0][:nSamples]

    sampledData = utils.functions.sampleFromH5(h5SetsLocation, sampleList, standardize=False)
    if sampledData is None:
        print("(Not all) h5 files available. Skipping calculating RFI percentage per subband.")
        return

    names,positions,frequencies,times,uvws,sinTimes,observations,labels,setMetadata = sampledData
    labels[:,0,:] = 0

    nVisibilitiesPerSample = labels.shape[1] * labels.shape[2]
    nRfiPerSample = np.sum(labels,axis=(1,2))
    
    # Total RFI statistics
    totalPercentageRFI = np.sum(nRfiPerSample) / (nVisibilitiesPerSample * nSamples)
    print("Mean RFI percentage per sample: {:.2f}%".format(totalPercentageRFI*100))

    # RFI per subband
    startFrequencyPerSample  = frequencies[:,0]
    startFrequencies,nSamplesPerSubband = np.unique(startFrequencyPerSample, return_counts=True)
    nSubbands = len(startFrequencies)
    nRfiPerSubband = np.zeros(nSubbands)
    for subbandIdx, startFrequency in enumerate(startFrequencies):
        subbandIndices = startFrequencyPerSample == startFrequency
        nRfiPerSubband[subbandIdx] = np.sum(nRfiPerSample[subbandIndices])
    
    percentagePerSubband = nRfiPerSubband / (nSamplesPerSubband * nVisibilitiesPerSample)
    percentagePerSubband *= 100

    plt.figure(figsize=(10,6))
    plt.plot(startFrequencies,percentagePerSubband)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('RFI (%)')
    plt.xlim([startFrequencies[0],startFrequencies[-1]])
    plt.savefig(os.path.join(plotsLocation,'rfi_per_subband.png'),bbox_inches='tight')
    plt.close()

def PlotRfiPhenomena(plotsLocation):
    nSamples = 100
    sampleList = pickle.load(open(trainSamplesFilename, "rb"))
    sampleList = sampleList[0][:nSamples]

    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dimensionReduction', 'train', -1, trainSamplesFilename,dataSettings={'batchSize':10, 'normalizationMethod':0}, bufferAll=True, cacheDataset=True, nSamples = 500) # NOTE: for 500 samples there is a cache file
    _, labels = dataGenerator.getItems(nSamples)
    rfiResults, rfiRatioResults, rfiCategories, dataY, dataRfiWeak, dataRfiY, dataRfiX = utils.datasets.calcRfiTypesPerSample(labels,returnDebugImages=True)

    rfiTypePlotsLocation = os.path.join(plotsLocation, 'rfi phenomena')
    os.makedirs(rfiTypePlotsLocation, exist_ok=True)

    for sampleIdx in range(nSamples):
        fig, ax = plt.subplots(nrows=4,figsize=(10, 10))

        ax[0].set_title('Label')
        ax[0].imshow(dataY[sampleIdx])
        ax[0].invert_yaxis()
        ax[0].set_ylabel('Channel')
        ax[0].set_xticks([])
        
        ax[1].set_title('Erratic RFI')
        ax[1].imshow(dataRfiWeak[sampleIdx])
        ax[1].invert_yaxis()
        ax[1].set_ylabel('Channel')
        ax[1].set_xticks([])
        
        ax[2].set_title('Broadband RFI')
        ax[2].imshow(dataRfiY[sampleIdx])
        ax[2].invert_yaxis()
        ax[2].set_ylabel('Channel')
        ax[2].set_xticks([])
        
        ax[3].set_title('Narrow-band RFI')
        ax[3].imshow(dataRfiX[sampleIdx])
        ax[3].invert_yaxis()
        ax[3].set_ylabel('Channel')
        ax[3].set_xlabel('Time (s)')
  
        dpi=300
        plt.savefig(os.path.join(rfiTypePlotsLocation, "rfi types sample {}".format(sampleIdx)), dpi=dpi, bbox_inches='tight')
        plt.close()

def PlotAmplitudesHistogramNormalizations(plotsLocation, normalizations = [{'index':0, 'name': 'Scaled by log$_{10}$'},{'index':12, 'name': 'Median and MAD'},{'index':13, 'name': 'Mean and SD'}]):
    nSamples = 1000
    # sampleList = pickle.load(open(trainSamplesFilename, "rb"))
    # sampleList = sampleList[0][:nSamples]

    
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dimensionReduction', 'train', -1, trainSamplesFilename,dataSettings={'batchSize':10, 'normalizationMethod':0}, bufferAll=True, cacheDataset=True, nSamples = nSamples) # NOTE: for 500 samples there is a cache file
    observations, labels = dataGenerator.getItems()
    # reverse the log10 and add the 1 again to the observations
    unscaledObservations = np.power(10,np.abs(observations))-1

    bins = 500
    unscaledMagnitudes = unscaledObservations.flatten()
    unscaledRfiMagnitudes = unscaledObservations[labels==1].flatten()
    unscaledBackgroundMagnitudes = unscaledObservations[labels==0].flatten()
    unscaledHistogram = np.histogram(unscaledMagnitudes, bins=bins)
    unscaledRfiHistogram = np.histogram(unscaledRfiMagnitudes, bins=bins)
    unscaledRackgroundHistogram = np.histogram(unscaledBackgroundMagnitudes, bins=bins)

    yLimit = nSamples*1.2e3

    fig,ax = plt.subplots(nrows=1,ncols = 1+len(normalizations),figsize=(20,5))
    ax[0].set_title('Raw data')
    ax[0].plot(unscaledRfiHistogram[1][1:-1],unscaledRfiHistogram[0][1:], label='RFI')
    ax[0].plot(unscaledRackgroundHistogram[1][1:-1],unscaledRackgroundHistogram[0][1:], label='SOI')
    ax[0].set_xlabel('Amplitude (log scale)')
    ax[0].set_ylabel('Number of visibilities')
    ax[0].legend()
    ax[0].set_ylim([0,yLimit])
    ax[0].set_xscale('log')

    normalizationText = ''
    for plotIdx, normalization in enumerate(normalizations):
        normalizationMethod = normalization['index']
        normalizationName = normalization['name']

        if plotIdx > 0:
            normalizationText += '_'
        normalizationText += str(normalizationMethod)

        normalizedComplexX = utils.datasets.NormalizeComplex(observations, normalizationMethod)
        normalizedMagnitudes = np.abs(normalizedComplexX).flatten()
        normalizedRfiMagnitudes = np.abs(normalizedComplexX[labels==1]).flatten()
        normalizedBackgroundMagnitudes = np.abs(normalizedComplexX[labels==0]).flatten()

        normalizedHistogram = np.histogram(normalizedMagnitudes, bins=bins)
        normalizedRfiHistogram = np.histogram(normalizedRfiMagnitudes, bins=bins)
        normalizedBackgroundHistogram = np.histogram(normalizedBackgroundMagnitudes, bins=bins)

        ax[plotIdx+1].set_title(normalizationName)
        ax[plotIdx+1].plot(normalizedRfiHistogram[1][1:-1],normalizedRfiHistogram[0][1:], label='RFI')
        ax[plotIdx+1].plot(normalizedBackgroundHistogram[1][1:-1],normalizedBackgroundHistogram[0][1:], label='Background')
        ax[plotIdx+1].set_xlabel('Amplitude')
        ax[plotIdx+1].set_ylim([0,yLimit])

    plt.savefig(os.path.join(plotsLocation,'Magnitude_histrograms_normalizations_{}.png'.format(normalizationText)),bbox_inches='tight')
    plt.close()

def PlotNormalizedExamples(plotsLocation, normalizations = [{'index':0, 'name': 'None'},{'index':12, 'name': 'Median and MAD'},{'index':13, 'name': 'Mean and SD'}]):
    normalizationPlotslocation = os.path.join(plotsLocation, 'Normalized examples')
    os.makedirs(normalizationPlotslocation, exist_ok=True)

    nSamples = 200
    sampleList = pickle.load(open(trainSamplesFilename, "rb"))
    sampleList = sampleList[0][:nSamples]
    frequencyMap = None
    
    
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dimensionReduction', 'train', -1, trainSamplesFilename,dataSettings={'batchSize':10, 'normalizationMethod':0}, bufferAll=True, cacheDataset=True, nSamples = 500) # NOTE: for 500 samples there is a cache file

    for plotIdx, normalization in enumerate(normalizations):
        normalizationMethod = normalization['index']
        normalizationName = normalization['name']
        
        dataGenerator.normalizationMethod = normalizationMethod
        observations, _ = dataGenerator.getItems(stopIdx = nSamples)

        if normalizationMethod == 0:
            magnitudeX = np.abs(observations)
            scaledMagnitudeX = magnitudeX.copy()
            scaledMagnitudeX -= 5
            normalizeFactor = np.divide(scaledMagnitudeX, magnitudeX, out=np.zeros_like(magnitudeX,dtype = np.float32), where=magnitudeX!=0)
            normalizedComplexX = np.multiply(observations,normalizeFactor)
        else:
            normalizedComplexX = observations
        normalizedColor = dataGenerator.ConvertToImage(normalizedComplexX)

        for sampleIdx, colorImage in enumerate(normalizedColor):
            colorImage = (255*colorImage).astype(np.uint8)
            pilImage = Image.fromarray(colorImage)

            saveName = os.path.join(normalizationPlotslocation, 'Normalized_sample_{}_{}.png'.format(sampleIdx, normalizationName))
            pilImage.save(saveName)