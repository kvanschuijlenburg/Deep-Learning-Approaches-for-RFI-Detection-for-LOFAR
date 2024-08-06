import os
import math
import pickle

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from tqdm import tqdm
import scipy.stats as st
from scipy.stats import wilcoxon
from PIL import Image

import utils as utils
import unsupervised.cluster_algorithms as cluster_algorithms

datasetName = "LOFAR_L2014581 (recording)"
modelName = "kmeansSearch"
datafolderSubdir = 'dataset250k'

def searchOneEmbedding(args):
    embeddingFile,embeddingPath,clusterDir,clusterBasename, algorithm, kClusters = args
    embeddingFilename = os.path.join(embeddingPath, embeddingFile)
    nFeaturesOrEpoch, embedding, savedHash = utils.functions.LoadEmbedding(embeddingFilename)
    if np.isnan(embedding).any():
        print("Found NaN value in embedding {}. Continue".format(embeddingFile))
        return
    clusterResultsFilename = os.path.join(clusterDir, '{}{}.pkl'.format(clusterBasename, nFeaturesOrEpoch))

    metricOne = []
    metricTwo = []
    kValuesExperiment = []
    if os.path.exists(clusterResultsFilename):
        savedAlgorithm, savednFeaturesOrEpoch, kValuesExperiment, metricOne, metricTwo,_ = utils.functions.LoadClusterScores(clusterResultsFilename, savedHash)
        assert savedAlgorithm == algorithm
        assert savednFeaturesOrEpoch == nFeaturesOrEpoch
        metricOne = list(metricOne)
        metricTwo = list(metricTwo)

    searchKValues = [k for k in kClusters if k not in kValuesExperiment]
    if len(searchKValues) == 0:
        return
    
    if algorithm == 'kmeans':
        newMetricOne, newMetricTwo = cluster_algorithms.kmeansSearch(embedding, searchKValues, verbose=True)
    elif algorithm == 'gmm':
        newMetricOne, newMetricTwo = cluster_algorithms.gmmSearch(embedding, searchKValues)

    metricOne.extend(newMetricOne)
    metricTwo.extend(newMetricTwo)
    kValuesExperiment.extend(searchKValues)

    utils.functions.SaveClusterScores(clusterResultsFilename, algorithm, nFeaturesOrEpoch, kValuesExperiment, metricOne, metricTwo, savedHash)

def ClusterSearchEmbeddings(modelDir, algorithm, modelType, kClusters=list(range(5,51,1)), valData=False):
    # Find the file location of the embeddings
    if valData:
        embeddingPath = os.path.join(modelDir, 'embedding_val')
    else:
        embeddingPath = os.path.join(modelDir, 'embedding')

    if not os.path.exists(embeddingPath):
        raise Exception("No embedding directory found in {}".format(modelDir))
    embeddingFiles = os.listdir(embeddingPath)
    if len(embeddingFiles) == 0:
        raise Exception("No embedding files found in {}".format(embeddingPath))
    
    # Create the directory for the cluster results
    if algorithm == 'kmeans':
        clusterSubDirName = 'kmeans_metrics'
        clusterBasename = 'kMeansSearch'
    elif algorithm == 'gmm':
        clusterSubDirName = 'gmm_metrics'
        clusterBasename = 'gmmSearch'
    else:
        raise Exception("Unknown algorithm {}".format(algorithm))
    
    if modelType == 'dlModel':
        clusterBasename += '_epoch='
    elif modelType == 'traditional':
        clusterBasename += '_nFeatures='
    else:
        raise Exception("Unknown model type {}".format(modelType))
    
    if valData: clusterSubDirName += '_val'
    clusterDir = os.path.join(modelDir, clusterSubDirName)
    os.makedirs(clusterDir, exist_ok=True)
    
    if len(embeddingFiles) == 0:
        return
    experiments = [(embeddingFile, embeddingPath,clusterDir,clusterBasename, algorithm, kClusters) for embeddingFile in embeddingFiles]

    for experiment in experiments:
        searchOneEmbedding(experiment)
   
def ClusterEmbedding(modelDir, algorithm,modelType, nFeaturesOrEpoch, kClusters:int, valData=False):
    # Load embedding
    embeddingFile = utils.functions.GetEmbeddingFilename(modelDir, modelType, nFeaturesOrEpoch, valData)
    if not os.path.exists(embeddingFile):
        raise Exception("Embedding file {} not found".format(embeddingFile))
    savednFeaturesOrEpoch, embedding, embeddingHash = utils.functions.LoadEmbedding(embeddingFile)
    assert nFeaturesOrEpoch == savednFeaturesOrEpoch

    # Load clusters, if they exist
    clustersFilename = utils.functions.GetClustersFilename(modelDir, modelType, nFeaturesOrEpoch, kClusters, algorithm, valData)
    if os.path.exists(clustersFilename):
        savednFeaturesOrEpoch, clusters, centroids, savedHash = utils.functions.LoadClusters(clustersFilename)
        assert savedHash==embeddingHash
        assert savednFeaturesOrEpoch == nFeaturesOrEpoch
        return

    clusters, centroids = cluster_algorithms.kmeans(embedding, kClusters, returnCentroids=True)
    utils.functions.SaveClusters(clustersFilename, nFeaturesOrEpoch, clusters, centroids, embeddingHash)

def PlotCentroidsSamples(modelDir, plotsDir, algorithm, modelType, nFeaturesOrEpoch, kClusters:int, dataSettings, valData=False):
    embeddingFile = utils.functions.GetEmbeddingFilename(modelDir, modelType, nFeaturesOrEpoch, valData)
    if not os.path.exists(embeddingFile):
        raise Exception("Embedding file {} not found".format(embeddingFile))
    
    clustersFilename = utils.functions.GetClustersFilename(modelDir, modelType, nFeaturesOrEpoch, kClusters, algorithm, valData)
    if not os.path.exists(clustersFilename):
        raise Exception("Clusters file {} not found".format(clustersFilename))

    nfeaturesOrEpochEmbedding, embedding, embeddingHash = utils.functions.LoadEmbedding(embeddingFile)
    nfeaturesOrEpochClusters, clusters, centroids, clustersHash = utils.functions.LoadClusters(clustersFilename)
    assert nFeaturesOrEpoch == nfeaturesOrEpochEmbedding
    assert nFeaturesOrEpoch == nfeaturesOrEpochClusters
    assert embeddingHash == clustersHash

    # For each cluster, find the closest samples
    distances = np.zeros((len(centroids), len(embedding)))
    sortedDistances = np.zeros((len(centroids), len(embedding)),dtype=int)
    for clusterIdx, centroid in enumerate(centroids):
        distances[clusterIdx] = np.linalg.norm(embedding - centroid, axis=1)
        sortedDistances[clusterIdx] = np.argsort(distances[clusterIdx])

    # Load data
    h5SetsLocation = utils.functions.getH5SetLocation(dataSettings['datasetName'])
    if valData:
        samplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'valSamples',subdir=dataSettings['datasetSubDir'])
    else:
        samplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'trainSamples',subdir=dataSettings['datasetSubDir'])
    if dataSettings['subbands'] is None:
        nValSamples = 6000
    else:
        nValSamples = None
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dimensionReduction', 'original', 4,samplesFilename, dataSettings=dataSettings, bufferAll = True, cacheDataset = True, nSamples=nValSamples)
    samplesHash = dataGenerator.samplesHash
    assert clustersHash == samplesHash
    images = []
    for batchX in tqdm(dataGenerator):
        images.extend(dataGenerator.ConvertToImage(batchX))
    images = np.asarray(images)

    clusterSamplesPlotsDir = os.path.join(plotsDir, 'samplesClosestToCentroids_k={}'.format(kClusters))
    os.makedirs(clusterSamplesPlotsDir, exist_ok=True)

    topN = 5
    nClusters = len(centroids)
    for clusterIdx in range(nClusters):
        closestSamples = sortedDistances[clusterIdx][:topN]
        for topIdx, sampleIdx in enumerate(closestSamples):
            uintImage = (images[sampleIdx]*255).astype(np.uint8)
            pilImage = Image.fromarray(uintImage)
            saveName = 'cluster={}_top={}.png'.format(clusterIdx, topIdx)
            pilImage.save(os.path.join(clusterSamplesPlotsDir, saveName))

def PlotMaxSilhouetteScores(clusterSearch, searchName = 'kmeans', plotLocation = None, valData = False):
    if plotLocation is None:
        plotsLocation = utils.functions.getPlotLocation(datasetName, os.path.join(modelName, searchName),plotRoot = "D:\\plots")
    else:
        plotsLocation = plotLocation
    maxValuePerk = {}
    for clusterExperiment in clusterSearch:
        methodName = clusterExperiment['methodName']
        modelDir = clusterExperiment['modelDir']
        algorithm = clusterExperiment['algorithm']
        modelType = clusterExperiment['modelType']

        if algorithm == 'kmeans':
            clusterDirName = 'kmeans_metrics'
        elif algorithm == 'gmm':
            clusterDirName = 'gmm_metrics'

        if valData: clusterDirName += '_val'
            
        clusterDir = os.path.join(modelDir, clusterDirName)
        if not os.path.exists(clusterDir):
            raise Exception("No clustering results directory found for {}. Cluster embedding first".format(methodName))
        
        clusterFiles = os.listdir(clusterDir)
        if len(clusterFiles) == 0:
            print("No cluster files found in {}. Continue with next".format(clusterDir))
            continue
        
        scoresPerEpoch = {}
        nFeaturesList = []
        methodKvalues = []
        methodIntertia = []
        methodSilhouette = []
        for clusterFile in clusterFiles:
            clusterFilename = os.path.join(clusterDir, clusterFile)
            algorithm, nFeaturesOrEpoch, kValues, metricOne, metricTwo,_ = utils.functions.LoadClusterScores(clusterFilename)
            if 'kLimit' in clusterExperiment.keys():
                if nFeaturesOrEpoch > clusterExperiment['kLimit']:
                    continue
            nFeaturesList.append(nFeaturesOrEpoch)
            methodKvalues.append(kValues)
            methodIntertia.append(metricOne)
            methodSilhouette.append(metricTwo)
            
            scoresPerEpoch[nFeaturesOrEpoch] = [kValues, metricOne, metricTwo]
            
        sortedIndices = np.argsort(nFeaturesList)
        nFeaturesList = np.asarray(nFeaturesList)[sortedIndices]
        methodKvalues = np.asarray(methodKvalues)[sortedIndices]
        methodIntertia = np.asarray(methodIntertia)[sortedIndices]
        methodSilhouette = np.asarray(methodSilhouette)[sortedIndices]

        #plotKmeans2d(os.path.join(plotsLocation,'inertia_{}_2d'.format(methodName)), nFeaturesList, methodKvalues[0], methodIntertia, modelType)
        plotKmeans2d(os.path.join(plotsLocation,'silhouette_{}_2d'.format(methodName)), nFeaturesList, methodKvalues[0], methodSilhouette, modelType)
        maxValuePerk[methodName] = [np.max(methodIntertia, axis=0), np.max(methodSilhouette, axis=0)]
    plotCompareMethods(os.path.join(plotsLocation,'silhouette_maxValues'), maxValuePerk, kValues,'silhouette') 

def EvaluateClusteringSearch(clusterSearch, searchName = 'kmeans', resultsLocation = None, valData = False):
    if resultsLocation is None:
        plotsLocation = utils.functions.getPlotLocation(datasetName, os.path.join(modelName, searchName))
    else:
        plotsLocation = resultsLocation

    maxValuePerk = {}
    valuesPerSelectedModel={}
    for clusterExperiment in clusterSearch:
        methodName = clusterExperiment['methodName']
        modelDir = clusterExperiment['modelDir']
        algorithm = clusterExperiment['algorithm']
        modelType = clusterExperiment['modelType']

        if algorithm == 'kmeans':
            clusterDirName = 'kmeans_metrics'
        elif algorithm == 'gmm':
            clusterDirName = 'gmm_metrics'

        if valData: clusterDirName += '_val'
            
        clusterDir = os.path.join(modelDir, clusterDirName)
        if not os.path.exists(clusterDir):
            raise Exception("No clustering results directory found for {}. Cluster embedding first".format(methodName))
        
        clusterFiles = os.listdir(clusterDir)
        if len(clusterFiles) == 0:
            print("No cluster files found in {}. Continue with next".format(clusterDir))
            continue
        
        scoresPerEpoch = {}
        nFeaturesList = []
        methodKvalues = []
        methodIntertia = []
        methodSilhouette = []
        for clusterFile in clusterFiles:
            clusterFilename = os.path.join(clusterDir, clusterFile)
            algorithm, nFeaturesOrEpoch, kValues, metricOne, metricTwo,_ = utils.functions.LoadClusterScores(clusterFilename)
            if 'kLimit' in clusterExperiment.keys():
                if nFeaturesOrEpoch > clusterExperiment['kLimit']:
                    continue
            nFeaturesList.append(nFeaturesOrEpoch)
            methodKvalues.append(kValues)
            methodIntertia.append(metricOne)
            methodSilhouette.append(metricTwo)  
            scoresPerEpoch[nFeaturesOrEpoch] = [kValues, metricOne, metricTwo]
            
        sortedIndices = np.argsort(nFeaturesList)
        nFeaturesList = np.asarray(nFeaturesList)[sortedIndices]
        methodKvalues = np.asarray(methodKvalues)[sortedIndices]
        methodIntertia = np.asarray(methodIntertia)[sortedIndices]
        methodSilhouette = np.asarray(methodSilhouette)[sortedIndices]

        if 'highlight' in clusterExperiment.keys():
            valuesPerSelectedModel[methodName] = [scoresPerEpoch[clusterExperiment['highlight']][1], scoresPerEpoch[clusterExperiment['highlight']][2]]
        maxValuePerk[methodName] = [np.max(methodIntertia, axis=0), np.max(methodSilhouette, axis=0)]
        
    compareMethods(maxValuePerk, valuesPerSelectedModel, logLocation=plotsLocation)  
    plotCompareMethods(os.path.join(plotsLocation,'silhouette_selectedModels'), valuesPerSelectedModel, kValues,'silhouette') 
 
def wwTest(maxValuesOne, maxValuesTwo):
    # Wald-Wolfowitz runs test (runs test)
    binaryValues = (maxValuesOne > maxValuesTwo).astype(np.int32)

    runsList = []
    tmpList = []
    for i in binaryValues:
        if len(tmpList) == 0:
            tmpList.append(i)
        elif i == tmpList[len(tmpList)-1]:
            tmpList.append(i)
        elif i != tmpList[len(tmpList)-1]:
            runsList.append(tmpList)
            tmpList = [i]
    runsList.append(tmpList)
    numRuns = len(runsList)

    # Define parameters
    R = numRuns      # number of runs
    n1 = sum(binaryValues)      # number of 1's
    n2 = len(binaryValues) - n1 # number of 0's
    n = n1 + n2      # should equal len(L)

    # compute the standard error of R if the null (random) is true
    seR = math.sqrt( ((2*n1*n2) * (2*n1*n2 - n)) / ((n**2)*(n-1)) )

    # compute the expected value of R if the null is true
    muR = ((2*n1*n2)/n) + 1

    # test statistic: R vs muR
    wwZ = (R - muR) / seR

    # test the pvalue
    p_values_one = st.norm.sf(abs(wwZ))   #one-sided
    p_values_two = st.norm.sf(abs(wwZ))*2 #twosided
    return p_values_one, p_values_two, R

def wilcoxonTest(valuesOne, valuesTwo):
    difference = valuesOne-valuesTwo
    result = wilcoxon(difference)
    return result.pvalue

def compareMethods(maxValuesPerMethod, selectedValuesPerMethod, logLocation):
    statisticsFilepath = os.path.join(logLocation, 'statistics.txt')
    with open(statisticsFilepath, 'w') as statisticsFile:
        for selectionName, valuesPerMethod in zip(['max values', 'highlighted runs'],[maxValuesPerMethod, selectedValuesPerMethod]):
            for metricIdx, metricName in enumerate(['inertia', 'silhouette']):
                statisticsFile.write("Comparison {} for {} metric\n".format(selectionName, metricName))
                for methodNameOne, maxMetricsValuesOne in valuesPerMethod.items():
                    for methodNameTwo, maxMetricsValuesTwo in valuesPerMethod.items():
                        if methodNameOne == methodNameTwo: continue
                        maxValuesOne = np.asarray(maxMetricsValuesOne[metricIdx])
                        maxValuesTwo = np.asarray(maxMetricsValuesTwo[metricIdx])
                        wwTestResults = wwTest(maxValuesOne, maxValuesTwo)
                        wilcoxonP = wilcoxonTest(maxValuesOne, maxValuesTwo)
                        results = '{}     \t{}     \t  one-sided runs test: {:.4f} \t two-sided runs test: {:.4f}, \t R: {:.5f}, \t wilcoxon p: {:.8f}'.format(methodNameOne, methodNameTwo, wwTestResults[0], wwTestResults[1], wwTestResults[2], wilcoxonP)    
                        statisticsFile.write(results)
                        statisticsFile.write('\n')
                statisticsFile.write('\n\n')

def plotCompareMethods(saveFileName, valuesPerMethod, kValues,metrics):
    # plot the figure
    #ax = plt.figure(figsize=(15, 10)).gca()
    ax = plt.figure(figsize=(6, 6)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.gca().xaxis.grid(True)

    for modelName, valuesCurrentMethod in valuesPerMethod.items():
        if metrics == 'silhouette':
            plotMax = valuesCurrentMethod[1]
        else:
            plotMax = valuesCurrentMethod[0]
        plt.plot(kValues, plotMax, label=modelName)

    if metrics == 'silhouette':
        plt.ylabel('Silhouette score')
        plt.ylim(0,1.0)
    else:
        plt.ylabel('Inertia')
    plt.legend()
    plt.xlabel('k clusters')
    #plt.xticks(range(kValues[0], kValues[-1]+1), kValues)
    plt.xlim(kValues[0],kValues[-1])
    plt.savefig(saveFileName, dpi=300, bbox_inches='tight')
    plt.close()

def plotKmeans2d(saveFileName, nFeaturesList, kValues, methodMetric,modelType):
    methodMetric = np.flip(methodMetric, axis=0)

    fig = plt.figure(figsize=(8, 10))#.gca()
    ax = plt.axes()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if np.max(methodMetric) < 1:
        im = ax.imshow(methodMetric,cmap='viridis',vmin=0, vmax=1)
    else:
        im = ax.imshow(methodMetric,cmap='viridis',vmin=0)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)

    ax.set_xticks(range(len(kValues)), kValues)
    ax.set_yticks(range(len(nFeaturesList)-1,-1,-1),nFeaturesList)
    ax.locator_params(axis='x', nbins=10)
    if modelType == 'dlModel':
        ax.set_ylabel('Epochs')
    else:
        ax.set_ylabel('m features')
    ax.set_xlabel('k clusters')

    plt.savefig(saveFileName, dpi=300, bbox_inches='tight')
    plt.close()

def plotKmeans1d(saveFilename, scores, modelType, metrics='silhouette', highlightEpoch=None, limitK=None):
    # plot the figure
    #ax = plt.figure(figsize=(12, 10)).gca()
    ax = plt.figure(figsize=(6, 6)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.grid(True)

    nFeaturesOrEpochSorted = sorted(list(scores.keys()))
    cmap = plt.cm.get_cmap('viridis')
    normalize = plt.Normalize(0, len(nFeaturesOrEpochSorted)) 
    for plotIdx, nFeaturesOrEpochNumber in enumerate(nFeaturesOrEpochSorted):
        kValuesSortedIndices = np.argsort(scores[nFeaturesOrEpochNumber][0])
        scores[nFeaturesOrEpochNumber][0] = np.asarray(scores[nFeaturesOrEpochNumber][0])[kValuesSortedIndices]
        scores[nFeaturesOrEpochNumber][1] = np.asarray(scores[nFeaturesOrEpochNumber][1])[kValuesSortedIndices]
        scores[nFeaturesOrEpochNumber][2] = np.asarray(scores[nFeaturesOrEpochNumber][2])[kValuesSortedIndices]
        
        kValuesList = scores[nFeaturesOrEpochNumber][0]
        if metrics == 'silhouette':
            plotData = scores[nFeaturesOrEpochNumber][2]
        else:
            plotData = scores[nFeaturesOrEpochNumber][1]
        if modelType == 'dlModel':
            labelName = 'epoch {}'.format(nFeaturesOrEpochNumber)
        else:
            labelName = 'n features {}'.format(nFeaturesOrEpochNumber)

        nFeaturesEpochColor = cmap(normalize(plotIdx))
        if highlightEpoch is not None:
            if nFeaturesOrEpochNumber == highlightEpoch:
                nFeaturesEpochColor = 'red'
        plt.plot(kValuesList[:len(plotData)], plotData,color=nFeaturesEpochColor, label=labelName)

    if metrics == 'silhouette':
        plt.ylabel('Mean silhouette score')
        plt.ylim(0,1.0)
    else:
        plt.ylabel('Inertia')
    plt.xlabel('k clusters')

    # if len(kValuesList)>40:
    #     xTickValues = np.arange(kValuesList[0], kValuesList[-1], 2)
    # else:
    #     xTickValues = kValuesList   
    # plt.xticks(xTickValues,xTickValues)

    if limitK is not None:
        closestToLimit = 0
        for tempKvalue in kValuesList:
            if tempKvalue <= limitK:
                if tempKvalue > closestToLimit:
                    closestToLimit = tempKvalue
        plt.xlim(kValuesList[0],closestToLimit)
    else:
        plt.xlim(kValuesList[0],kValuesList[-1])
    plt.legend(loc='upper right',prop={'size':6.5})
    plt.savefig(saveFilename, dpi=300, bbox_inches='tight')
    plt.close()

def PlotKmeansDirectory(modelDir, plotsLocation, modelType, highlightEpoch=None, valData=False, limitK=None, prefixSavename = ''):
    os.makedirs(plotsLocation, exist_ok=True)
    if highlightEpoch is None:
        saveFilename = os.path.join(plotsLocation, '{}kmeans search'.format(prefixSavename))
    else:
        saveFilename = os.path.join(plotsLocation, '{}kmeans search highlighted={}'.format(prefixSavename,highlightEpoch))

    if valData: 
        kMeansMetricsDir = os.path.join(modelDir, 'kmeans_metrics_val')
    else:
        kMeansMetricsDir = os.path.join(modelDir, 'kmeans_metrics')

    if not os.path.exists(kMeansMetricsDir):
        raise Exception("No kMeans directory found in {}".format(modelDir))
    
    kMeansFiles = os.listdir(kMeansMetricsDir)
    if len(kMeansFiles) == 0:
        raise Exception("No kMeans files found in {}. Continue with next".format(kMeansMetricsDir))
    
    scoresPerEpoch = {}
    for kMeansFile in kMeansFiles:
        kMeansFilename = os.path.join(kMeansMetricsDir, kMeansFile)
        algorithm, nFeaturesOrEpoch, kClusters, metricOne, metricTwo, savedHash = utils.functions.LoadClusterScores(kMeansFilename)
        scoresPerEpoch[nFeaturesOrEpoch] = [kClusters, metricOne, metricTwo]

    plotKmeans1d(saveFilename, scoresPerEpoch, modelType,highlightEpoch=highlightEpoch, limitK=limitK)