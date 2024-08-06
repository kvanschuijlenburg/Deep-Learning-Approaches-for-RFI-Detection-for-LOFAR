import os
import pickle

import numpy as np
from tqdm import tqdm
from sklearn import manifold

import utils as utils

def tsneSearch(dataX, embeddingX, datasetName, modelName, clusters = None, kClusters = None):
    nFeatures = embeddingX.shape[1]
    if kClusters is None:
        plotsLocation = utils.functions.getPlotLocation(datasetName, os.path.join('tsne', "{}_{}".format(modelName,nFeatures)),plotRoot = "D:\\plots")
    else:
        plotsLocation = utils.functions.getPlotLocation(datasetName, os.path.join('tsne', "{}_{}_clusters_{}".format(modelName,nFeatures, kClusters)),plotRoot = "D:\\plots")

    n_components = 2
    nIterations = 20000
    perplexities = [30, 20, 10, 5, 50, 75, 100]#, 150, 200, 300, 500]

    for perplexity in tqdm(perplexities,desc="perplexity",position=0, leave=False):
        # Fit tsne on data
        tsne = manifold.TSNE(n_components=n_components, init="random", random_state=0, perplexity=perplexity, learning_rate='auto', n_iter=nIterations) # learning_rate='auto'
        tsneFeatures = tsne.fit_transform(embeddingX)

        # Visualize tsne fit
        scatterTitle = "scatter perplexity {}".format(perplexity)
        visualizationTitle = "visualization perplexity {}".format(perplexity)
        utils.plotter.tsneScatter(tsneFeatures, plotsLocation, scatterTitle)
        if clusters is not None:
            clusterLabels, clusteringAlgorithm = clusters
            clusterScatterTitle = "{} cluster scatter perplexity {}".format(clusteringAlgorithm, perplexity)
            utils.plotter.tsneClusterScatter(tsneFeatures, plotsLocation, clusterScatterTitle, clusterLabels)
        utils.plotter.tsneVisualization(tsneFeatures, dataX, plotsLocation, visualizationTitle)

def tsne(embeddingX, perplexity):
    n_components = 2
    nIterations = 20000
    tsne = manifold.TSNE(n_components=n_components, init="random", random_state=0, perplexity=perplexity, learning_rate='auto', n_iter=nIterations)
    tsneFeatures = tsne.fit_transform(embeddingX)
    return tsneFeatures

def tsneFromObservations():
    nTrainingSamples = None
    amplitude = True

    datasetName = "LOFAR_L2014581 (recording)"
    modelName = "observations"
    if amplitude: modelName = "amplitudes_" + modelName

    dataX = utils.datasets.getSingleH5Set(datasetName, 'train.h5', nTrainingSamples)
    if amplitude: dataX = np.stack((dataX[:,:,:,0],dataX[:,:,:,2],dataX[:,:,:,4],dataX[:,:,:,6]),axis=-1)
    embeddingX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]*dataX.shape[2]*dataX.shape[3]))

    tsneSearch(dataX, embeddingX, datasetName, modelName)

def tsneEmbeddings(compareModels):
    for model in compareModels:
        methodName = model['method']
        tsnePerplexity = model['perplexity']
        experimentModelDir = os.path.join(model['modelDir'],"clusterPipelineData")
        os.makedirs(experimentModelDir,exist_ok=True)

        if 'epoch' in model.keys():
            embeddingFilename = os.path.join(experimentModelDir, 'embedding_epoch={}.pkl'.format(model['epoch']))
            tsneFilename = os.path.join(experimentModelDir, 'tsne_epoch={}_perplexity={}.pkl'.format(model['epoch'], tsnePerplexity))
        else:
            embeddingFilename = os.path.join(experimentModelDir, 'embedding_nFeatures={}.pkl'.format(model['nFeatures']))
            tsneFilename = os.path.join(experimentModelDir, 'tsne_nFeatures={}_perplexity={}.pkl'.format(model['nFeatures'],tsnePerplexity))
        
        if os.path.exists(tsneFilename):
            print("tsne features for {} already exist.".format(methodName))
            continue
        
        print("Start tsne for {}.".format(methodName))
        nFeaturesOrEpoch, embedding, savedHash = utils.functions.LoadEmbedding(embeddingFilename)
        tsneFeatures = tsne.tsne(embedding, perplexity=tsnePerplexity)
        utils.functions.SaveTsneFeatures(tsneFilename, tsneFeatures,nFeaturesOrEpoch, savedHash)

def TsneSearchEmbedding(modelDir, modelType, nFeaturesOrEpoch = None, valData=False):
    if valData: 
        embeddingPath = os.path.join(modelDir, 'embedding_val')
    else:
        embeddingPath = os.path.join(modelDir, 'embedding')
    if not os.path.exists(embeddingPath):
        raise Exception("No embedding directory found in {}".format(modelDir))
    
    # Get required embedding files
    embeddingFiles = os.listdir(embeddingPath)
    if nFeaturesOrEpoch is not None:
        if modelType == 'dlModel':
            embeddingFilename = 'embedding_epoch={}.pkl'.format(nFeaturesOrEpoch)
        else:
            embeddingFilename = 'embedding_nFeatures={}.pkl'.format(nFeaturesOrEpoch)
        embeddingFiles = [embeddingFilename]
    if len(embeddingFiles) == 0:
        raise Exception("No embedding files found in {}".format(embeddingPath))

    tsneDirName = 'tsne_features'
    if valData: tsneDirName = 'tsne_features_val'
    tsneDir = os.path.join(modelDir, tsneDirName)  
    os.makedirs(tsneDir, exist_ok=True)

    for embeddingFile in embeddingFiles:
        embeddingFilename = os.path.join(embeddingPath, embeddingFile)
        nFeaturesOrEpoch, embedding, savedHash = utils.functions.LoadEmbedding(embeddingFilename)
        if np.isnan(embedding).any():
            print("Found NaN value in embedding {}. Continue".format(embeddingFile))
            continue

        perplexities = [30, 20, 50, 10, 5, 75, 100]
        for perplexity in tqdm(perplexities,desc="perplexity",position=0, leave=False):
            if modelType == 'dlModel':
                tsneName = 'tsne_epoch={}_perplexity={}.pkl'.format(nFeaturesOrEpoch,perplexity)
            else:
                tsneName = 'tsne_nFeatures={}_perplexity={}.pkl'.format(nFeaturesOrEpoch,perplexity)
            tsneFilename = os.path.join(tsneDir, tsneName)
            if os.path.exists(tsneFilename):
                with open(tsneFilename, 'rb') as file:
                    data = pickle.load(file)
                #Fix this: # len(data.shape) is indeed 2, but because it has size 5992x2. It are just the tsne features. nFeaturesOrEpoch and savedHash must be added
                if len(data)!=3:
                    if data.shape == (5992,2):
                        utils.functions.TsneVerifyAdd(tsneFilename,nFeaturesOrEpoch,savedHash)
                    else:
                        raise Exception("tsne file cannot be automatically fixed")
                continue

            # Fit tsne on data
            tsne = manifold.TSNE(n_components=2, init="random", random_state=0, perplexity=perplexity, learning_rate='auto', n_iter=20000) # learning_rate='auto'
            tsneFeatures = tsne.fit_transform(embedding)

            utils.functions.SaveTsneFeatures(tsneFilename,tsneFeatures, nFeaturesOrEpoch, savedHash)

def PlotTsneScatter(modelDir, plotLocation,epoch, valData=False):
    if valData: 
        featuresModelDir = os.path.join(modelDir, 'tsne_features_val')
    else:
        featuresModelDir = os.path.join(modelDir, 'tsne_features')
    if not os.path.exists(featuresModelDir):
        raise Exception("No tsne features directory found in {}".format(modelDir))
    
    epochPlotLocation = os.path.join(plotLocation, 'epoch {}'.format(epoch))
    os.makedirs(epochPlotLocation, exist_ok=True)

    # Load embedding and data
    perplexities = [5, 10, 20, 30, 50, 75, 100]
    for perplexity in perplexities:
        featuresFilename = os.path.join(featuresModelDir, 'tsne_epoch={}_perplexity={}.pkl'.format(epoch,perplexity))
        if os.path.exists(featuresFilename)==False:
            continue

        nFeaturesOrEpoch, tsneFeatures, savedHash = utils.functions.LoadTsneFeatures(featuresFilename)
        if nFeaturesOrEpoch != epoch:
            continue
        utils.plotter.tsneScatter(tsneFeatures, epochPlotLocation, plotTitle='tsne_epoch={}_perplexity={}'.format(epoch,perplexity))

def PlotTsneKmeans(modelDir, plotLocation, modelType, algorithm, nFeaturesOrEpoch, kClusters, valData=False):
    if valData:
        featuresModelDir = os.path.join(modelDir, 'tsne_features_val')
    else:
        featuresModelDir = os.path.join(modelDir, 'tsne_features')
    clustersFilename = utils.functions.GetClustersFilename(modelDir,modelType,nFeaturesOrEpoch,kClusters,algorithm, valData)

    if not os.path.exists(featuresModelDir):
        raise Exception("No tsne features directory found in {}".format(modelDir))
    if not os.path.exists(clustersFilename):
        raise Exception("No kmeans clusters file found in {}".format(clustersFilename))
    
    epochPlotLocation = os.path.join(plotLocation, 'epoch {}'.format(nFeaturesOrEpoch))
    os.makedirs(epochPlotLocation, exist_ok=True)

    nFeaturesOrEpochClusters, clusters, centroids, clustersHash = utils.functions.LoadClusters(clustersFilename)
    assert nFeaturesOrEpochClusters == nFeaturesOrEpoch

    # Load embedding and data
    perplexities = [5, 10, 20, 30, 50, 75, 100]
    for perplexity in perplexities:
        featuresFilename = os.path.join(featuresModelDir, 'tsne_epoch={}_perplexity={}.pkl'.format(nFeaturesOrEpoch,perplexity))
        if os.path.exists(featuresFilename)==False:
            continue

        nFeaturesOrEpochTsne, tsneFeatures, tsneHash = utils.functions.LoadTsneFeatures(featuresFilename)
        assert clustersHash == tsneHash
        if nFeaturesOrEpochTsne != nFeaturesOrEpoch:
            continue
        saveName = 'tsneClusters_epoch={}_clusters={}_perplexity={}'.format(nFeaturesOrEpoch,kClusters, perplexity)
        utils.plotter.tsneClusterScatter(tsneFeatures, epochPlotLocation, plotTitle=saveName, clusters=clusters, plotCenters=True)

def PlotTsneVisualization(modelDir, plotLocation,epoch,dataSettings, valData=False):
    if valData: 
        featuresModelDir = os.path.join(modelDir, 'tsne_features_val')
    else:
        featuresModelDir = os.path.join(modelDir, 'tsne_features')
    if not os.path.exists(featuresModelDir):
        raise Exception("No tsne features directory found in {}".format(modelDir))
    
    epochPlotLocation = os.path.join(plotLocation, 'epoch {}'.format(epoch))
    os.makedirs(epochPlotLocation, exist_ok=True)

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
    images = []
    for batchX in dataGenerator:
        images.extend(dataGenerator.ConvertToImage(batchX))
    images = np.asarray(images)  

    # Load embedding and data
    perplexities = [5, 10, 20, 30, 50, 75, 100]
    for perplexity in perplexities:
        featuresFilename = os.path.join(featuresModelDir, 'tsne_epoch={}_perplexity={}.pkl'.format(epoch,perplexity))
        if os.path.exists(featuresFilename)==False:
            continue

        nFeaturesOrEpoch, tsneFeatures, savedHash = utils.functions.LoadTsneFeatures(featuresFilename)
        assert savedHash == samplesHash
        if nFeaturesOrEpoch != epoch:
            continue

        utils.plotter.tsneVisualization(tsneFeatures,images,epochPlotLocation,plotTitle='tsneVisualization_epoch={}_perplexity={}'.format(epoch,perplexity))