import os

import numpy as np
from tqdm import tqdm

import unsupervised.tsne as tsne
import utils as utils
from dino.dinoExperiments import GetDinoModelDir, GetExperiment
from dino.dino_utils import DinoLoader

datasetName = "LOFAR_L2014581 (recording)"
datafolderSubdir = 'dataset250k'
overideBatchSize = 10

def PlotTsne(experiment, epoch, plotsLocation, valData = True, prefixSavename = '', kValues = list(range(5,51))):
    modelDir = GetDinoModelDir(experiment=experiment)
    dinoModelSettings,dataSettings,ganModelSettings = GetExperiment(experiment)

    # Load data for tsne plots
    h5SetsLocation = utils.functions.getH5SetLocation(datasetName)
    if valData:
        samplesFilename = utils.functions.getDatasetLocation(datasetName, 'valSamples',subdir=datafolderSubdir)
    else:
        samplesFilename = utils.functions.getDatasetLocation(datasetName, 'trainSamples',subdir=datafolderSubdir)
    test_dataset = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dino', 'test', dinoModelSettings['nChannels'],samplesFilename, dinoModelSettings, dataSettings=dataSettings, bufferAll = True, cacheDataset = True, nSamples=6000) 

    perplexity = 30
    embeddingFilename = utils.functions.GetEmbeddingFilename(modelDir,'dlModel',epoch, valData=valData)
    tsneFilename = utils.functions.GetTsneFilename(modelDir,'dlModel',epoch,perplexity,valData=valData)
    if os.path.exists(tsneFilename)==False:
        loadedEpoch, embedding, savedHash = utils.functions.LoadEmbedding(embeddingFilename)
        assert loadedEpoch == epoch, "Loaded epoch does not match requested epoch"
        tsneFeatures = tsne.tsne(embedding, perplexity=perplexity)
        utils.functions.SaveEmbedding(tsneFilename, tsneFeatures, epoch, savedHash)
    else:
        nFeaturesOrEpoch, tsneFeatures, savedHash = utils.functions.LoadTsneFeatures(tsneFilename)
    
    utils.plotter.tsneScatter(tsneFeatures, plotsLocation, '{}tSNE epoch={}, perp={}'.format(prefixSavename,epoch,perplexity))#tsneFeatures, modelDir, plotsLocation, epoch, perplexity, valData=valData)
    imageX = []
    for batchX in test_dataset:
        imageX.extend(test_dataset.ConvertToImage(batchX))
    imageX= np.asarray(imageX)
    utils.plotter.tsneVisualization(tsneFeatures, imageX, plotsLocation, '{}tSNE visualization epoch={}, perp={}'.format(prefixSavename,epoch,perplexity))

def CalcEmbedding(experiment, epoch, valData=False):
    dinoModelSettings,dataSettings,ganModelSettings = GetExperiment(experiment, False)
    modelName = utils.models.getDinoModelName(dinoModelSettings,dataSettings,ganModelSettings)

    # Get model and dataset locations
    modelDir, _ = utils.functions.getModelLocation(os.path.join('dino',modelName))
    h5SetsLocation = utils.functions.getH5SetLocation(dataSettings['datasetName'])
    plotLocation = utils.functions.getPlotLocation(dataSettings['datasetName'], modelName)
    if valData:
        samplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'valSamples',subdir=dataSettings['datasetSubDir'])
        embeddingDir = os.path.join(modelDir, "embedding_val")
    else:
        samplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'trainSamples',subdir=dataSettings['datasetSubDir'])
        embeddingDir = os.path.join(modelDir, "embedding")
        
    os.makedirs(embeddingDir, exist_ok=True)
    embeddingFilename = os.path.join(embeddingDir, 'embedding_epoch={}.pkl'.format(epoch))
    if os.path.exists(embeddingFilename):
        print("Embedding of experiment {}, epoch {} already exist".format(modelName, epoch))
        return
    
    print("Calculate embedding of experiment {}, epoch {}.".format(modelName, epoch))
    
    # Load model and data
    dinoLoader = DinoLoader(dinoModelSettings, dataSettings, ganModelSettings)
    model, loadedEpoch = dinoLoader.loadTeacher(loadEpoch=epoch)
    assert loadedEpoch == epoch

    dataSettings['batchSize'] = overideBatchSize
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dino', 'original',dinoModelSettings['nChannels'] ,samplesFilename, dinoModelSettings, dataSettings=dataSettings, bufferAll = True, cacheDataset=True,nSamples=6000)
    samplesHash = dataGenerator.samplesHash

    epochPlotsLocation = os.path.join(plotLocation, 'results_epoch_{}'.format(loadedEpoch))
    os.makedirs(epochPlotsLocation, exist_ok=True)

    embedding = []
    for batchX in tqdm(dataGenerator):
        embedding.extend(model.predictEmbedding(batchX))
    embedding = np.asarray(embedding)

    utils.functions.SaveEmbedding(embeddingFilename, embedding,loadedEpoch, samplesHash)