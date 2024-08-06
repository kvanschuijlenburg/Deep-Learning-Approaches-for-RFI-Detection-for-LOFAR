import os
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

environment = os.environ.get('OS','')
if environment == "Windows_NT":
    os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
    os.environ['PATH'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin" + os.pathsep + os.environ['PATH']

import utils as utils
from dino.dinoExperiments import GetExperiment as GetDinoExperiment
from gan.gan_utils import GanLoader
from gan.ganExperiments import GetExperiment
from utils.model_utils import defaultDataSettings

datasetName = "LOFAR_L2014581 (recording)"
datasetSubdir = 'dataset250k'

# Experiment settings
debug = False

if environment == "Windows_NT":
    verbose=1
else:  
    verbose=2
    debug=False

if len(sys.argv) == 2: # One more since the fullfilename is one arg
    experiment = int(sys.argv[1])
else:
    experiment = 1

kValues = range(5,31,2)

def predictModel(experimentNumber, run, dataset, predictionsFilename = None, evaluationFilename = None):
    # Get experiment settings and model name
    dataSettings, ganModelSettings, dinoModelSettings = GetExperiment(experimentNumber,debug=False)
    modelName = utils.models.getGanModelName(ganModelSettings, dataSettings, dinoModelSettings)
    
    # Instantiate the model
    if ganModelSettings['loadDinoEncoder'] is not None:
        dinoModelSettings,dinoDataSettings,dinoGanModelSettings = GetDinoExperiment(ganModelSettings['loadDinoEncoder'] ,False,False)
    print(modelName)

    # Get the dataset locations
    offsetSamples = None
    if dataset == 'train':
        nSamples = 1000
        raise Exception("how many training samples?")
        h5SetsLocation = utils.functions.getH5SetLocation(defaultDataSettings['datasetName'])
        samplesFilename = utils.functions.getDatasetLocation(defaultDataSettings['datasetName'], 'trainSamples',subdir=defaultDataSettings['datasetSubDir'])
    elif dataset == 'val':
        nSamples = 1000 # TODO: make a global variable for this
        h5SetsLocation = utils.functions.getH5SetLocation(defaultDataSettings['datasetName'])
        samplesFilename = utils.functions.getDatasetLocation(defaultDataSettings['datasetName'], 'valSamples',subdir=defaultDataSettings['datasetSubDir'])
    elif dataset == 'test':
        nSamples = None
        h5SetsLocation = utils.functions.getH5SetLocation("Haomin Sun et al. test set")
        samplesFilename = utils.functions.getDatasetLocation("Haomin Sun et al. test set", 'testSamples')
    elif dataset == 'valTest':
        nSamples = 1000
        offsetSamples = 6000
        h5SetsLocation = utils.functions.getH5SetLocation(defaultDataSettings['datasetName'])
        samplesFilename = utils.functions.getDatasetLocation(defaultDataSettings['datasetName'], 'valSamples',subdir=defaultDataSettings['datasetSubDir'])
    else:
        raise Exception("Unknown dataset: {}".format(dataset))

    # Construct location for plots
    # TODO: plotsLocation seems only to be used in GanLoader. Can we remove this?
    plotsLocation = utils.functions.getPlotLocation(dataSettings['datasetName'], modelName)

    # Loading data
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'gan', 'val', dataSettings['inputShape'][2], samplesFilename, ganParameters=ganModelSettings,dataSettings=dataSettings, bufferAll=True, cacheDataset=True, nSamples = nSamples, offsetSamples=offsetSamples)
    samplesHash = dataGenerator.samplesHash
    dataShape = dataGenerator.getDataShape()

    # Load the model
    ganLoader = GanLoader(dataSettings, ganModelSettings, plotsLocation, modelName,dataShape=dataShape, run=run,dinoModelSettings=dinoModelSettings)
    ganLoader.makeModel()
    epoch = ganLoader.restoreWeights(restorePartial=True)
    modelDir = ganLoader.trainModelDir

    if epoch == 0:
        raise Exception("No checkpoint found. Train the model first.")
    
    print("Checkpoint found, loaded epoch: {}".format(epoch))
    gan = ganLoader.getModel()
    
    # Predict features
    print("Predict embedding")
    predictions = []
    dataX = []
    trueLabels = []
    for batchData in tqdm(dataGenerator):
        batchDataX, batchDataY = batchData
        batchPrediction = gan.generator.predict(batchDataX,verbose=0)
        predictions.extend(batchPrediction)
        trueLabels.extend(batchDataY)
        dataX.extend(batchDataX)
    trueLabels = np.asarray(trueLabels)
    dataX = np.asarray(dataX)
    predictions = np.asarray(predictions)

    if predictionsFilename is not None:
        print("Convert data to color images")
        colorImages = []
        for batchData in tqdm(dataGenerator):
            batchDataX, batchDataY = batchData
            batchImage = dataGenerator.ConvertToImage(batchDataX)
            colorImages.extend(batchImage)
        colorImages = np.asarray(colorImages)

        predictionsData = [epoch, dataX, colorImages, trueLabels, predictions, samplesHash]
        with open(predictionsFilename, 'wb') as file:
            pickle.dump(predictionsData, file)

    if evaluationFilename is not None:
        print("Calculate model's metrics")
        metricsResults = []
        predictions = predictions[:,:,:,1]>0.5
        trueLabels = trueLabels[:,:,:,1].astype(bool)
        for prediction, label in zip(predictions, trueLabels):
            tp = np.sum(np.logical_and(prediction, label))
            fp = np.sum(np.logical_and(prediction, np.logical_not(label)))
            tn = np.sum(np.logical_and(np.logical_not(prediction), np.logical_not(label)))
            fn = np.sum(np.logical_and(np.logical_not(prediction), label))
            metricsResults.append([tp,fp,tn,fn])
        metricsResults = np.asarray(metricsResults)
        saveMetricsResult = [metricsResults, samplesHash]
        with open(evaluationFilename, 'wb') as file:
            pickle.dump(saveMetricsResult, file)
        
def Predict(experiment, dataset, run=0):
    ganDataSettings, ganModelSettings, dinoModelSettings = GetExperiment(experiment,False)
    ganName = utils.models.getGanModelName(ganModelSettings, ganDataSettings, dinoModelSettings)
    experimentDir, _ = utils.functions.getModelLocation(os.path.join('gan',ganName),'run_'+str(run))

    if dataset == 'train':
        predictionsDir = os.path.join(experimentDir ,'predictions')
    elif dataset == 'val':
        predictionsDir = os.path.join(experimentDir ,'predictions_val')
    elif dataset == 'test':
        predictionsDir = os.path.join(experimentDir ,'predictions_test')
    elif dataset == 'valTest':
        predictionsDir = os.path.join(experimentDir ,'predictions_valTest')

    os.makedirs(predictionsDir, exist_ok=True)
    predictionsFilename = os.path.join(predictionsDir, 'predictions_last_epoch.pkl')
    if not os.path.exists(predictionsFilename):
        print("No predictions for {}. Load model and predict dataset".format(predictionsFilename))
        predictModel(experiment, run, dataset, predictionsFilename=predictionsFilename)

def CalculateMetrics(experiment,dataset, run=0):
    ganDataSettings, ganModelSettings, dinoModelSettings = GetExperiment(experiment,False)
    ganName = utils.models.getGanModelName(ganModelSettings, ganDataSettings, dinoModelSettings)
    experimentDir, _ = utils.functions.getModelLocation(os.path.join('gan',ganName),'run_'+str(run))

    if dataset == 'train':
        evaluationFilename = os.path.join(experimentDir ,'evaluation.pkl')
    elif dataset == 'val':
        evaluationFilename = os.path.join(experimentDir ,'evaluation_val.pkl')
    elif dataset == 'test':
        evaluationFilename = os.path.join(experimentDir ,'evaluation_test.pkl')
    elif dataset == 'valTest':
        evaluationFilename = os.path.join(experimentDir ,'evaluation_valTest.pkl')
    
    if not os.path.exists(evaluationFilename):
        print("No evaluation found for {}. Start predicting model".format(evaluationFilename))
        predictModel(experiment, run, dataset, evaluationFilename = evaluationFilename)
    else:
        print("Evaluation already done for {}. Skip".format(evaluationFilename))

def calcEmbedding(experiment, dataset ,run=None):
    # Get experiment settings and model name
    dataSettings, ganModelSettings, dinoModelSettings = GetExperiment(experiment,debug)
    modelName = utils.models.getGanModelName(ganModelSettings, dataSettings, dinoModelSettings)

    # Get the dataset locations
    h5SetsLocation = utils.functions.getH5SetLocation(dataSettings['datasetName'])
    plotsLocation = utils.functions.getPlotLocation(dataSettings['datasetName'], modelName)


    if dataset == 'train':
        samplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'trainSamples',subdir=dataSettings['datasetSubDir'])
    elif dataset == 'val':
        samplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'valSamples',subdir=dataSettings['datasetSubDir'])
    elif dataset == 'test':
        samplesFilename = utils.functions.getDatasetLocation("Haomin Sun et al. test set", 'testSamples')
    
    # Load data generator
    nSamples = 6000
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'gan', 'train', dataSettings['inputShape'][2], samplesFilename, ganParameters=ganModelSettings,dataSettings=dataSettings, bufferAll=True, cacheDataset=True, nSamples = nSamples)
    samplesHash = dataGenerator.samplesHash
    dataShape = dataGenerator.getDataShape()

    # Instantiate the model
    ganLoader = GanLoader(dataSettings, ganModelSettings, plotsLocation, modelName,dataShape=dataShape, run=run,dinoModelSettings=dinoModelSettings)
    modelDir = ganLoader.trainModelDir
    ganLoader.makeModel()
    epoch = ganLoader.restoreWeights(restorePartial=True)
    if epoch == 0:
        raise Exception("No checkpoint found. Train the model first.")

    # Before loading the checkpoint, check if the embedding already exists
    if dataset == 'train':
        embeddingDir = os.path.join(modelDir, 'embedding')
    elif dataset == 'val':
        embeddingDir = os.path.join(modelDir, 'embedding_val')
    elif dataset == 'test':
        embeddingDir = os.path.join(modelDir, 'embedding_test')
    os.makedirs(embeddingDir, exist_ok=True)
    embeddingFilename = os.path.join(embeddingDir, 'embedding_epoch={}.pkl'.format(epoch))

    # Check if embedding exists
    if os.path.exists(embeddingFilename):
        existingEpoch, _, _ = utils.functions.LoadEmbedding(embeddingFilename, samplesHash)
        assert existingEpoch == epoch
        # utils.functions.SaveEmbedding(embeddingFilename, embedding, epoch, samplesHash)
        print("RFI-GAN embedding already calculated")
        return

    print("Checkpoint found, evaluate epoch {}".format(epoch))
    gan = ganLoader.getModel()

    # Predict features
    print("Predict embedding")
    embedding = []
    for batchData in tqdm(dataGenerator):
        batchDataX, batchDataY = batchData
        batchEmbedding = gan.generator.encoder(batchDataX, training=False)[0]
        embedding.extend(batchEmbedding)
    embedding = np.asarray(embedding)
    embedding = np.reshape(embedding, (embedding.shape[0], -1))
    utils.functions.SaveEmbedding(embeddingFilename, embedding, epoch, samplesHash)

def measureInferenceTime(experiment,batchSize, repeat=1):
    # Get experiment settings and model name
    dataSettings, ganModelSettings, dinoModelSettings = GetExperiment(experiment,debug=False)
    modelName = utils.models.getGanModelName(ganModelSettings, dataSettings, dinoModelSettings)

    dataSettings['batchSize'] = batchSize # overide batch size
    
    # Instantiate the model
    if ganModelSettings['loadDinoEncoder'] is not None:
        dinoModelSettings,dinoDataSettings,dinoGanModelSettings = GetDinoExperiment(ganModelSettings['loadDinoEncoder'] ,False,False)
    print("Mearure inference time for model {}".format(modelName))

    # Get the dataset locations
    offsetSamples = None
    nSamples = 6000
    h5SetsLocation = utils.functions.getH5SetLocation(defaultDataSettings['datasetName'])
    samplesFilename = utils.functions.getDatasetLocation(defaultDataSettings['datasetName'], 'trainSamples',subdir=defaultDataSettings['datasetSubDir'])

    # Loading data
    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'gan', 'val', dataSettings['inputShape'][2], samplesFilename, ganParameters=ganModelSettings,dataSettings=dataSettings, bufferAll=True, cacheDataset=True, nSamples = nSamples, offsetSamples=offsetSamples)
    dataShape = dataGenerator.getDataShape()

    # Load the model
    ganLoader = GanLoader(dataSettings, ganModelSettings, modelName=modelName,dataShape=dataShape, run=0,dinoModelSettings=dinoModelSettings)
    ganLoader.makeModel()
    #epoch = ganLoader.restoreWeights(restorePartial=True)
    #print("Checkpoint found, loaded epoch: {}".format(epoch))
    gan = ganLoader.getModel()
    
    results = []
    for run in tqdm(range(repeat), desc="Repetition", position=0):
        startTime = time.time()
        predictions = []
        for batchData in tqdm(dataGenerator, desc="Batch", position=1):
            batchDataX, batchDataY = batchData
            batchPrediction = gan.generator.predict(batchDataX,verbose=0)
            predictions.extend(batchPrediction)
        endTime = time.time()
        inferenceTime = endTime - startTime
        results.append(inferenceTime)
    print("Inference time of {} samples: {}".format(nSamples,results))
    return results