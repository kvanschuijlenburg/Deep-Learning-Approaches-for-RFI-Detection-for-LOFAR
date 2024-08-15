import os
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils as utils
from dino.dino_utils import DinoLoader
from dino.dinoExperiments import GetExperiment as GetDinoExperiment
from gan.gan_utils import GanLoader
from gan.ganExperiments import GetExperiment


# Experiment settings 
tensorBoardModelStatistics = False
tensorBoardModelPerformance = False
tensorBoardSaveEmbedding = False
cacheDinoEmbedding = False
verbose=1

def train(experiment, run=None):
    # Get experiment settings and model name
    dataSettings, ganModelSettings, dinoModelSettings = GetExperiment(experiment)
    modelName = utils.models.getGanModelName(ganModelSettings, dataSettings, dinoModelSettings)
    print(modelName)

    if ganModelSettings['earlyStopping'] is None:
        nTrainEpochs = ganModelSettings['scheduledEpochs']
    else:
        nTrainEpochs = ganModelSettings['earlyStopping']

    # Prior check if the run has been finished
    if run is not None:
        #self.logName = tf.Variable("run_{}_{}".format(run, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        trainModelDir, _ = utils.functions.getModelLocation(os.path.join('gan',modelName),'run_'+str(run))
        print('search for checkpoint in: ',trainModelDir)
        lastCheckpointFilenameOne = os.path.join(trainModelDir, "ckpt-{}.INDEX".format(nTrainEpochs))
        lastCheckpointFilenameTwo = os.path.join(trainModelDir, "ckpt-{}.index".format(nTrainEpochs))
        if os.path.exists(lastCheckpointFilenameOne) or os.path.exists(lastCheckpointFilenameTwo):
            print("GAN experiment {}: Checkpoint {} already exists for run {}".format(experiment,nTrainEpochs,run))
            return

    # Get the dataset locations
    h5SetsLocation = utils.functions.getH5SetLocation(dataSettings['datasetName'])
    trainSamplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'trainSamples',subdir=dataSettings['datasetSubDir'])
    valSamplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'valSamples',subdir=dataSettings['datasetSubDir'])

    # Construct location for plots
    plotsLocation = utils.functions.getPlotLocation(dataSettings['datasetName'], modelName)

    # Loading data
    trainGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'gan', 'train', dataSettings['inputShape'][2], trainSamplesFilename, ganParameters=ganModelSettings,dataSettings=dataSettings, bufferAll=True, cacheDataset=True, nSamples = dataSettings['nTrainingSamples'])
    valGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'gan', 'val', dataSettings['inputShape'][2],valSamplesFilename, ganParameters=ganModelSettings, dataSettings=dataSettings, bufferAll=True, cacheDataset=True, nSamples=dataSettings['nValSamples'])
    dataShape = trainGenerator.getDataShape()

    if ganModelSettings['metadataStyleInput'] is not None:
        metadataStyleVectorSize=trainGenerator.setMetadataStyleInput(ganModelSettings['metadataStyleInput'])
        valGenerator.setMetadataStyleInput(ganModelSettings['metadataStyleInput'])
    else:
        metadataStyleVectorSize=None

    # Instantiate the model
    if ganModelSettings['loadDinoEncoder'] is not None:
        dinoModelSettings,dinoDataSettings,dinoGanModelSettings = GetDinoExperiment(ganModelSettings['loadDinoEncoder'])
    else:
        dinoModelSettings
    ganLoader = GanLoader(dataSettings, ganModelSettings, plotsLocation, modelName,dataShape=dataShape, run=run, dinoModelSettings=dinoModelSettings)
    ganLoader.makeModel(metadataStyleVectorSize)
    startEpoch = ganLoader.restoreWeights()

    if cacheDinoEmbedding and ganModelSettings['loadDinoEncoder'] is not None:
        if ganModelSettings['concatDino'] or ganModelSettings['modConv']:
            trainDinoFeaturesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'dinoEmbeddingTrain_nSamples={}'.format( dataSettings['nTrainingSamples']),subdir=dataSettings['datasetSubDir'])
            valDinoFeaturesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'dinoEmbeddingVal_nSamples={}'.format( dataSettings['nValSamples']),subdir=dataSettings['datasetSubDir'])
            
            cacheTrain = True
            cacheVal = True
            if os.path.exists(trainDinoFeaturesFilename):
                with open(trainDinoFeaturesFilename, 'rb') as dinoTrainFeaturesFile:
                    trainHash, trainDinoFeatures = pickle.load(dinoTrainFeaturesFile)
                    if trainHash == trainGenerator.samplesHash:
                        cacheTrain = False

            if os.path.exists(valDinoFeaturesFilename):
                with open(valDinoFeaturesFilename, 'rb') as dinoValFeaturesFile:
                    valHash, valDinoFeatures = pickle.load(dinoValFeaturesFile)
                    if valHash == valGenerator.samplesHash:
                        cacheVal = False

            if cacheTrain or cacheVal:
                dinoModelSettings,dinoDataSettings,dinoGanModelSettings = GetDinoExperiment(ganModelSettings['loadDinoEncoder'])
                dinoLoader = DinoLoader(dinoModelSettings, dinoDataSettings, dinoGanModelSettings)
                dinoTeacher, loadedEpoch = dinoLoader.loadTeacher(ganModelSettings['loadDinoEpoch'])
                print("Loaded DINO teacher to cache train or val from epoch {}.".format(loadedEpoch))

                if cacheTrain:
                    print("Calculate train embedding")
                    trainDinoFeatures = []
                    for batch in tqdm(trainGenerator):
                        batchX = batch[0]
                        predictedX, _ = dinoTeacher.backbone(batchX)
                        trainDinoFeatures.extend(predictedX)
                    trainDinoFeatures = np.asarray(trainDinoFeatures)
                    with open(trainDinoFeaturesFilename, 'wb') as dinoTrainFeaturesFile:
                        pickle.dump([trainGenerator.samplesHash, trainDinoFeatures], dinoTrainFeaturesFile)

                if cacheVal:
                    print("Calculate val embedding")
                    valDinoFeatures = []
                    for batch in tqdm(valGenerator):
                        batchX = batch[0]
                        predictedX, _ = dinoTeacher.backbone(batchX)
                        valDinoFeatures.extend(predictedX)
                    valDinoFeatures = np.asarray(valDinoFeatures)
                    with open(valDinoFeaturesFilename, 'wb') as dinoValFeaturesFile:
                        pickle.dump([valGenerator.samplesHash, valDinoFeatures], dinoValFeaturesFile)

            trainGenerator.AppendDinoFeatures(trainDinoFeatures)
            valGenerator.AppendDinoFeatures(valDinoFeatures)
            ganLoader.gan.generator.concatDinoEncoder = True

    if startEpoch == 0:
        if ganModelSettings['loadDinoEncoder'] is not None:
            print("No checkpount found. Loading encoder weights from DINO model.")
            dinoModelSettings,dinoDataSettings,dinoGanModelSettings = GetDinoExperiment(ganModelSettings['loadDinoEncoder'])
            dinoLoader = DinoLoader(dinoModelSettings, dinoDataSettings, dinoGanModelSettings)
            encoderWeights, loadedDinoEpoch = dinoLoader.loadEncoderWeights(ganModelSettings['loadDinoEpoch'])
            if loadedDinoEpoch == 0:
                raise Exception("No DINO checkpoint found. DINO-train the model first.")
            ganLoader.SetDinoEncoderWeights(encoderWeights)
            print("Loaded DINO weights from epoch {}.".format(loadedDinoEpoch))
        else:
            print("No checkpoint found. Training from scratch.")
    elif startEpoch >= nTrainEpochs:
        print("Training already finished. Return.")
        return
    else:
        print("Checkpoint found, continue training from epoch {}".format(startEpoch+1))

    gan = ganLoader.getModel()
    if tensorBoardSaveEmbedding:
        callbacks = ganLoader.loadCallbacks(tensorBoardModelStatistics=tensorBoardModelStatistics,tensorBoardModelPerformance = tensorBoardModelPerformance, tensorBoardModelEmbeddingGenerator = valGenerator)
    else:
        callbacks = ganLoader.loadCallbacks(tensorBoardModelStatistics=tensorBoardModelStatistics,tensorBoardModelPerformance = tensorBoardModelPerformance)

    if startEpoch<ganModelSettings['freezeEncoderEpochs']:
        print("Freeze encoder")
        gan.generator.encoder.trainable = False
        gan.compile(metrics = gan.compileMetrics)
        gan.fit(trainGenerator, batch_size= dataSettings['batchSize'], initial_epoch=startEpoch, epochs=ganModelSettings['freezeEncoderEpochs'], verbose=verbose,validation_data=valGenerator, callbacks=callbacks)
        startEpoch = ganModelSettings['freezeEncoderEpochs']
    
    if gan.generator.encoder.trainable == False:
        print("Unfreeze encoder.")
        gan.generator.encoder.trainable = True
        gan.compile(metrics = gan.compileMetrics)
        if ganModelSettings['freezeEncoderEpochs']>0:
            gan.generator.trainEncoderInInferenceMode = True
    
    sumWeights = tf.reduce_sum([tf.reduce_sum(variable) for variable in gan.generator.encoder.weights])
    print ('Sum of weights after compiling with new lr: {}'.format(sumWeights))

    gan.fit(trainGenerator, 
        batch_size= dataSettings['batchSize'], 
        initial_epoch=startEpoch, 
        epochs=nTrainEpochs, 
        verbose=verbose,
        validation_data=valGenerator, 
        callbacks=callbacks)

def RepeatTrain(experiment, nRepetitions):
    for run in range(nRepetitions):
        print("Starting run {}/{}".format(run+1,nRepetitions))
        train(experiment, run)
        print("Run {}/{} done".format(run+1,nRepetitions))
        print()