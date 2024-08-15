import os

import tensorflow as tf
import tensorflow_addons as tfa

from dino.dino_utils import DinoLoader
import utils as utils
from dino.dinoExperiments import GetExperiment

cacheDataset = True
bufferDataset = True

def train(experiment, saveEmbeddingEachEpoch, resultsLocation=None, logHistogramsEachEpoch=False, verbose=1,debug=False):
    """
    Training the DINO model. When the model has a vision transformer (ViT) as backbone, the attention maps are saved automatically. 
    """

    dinoModelSettings,dataSettings,ganModelSettings = GetExperiment(experiment,debug=debug)

    modelName = utils.models.getDinoModelName(dinoModelSettings,dataSettings,ganModelSettings)
    print("Experiment {}: {}".format(experiment, modelName))

    # Get model and dataset locations
    h5SetsLocation = utils.functions.getH5SetLocation(dataSettings['datasetName'])
    trainSamplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'trainSamples',subdir=dataSettings['datasetSubDir'])
    valSamplesFilename = utils.functions.getDatasetLocation(dataSettings['datasetName'], 'valSamples',subdir=dataSettings['datasetSubDir'])
    
    if dataSettings['subbands'] is None:
        dataSettings['nSubbands'] = len(os.listdir(h5SetsLocation))
    elif dataSettings['subbands'] == 'habrok':
        dataSettings['nSubbands'] = len(os.listdir(h5SetsLocation))
    else:
        dataSettings['nSubbands'] = len(dataSettings['subbands'])

    # Load datasets
    if dataSettings['subbands'] is None:
        nValSamples = dataSettings['nDinoValSamples']
    else:
        nValSamples = None

    train_dataset = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dino', 'train',dinoModelSettings['nChannels'] ,trainSamplesFilename, dinoModelSettings, dataSettings=dataSettings, bufferAll = bufferDataset, cacheDataset = cacheDataset, nSamples=dataSettings['nSsDinoSamples'])
    val_dataset = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dino', 'val', dinoModelSettings['nChannels'],valSamplesFilename, dinoModelSettings, dataSettings=dataSettings, bufferAll = bufferDataset, cacheDataset = cacheDataset, nSamples=nValSamples)
    
    attentionMapGenerator = None
    embeddingEpochGenerator = None
    supervisedValGenerator = None

    if 'vit' in dinoModelSettings['architecture']:
        attentionMapGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dino', 'original',dinoModelSettings['nChannels'] ,trainSamplesFilename, dinoModelSettings, dataSettings=dataSettings,bufferAll = bufferDataset, cacheDataset = cacheDataset, nSamples=32)
    
    if saveEmbeddingEachEpoch:
        embeddingEpochGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dino', 'test', dinoModelSettings['nChannels'],valSamplesFilename, dinoModelSettings, dataSettings=dataSettings, bufferAll = bufferDataset, cacheDataset = cacheDataset, nSamples=nValSamples) 
    
    # Load model
    dinoLoader = DinoLoader(dinoModelSettings, dataSettings, ganModelSettings,resultsDir=resultsLocation, keepAllEpochs = saveEmbeddingEachEpoch)
    dinoLoader.buildModel(len(train_dataset))
    startEpoch = dinoLoader.loadCheckpoint()
    callbacks = dinoLoader.loadCallbacks(attentionMapGenerator = attentionMapGenerator, embeddingEpochGenerator = embeddingEpochGenerator, supervisedValGenerator = supervisedValGenerator, logHistogramsEachEpoch=logHistogramsEachEpoch)
    model = dinoLoader.getModel()

    #Fit model
    print("Model loaded")
    if startEpoch < dinoModelSettings['freeze_last_layer']:
        print(" first number of epochs are with frozen last layer.")
        model.student_model.head.last_layer.trainable = False
        model.compile(optimizer=dinoLoader.optimizer)
        model.fit(train_dataset,validation_data=val_dataset,initial_epoch=startEpoch, epochs=dinoModelSettings['freeze_last_layer'],verbose=verbose,callbacks=callbacks)
        startEpoch = dinoModelSettings['freeze_last_layer']
    
    if model.student_model.head.last_layer.trainable == False:
        print("Unfreezing last layer")
        model.student_model.head.last_layer.trainable = True
        
        if dinoLoader.dinoSettings['optimizer'] == 'adam':
            # weight decay should be used by the adamW optimizer
            optimizer = tf.keras.optimizers.Adam(dinoLoader.lrScheduler)
        elif dinoLoader.dinoSettings['optimizer'] == 'adamw':
            optimizer = tfa.optimizers.AdamW(learning_rate=dinoLoader.lrScheduler, weight_decay=dinoLoader.wdScheduler)
        else:
            raise ValueError("Optimizer not recognized")
        
        print("Compiling model")
        model.compile(optimizer=optimizer)
        print("Updating checkpoint")
        dinoLoader.updateCheckpoint()
        print("Updating callbacks")
        callbacks = dinoLoader.loadCallbacks(attentionMapGenerator = attentionMapGenerator, embeddingEpochGenerator = embeddingEpochGenerator, supervisedValGenerator = supervisedValGenerator, logHistogramsEachEpoch=logHistogramsEachEpoch)

    print("Start training")    
    model.fit(train_dataset,validation_data=val_dataset,initial_epoch=startEpoch, epochs=dinoModelSettings['nEpochs'],verbose=verbose,callbacks=callbacks)