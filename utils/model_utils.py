import os
import pickle
import threading
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from keras import backend
from tqdm import tqdm
try:
    from keras.utils import tf_utils
except ImportError:
    pass

import utils as utils


matplotlib.use('agg')

defaultDataSettings = {
    'datasetName' : "LOFAR_L2014581 (recording)",
    'datasetSubDir' : 'dataset250k',
    "inputShape" : (64, 256, 8),
    "outputShape" :  (64, 256, 2),
    'batchSize': 8,
    'accumulateBatches':0, # Number of batches to accumulate before updating the weights. This can be used to simulate larger batch sizes and can help with convergence. 0 for no accumulation.
    'normalizationMethod': 10,
    'nTrainingSamples': None,
    'nValSamples': 1000,
    'nDinoValSamples':6000,
    'nSsDinoSamples' : None,
    'nSsGanSamples' : None,
    'subbands' : None,
    'nSubbands' : None,
    'freqStepSize' : 0.1953125,
    'startFreq' : 130.56640625,
    'subbandPrefix' : 'L2014581_SAP000_SB0',
    'subbandPostfix' : '_uv.h5',
    'augmentation': 'f', # Only used for DINO. f is horizontal and vertical flips (default)
}

# Gan settings
defaultGanModelSettings = {
    'modelBaseName': 'gan_v3',
    'skipsEnabled':True,
    'generatorDepth': 4,
    'pretextTask': 1,
    'freezeEncoderEpochs': 0,
    'scheduledEpochs': 350,
    'dropoutRate': 0.0,
    'batchNormMomentum': 0.9,
    'smooth': 0.1,
    'L1_lambda': 10,
    'lrStart': 1e-4,
    'lrEnd': 3.5e-5,
    'lrFinetune': 1e-6,
    'eps': 10e-5,
    'lamb': 0.004,
    'loadDinoEncoder':None,
    'loadDinoEpoch':None,
    'earlyStopping':None,
    'concatDino':False,
    'modConv':False,
    'modConvEncoder':False,
    'modulatedConvActivation':None,
    'modConvResUpNormAdd':None, 
    'styleMappingLrMul':None,
    'maskStyleInput':False,
    'metadataStyleInput':None,
    'nFeaturesW':512,
    'nMappingLayers':3,
    'styleMappingNormInput':True,
    }

# DINO default settings for GAN from 12-4-2024:
defaultDinoModelWithGanSettings = {
    'modelBaseName': 'dino',
    'architecture': 'gan',
    'patchSize': 4,
    'outputDim': 8192,  # Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.
    'teacherGlobalSize': (64, 256),
    'studentLocalSize': (32, 112),
    'globalScale': (0.5, 0.75),
    'localScale': (0.25, 0.328125),
    'nChannels': 8,
    'nEpochs': 100,
    'nLocalCrops': 6,
    'nGlobalCrops': 2,
    'hiddenSize':None,
    'sampleDim':None,
    'patchFlatteningSwap':False,
    'customPositionalEncoding':None,
    'metadataEncoding' : None,
    'normLastLayer': True, # Whether or not to weight normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
    'use_bn_in_head':True, # Whether to use batch normalizations in projection head
    'momentumTeacher': 0.996, # 0.996, Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.
    'weightDecay': 0.000001, # 0.04, Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.
    'weight_decay_end': 0.000001, # 0.4, inal value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.
    'freeze_last_layer':0, # Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.
    'learningRate':0.3, # 0.0005 Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.
    "warmup_epochs":10, # 10 "Number of epochs for the linear learning-rate warm up."
    "min_lr":0.0048, # "Target LR at the end of optimization. We use a cosine LR schedule with linear warmup."
    "optimizer":'adamw', # "Type of optimizer. We recommend using adamw with ViTs."
    'warmup_teacher_temp':0.04, #Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.
    'warmup_teacher_temp_epochs':50, # Number of warmup epochs for the teacher temperature (Default: 30).
    'teacher_temp':0.07, # Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.
    'reducedGanDimension':320,
    'clipGradient':0, # 'Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.    
}

# Dino settings
defaultDinoModelSettings = {
    'modelBaseName': 'dino_v6',
    'architecture': 'vit-s',
    'patchSize': 4,
    'outputDim': 4096,       # DINO: 65536 Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.
    'teacherGlobalSize': (64, 256),
    'studentLocalSize': (24, 96),
    'globalScale': (0.5, 0.75),
    'localScale': (0.125, 0.25),
    'nChannels': 4,
    'nEpochs': 100,
    'nLocalCrops': 8,
    'nGlobalCrops': 2,
    'hiddenSize':None,
    'sampleDim':None,
    'patchFlatteningSwap':False,
    'customPositionalEncoding':None,
    'metadataEncoding' : None,
    'normLastLayer': False, # Whether or not to weight normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
    'use_bn_in_head':False, # Whether to use batch normalizations in projection head
    'momentumTeacher': 0.9999, # 0.996, Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.
    'weightDecay': 0.00001, # 0.04, Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.
    'weight_decay_end': 0.00001, # 0.4, inal value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.
    'freeze_last_layer':0, # Number of epochs during which we keep the output layer fixed. Typically doing so during the first epoch helps training. Try increasing this value if the loss does not decrease.
    'learningRate':0.0005, # 0.0005 Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.
    "warmup_epochs":0, # 10 "Number of epochs for the linear learning-rate warm up."
    "min_lr":1e-6, # "Target LR at the end of optimization. We use a cosine LR schedule with linear warmup."
    "optimizer":'adamw', # "Type of optimizer. We recommend using adamw with ViTs."
    'warmup_teacher_temp':0.04, #Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.
    'warmup_teacher_temp_epochs':0, # Number of warmup epochs for the teacher temperature (Default: 30).
    'teacher_temp':0.04, # Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.
    'reducedGanDimension':None,
    'clipGradient':0,
}

keyAbbreveations={
    # Data
    "inputShape" : "inShape",
    "outputShape" : "outShape",
    'batchSize': 'bs',
    'normalizationMethod': "norm",
    'nTrainingSamples': "nSamples",
    'nSsDinoSamples' : 'nSsDino',
    'nSsGanSamples' : 'nSsGan',
    'nDinoValSamples': 'nDinoVal', 
    'subbands' : 'subb',
    'accumulateBatches':'accBatch',
    'augmentation': 'aug',

    # DINO
    'outputDim' : 'outDim',
    'dBatchSize' : 'dbs',
    'teacherGlobalSize' : 'globSize',
    'studentLocalSize' : 'locSize',
    'globalScale' : 'globSc',
    'localScale' : 'locSc',
    'nChannels' : 'ch',
    'nEpochs' : 'ep',
    'nLocalCrops' : 'nLoc',
    'nGlobalCrops' : 'nGlob',
    'hiddenSize' : 'hSize',
    'sampleDim' : 'sampleDim',
    'patchFlatteningSwap' : 'patchSwap',
    'customPositionalEncoding' : 'custEnc',
    'metadataEncoding' : 'metaEnc',
    'normLastLayer' : 'normLast',
    'use_bn_in_head':'bnHead',
    'momentumTeacher' : 'mom',
    'weightDecay' : 'wd',
    'weight_decay_end' : 'wdEnd',
    'freeze_last_layer' :'freezeLast',
    'learningRate' : 'lr',
    'warmup_epochs' : 'lrEp',
    'min_lr' : 'lrEnd',
    'optimizer' : 'opt',
    'warmup_teacher_temp' : 'Tstart',
    'warmup_teacher_temp_epochs' : 'Tep',
    'teacher_temp' :'T',
    'reducedGanDimension':'redGan',
    'clipGradient':'clipGr',
    

    # GAN
    'freezeEncoderEpochs': 'freezeEncEp',
    'loadDinoEncoder':'dinoExp',
    'loadDinoEpoch': 'dinoEp',
    'earlyStopping': 'nEsEp',
    'concatDino':'concat',
    'scheduledEpochs':'nEp',
    'modConv':'modConv',
    'modConvEncoder':'modConvEnc',
    'modulatedConvActivation':'modCact',
    'modConvResUpNormAdd':'modCResNormAdd', 
    'styleMappingLrMul':'styleLrMul',
    'dropoutRate':'drop',
    'metadataStyleInput':'metaStyle',
    'nMappingLayers':'nMapL',
    'styleMappingNormInput':'styleNorm',
}


def compressSettings(key, value):
    if key in keyAbbreveations.keys():
        compressedKey = keyAbbreveations[key]
    else:
        compressedKey = key
    
    compressedValue = value
    if isinstance(compressedValue, tuple):
        compressedValue = [v for v in compressedValue]

    if isinstance(compressedValue, list):
        textList = str(compressedValue[0])
        for element in compressedValue[1:]:
            textList += '_{}'.format(element)
        compressedValue = textList

    if isinstance(compressedValue, str):
        if compressedValue == 'leaky_relu':
            compressedValue = 'lRelu'

    return compressedKey, compressedValue

def dinoCustomSettingsDict(dinoModelSettings,dataSettings, ganSettings):
    prefix = ""
    if 'experiment' in dinoModelSettings.keys():
        prefix += '{}_'.format(dinoModelSettings['experiment'])
    
    prefix += '{}_{}'.format(dinoModelSettings['modelBaseName'],dinoModelSettings['architecture'])

    if 'vit' in dinoModelSettings['modelBaseName']:
        prefix += dinoModelSettings['patchSize']

    settingsDict = { }
    
    for key, value in dataSettings.items():
        if key == 'nSubbands' or key == 'nSsGanSamples' or key == 'nTrainingSamples':
            continue
        defaultValue = defaultDataSettings[key]
        if value != defaultValue:
            settingsDict[key] = value

    for key, value in dinoModelSettings.items():
        if key == 'modelBaseName' or key == 'architecture' or key=='experiment':
            continue
        if dinoModelSettings['architecture'] == 'gan':
            defaultValue = defaultDinoModelWithGanSettings[key]
        else:
            defaultValue = defaultDinoModelSettings[key]
        if value != defaultValue:
            settingsDict[key] = value

    if ganSettings is not None:
        for key, value in ganSettings.items():
            if key == 'modelBaseName' or key == 'freezeEncoderEpochs' or key == 'loadDinoEncoder' or key == 'experiment':
                continue
            defaultValue = defaultGanModelSettings[key]
            if value != defaultValue:
                settingsDict[key] = value
    return prefix, settingsDict

def ganCustomSettingsDict(ganModelSettings,dataSettings, dinoModelSettings):
    prefix = ""
    if 'experiment' in ganModelSettings.keys():
        prefix += '{}_'.format(ganModelSettings['experiment']) 
    prefix += ganModelSettings['modelBaseName']

    settingsDict = {}
    for key, value in dataSettings.items():
        if key == 'nSubbands' or key == 'nSsDinoSamples':
            continue
        defaultValue = defaultDataSettings[key]
        if value != defaultValue:
            settingsDict[key] = value

    for key, value in ganModelSettings.items():
        if key == 'modelBaseName' or key == 'experiment':
            continue
        defaultValue = defaultGanModelSettings[key]
        if value != defaultValue:
            settingsDict[key] = value

    return prefix, settingsDict

def getDinoModelName(dinoModelSettings,dataSettings, ganSettings = None):
    dinoModelName, dinoCustomSettings = dinoCustomSettingsDict(dinoModelSettings,dataSettings, ganSettings)

    for key, value in dinoCustomSettings.items():
        compressedKey, compressedValue = compressSettings(key, value)
        dinoModelName += '_{}={}'.format(compressedKey, compressedValue)

    # replace dot by _
    dinoModelName = dinoModelName.replace('.','_')
    dinoModelName = dinoModelName.replace('[','')
    dinoModelName = dinoModelName.replace(']','')
    dinoModelName = dinoModelName.replace('(','')
    dinoModelName = dinoModelName.replace(')','')

    return dinoModelName

def getGanModelName(ganModelSettings,dataSettings, dinoModelSettings=None):
    ganModelName, ganCustomSettings = ganCustomSettingsDict(ganModelSettings,dataSettings, dinoModelSettings)

    for key, value in ganCustomSettings.items():
        compressedKey, compressedValue = compressSettings(key, value)
        appendName = '_{}={}'.format(compressedKey, compressedValue)
        if compressedValue is not None:
            if isinstance(compressedValue, bool):
                if compressedValue: # This would remove the =True from the filename
                    appendName = '_{}'.format(compressedKey)
        ganModelName += appendName
    return ganModelName

class loss():
    def ssim_loss(y_true, y_pred):
        ssim = tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))
        return 1-ssim

    # def dice_coef(y_true, y_pred, smooth=1.0):
    #     # TODO: cite or delete https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
        
    #     y_true = tf.cast(y_true, y_pred.dtype)
    #     y_true_f = backend.flatten(y_true)
    #     y_pred_f = backend.flatten(y_pred)

    #     multiplied = y_true_f*y_pred_f
    #     intersection = backend.sum(multiplied)
    #     dice = (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
    #     return 1.0-dice

class metrics():
    def calcBatchMetrics(metrics, yTrue,yPred):
        metricResult = np.zeros(len(metrics))
        for metricIndex, metric in enumerate(metrics):
            currentMetric = metric(yTrue, yPred)
            metricResult[metricIndex] = currentMetric
        return metricResult
    
    def accuracy(y_true, y_pred):
        thresholdedYTrue = backend.round(backend.clip(y_true, 0, 1))
        thresholdedYPred = backend.round(backend.clip(y_pred, 0, 1))
        correct_predictions = tf.equal(thresholdedYTrue, thresholdedYPred)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy
    
    def recall(y_true, y_pred):
        # TODO: cite https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        # TODO: cite https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        return precision

    def f1(y_true, y_pred):
        # TODO: cite https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))  
        precision = true_positives / (predicted_positives + backend.epsilon())
        recall = true_positives / (possible_positives + backend.epsilon())
        return 2*((precision*recall)/(precision+recall+backend.epsilon()))

class Callbacks():
    class CheckpointSaver(tf.keras.callbacks.Callback):
        def __init__(self, checkpointMngr):
            super(Callbacks.CheckpointSaver, self).__init__()
            self.checkpointMngr = checkpointMngr

        def on_epoch_end(self, epoch, logs=None):
            self.checkpointMngr.save()

    class Plotter(tf.keras.callbacks.Callback):
        def __init__(self, dataGenerator, plotsLocation, plotPretext = False, plotPredictions=False, generativeModel = False):
            super(Callbacks.Plotter, self).__init__()
            self.dataGenerator = dataGenerator
            self.saveLocation = plotsLocation
            self.plotPretext = plotPretext
            self.plotPredictions = plotPredictions
            self.generativeModel = generativeModel

        def plotInThread(self, savename, data):
            if self.generativeModel:
                [observation, generatedObservation] = data

                if observation.shape[3] == 8:
                    realPart = observation[:, :, :,0:4]
                    imagPart = observation[:, :, :,4:8]
                    observationComplex = tf.complex(realPart, imagPart)
                    observationMagnitude = np.abs(observationComplex)
                else:
                    observationMagnitude = observation

                observationColor = np.zeros((observationMagnitude.shape[0],observationMagnitude.shape[1],observationMagnitude.shape[2],3))
                observationColor[:,:,:,0] = observationMagnitude[:,:,:,0]
                observationColor[:,:,:,1] = np.clip(0.5*observationMagnitude[:,:,:,1] + 0.5*observationMagnitude[:,:,:,2], 0.0, 1.0)
                observationColor[:,:,:,2] = observationMagnitude[:,:,:,3]

                if observation.shape[3] == 8:
                    realPart = generatedObservation[:, :, :, 0:4]
                    imagPart = generatedObservation[:, :, :, 4:8]
                    generatedComplex = tf.complex(realPart, imagPart)
                    generatedMagnitude = np.abs(generatedComplex)
                else:
                    generatedMagnitude = generatedObservation

                generatedColor = np.zeros((generatedMagnitude.shape[0],generatedMagnitude.shape[1],generatedMagnitude.shape[2],3))
                generatedColor[:,:,:,0] = generatedMagnitude[:,:,:,0]
                generatedColor[:,:,:,1] = np.clip(0.5*generatedMagnitude[:,:,:,1] + 0.5*generatedMagnitude[:,:,:,2], 0.0, 1.0)
                generatedColor[:,:,:,2] = generatedMagnitude[:,:,:,3]

                utils.plotter.plotGeneratedPredictionsLabels(generatedColor, observationColor,saveFileName=savename)  
            else:
                utils.plotter.plotLabels(data, saveFileName=savename)    

        def on_epoch_begin(self, epoch, logs=None):
            if self.plotPretext:
                observations = self.dataGenerator[0]
                pretexted = observations

                plotData = np.concatenate([observations, pretexted], axis=3)
                saveFileName = os.path.join(self.saveLocation,'pretext_at_epoch_{:04d}.png'.format(epoch))
                thread = threading.Thread(target=self.plotInThread, args=(saveFileName,plotData))
                thread.start()

        def on_epoch_end(self, epoch, logs=None):
            if self.plotPredictions:
                observations = self.dataGenerator[0]
                predictions = self.model.predict(observations)
                plotData = np.concatenate([observations, predictions], axis=0)
                saveFileName = os.path.join(self.saveLocation,'image_at_epoch_{:04d}.png'.format(epoch))
                thread = threading.Thread(target=self.plotInThread, args=(saveFileName,plotData))
                thread.start()  

    class Evaluator(tf.keras.callbacks.Callback):
        def __init__(self, model, dataGenerator,  modelDir):
            super(Callbacks.Evaluator, self).__init__()
            self.teacherModel = model
            self.modelDir = modelDir
            self.dataGenerator = dataGenerator
            self.evaluationFinishedEvent  = threading.Event()
            self.evaluationFinishedEvent.set()

        def evaluateInThread(self, epoch, predictions, modelDir, samplesHash):
            predictions = np.asarray(predictions)
            embeddingLocation = os.path.join(modelDir, 'embedding_val')
            os.makedirs(embeddingLocation, exist_ok=True)

            with open(os.path.join(embeddingLocation, 'embedding_epoch={}.pkl'.format(epoch+1)), 'wb') as file:
                pickle.dump([epoch+1, predictions,samplesHash], file)
            self.evaluationFinishedEvent.set()

        def evalKmeans(self, epoch):
            predictions = self.teacherModel.predictEmbedding(self.dataGenerator)
            samplesHash = self.dataGenerator.samplesHash

            # if thread is already finished, start a new one
            self.evaluationFinishedEvent.wait()
            thread = threading.Thread(target=self.evaluateInThread, args=(epoch, predictions,self.modelDir, samplesHash))
            thread.start()

        def on_epoch_end(self, epoch, logs=None):
            self.evalKmeans(epoch)

    class SupervisedEvaluator(tf.keras.callbacks.Callback):
        def __init__(self, model, trainData, valData,  logDir):
            super(Callbacks.SupervisedEvaluator, self).__init__()
            self.teacherModel = model
            self.logDir = logDir
            self.trainX = trainData[0]
            self.trainY = trainData[1]
            self.valX = valData[0]
            self.valY = valData[1]
            self.runInThread = False
            self.batchSize = 10 # TODO: get from model

            #self.dataGenerator = dataGenerator
            self.evaluationFinishedEvent  = threading.Event()
            self.evaluationFinishedEvent.set()

        def evaluateInThread(self, epoch, trainEmbedding, trainY, valEmbedding, valY, logDir):
            trainEmbedding = np.asarray(trainEmbedding)
            valEmbedding = np.asarray(valEmbedding)
            
            classificationModel = 'knn'

            if classificationModel == 'svm':
                # Train svm on trainEmbedding and trainY
                clf = svm.SVC()
                clf.fit(trainEmbedding, trainY)
                # Evaluate svm on valEmbedding and valY
                yPredicted = clf.predict(valEmbedding)
            elif classificationModel == 'linear':
                nClasses = np.unique(trainY).shape[0]

                yTrainOnehot = tf.keras.utils.to_categorical(trainY, num_classes=nClasses)
                yValOnehot = tf.keras.utils.to_categorical(valY, num_classes=nClasses)

                classifierModel = LinearClassifier(num_labels=nClasses)
                classifierModel.build((None, trainEmbedding.shape[1]))
                classifierModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                classifierModel.fit(trainEmbedding, yTrainOnehot, epochs=10, batch_size=100, validation_data=(valEmbedding, yValOnehot))
                predictedOnehot = classifierModel.predict(valEmbedding)
                yPredicted = np.argmax(predictedOnehot, axis=1)

            elif classificationModel == 'knn':
                # Train knn on trainEmbedding and trainY
                clf = KNeighborsClassifier(n_neighbors=5)
                clf.fit(trainEmbedding, trainY)
                # Evaluate knn on valEmbedding and valY
                yPredicted = clf.predict(valEmbedding)

            accuracy = np.mean(valY==yPredicted)

            lb = LabelBinarizer()
            lb.fit(trainY)
            classReport = classification_report(lb.transform(valY), lb.transform(yPredicted))

            logLocation = os.path.join(logDir, 'class evaluation')
            os.makedirs(logLocation, exist_ok=True)

            # save as text file
            if epoch == -1:
                logFileName = 'evaluation_epoch=0 (before training).txt'
                predictionsFilename = 'predictions_epoch=0 (before training).pkl'
            else:
                logFileName = 'evaluation_epoch={}.txt'.format(epoch+1)
                predictionsFilename = 'predictions_epoch={}.pkl'.format(epoch+1)

            with open(os.path.join(logLocation, logFileName), 'w') as file:
                file.write('Classification model: {}\n'.format(classificationModel))
                file.write('Accuracy: {}\n'.format(accuracy))
                file.write('classification report: \n{}'.format(classReport))

            with open(os.path.join(logLocation, predictionsFilename), 'wb') as file:
                pickle.dump([epoch+1, valY, yPredicted], file)
            self.evaluationFinishedEvent.set()

        def evaluate(self, epoch):
            trainEmbedding = []
            valEmbedding = []
            nTrainBatches = int(len(self.trainX)/self.batchSize)
            nValBatches = int(len(self.valX)/self.batchSize)
            print()
            for batchIdx in tqdm(range(nTrainBatches), desc='Train Embedding'):
                batchX = self.trainX[batchIdx*self.batchSize:(batchIdx+1)*self.batchSize]
                trainEmbedding.extend(self.teacherModel.predictEmbedding(batchX))

            for batchIdx in tqdm(range(nValBatches), desc='Val Embedding'):
                batchX = self.valX[batchIdx*self.batchSize:(batchIdx+1)*self.batchSize]
                valEmbedding.extend(self.teacherModel.predictEmbedding(batchX))

            if self.runInThread:
                # if thread is already finished, start a new one
                self.evaluationFinishedEvent.wait()
                thread = threading.Thread(target=self.evaluateInThread, args=(epoch, trainEmbedding, self.trainY, valEmbedding, self.valY,self.logDir))
                thread.start()
            else:
                print('Epoch {}: start evaluation without multithreading'.format(epoch))
                self.evaluateInThread(epoch, trainEmbedding, self.trainY, valEmbedding, self.valY,self.logDir)
                print('Epoch {}: eval done'.format(epoch))

        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                self.evaluate(epoch=-1) # with epoch -1, the evaluation is done before training

        def on_epoch_end(self, epoch, logs=None):
            self.evaluate(epoch)

    class AttentionMapPlotter(tf.keras.callbacks.Callback):
        def __init__(self, model, dataGenerator, plotLocation):
            super(Callbacks.AttentionMapPlotter, self).__init__()
            self.teacherModel = model
            self.dataGenerator = dataGenerator
            self.plotLocation = plotLocation

        def on_epoch_end(self, epoch, logs=None):
            flatAttentionMaps, dataX = self.teacherModel.predictAttentionMap(self.dataGenerator)
            images = self.dataGenerator.ConvertToImage(dataX)

            patchSize = self.teacherModel.patchSize
            saveLocation = os.path.join(self.plotLocation, 'epoch_{:03d}'.format(epoch+1))
            os.makedirs(saveLocation, exist_ok=True)
            thread = threading.Thread(target=self.plotInThread, args=(saveLocation,patchSize,flatAttentionMaps, images))
            thread.start()

        def plotInThread(self, epochPlotsLocation,patchSize, flatAttentionMaps,images):
            flatAttentionMaps = np.asarray(flatAttentionMaps)
            images = np.asarray(images)

            numberOfHeads = flatAttentionMaps.shape[1]
            # Calculate the attention maps
            h_featmap = images.shape[1] // patchSize
            w_featmap = images.shape[2] // patchSize

            for sampleIdx, (flatAttention, image) in enumerate(zip(flatAttentionMaps, images)):
                # Reshape the attention maps
                attentions = flatAttention.reshape(numberOfHeads, h_featmap, w_featmap, 1)
                resizedAttentions = tf.image.resize(attentions, (images.shape[1], images.shape[2]), method='bicubic')

                self.plotAttentionMap(image, resizedAttentions, epochPlotsLocation, sampleIdx)

        def plotAttentionMap(self, image, masks, epochPlotsLocation, sampleIdx, thresholdedMask=None, thesholdedHeads=None):
            nRows = 3+len(masks)
            if thresholdedMask is None:
                nCols = 1
            else:
                nCols = 2
            fig, axes = plt.subplots(nrows=nRows,ncols=nCols,figsize=(nCols*12,nRows*4))

            meanAttention = np.mean(masks,axis=0)
            meanAttention = (meanAttention - meanAttention.min()) / (meanAttention.max() - meanAttention.min())
            maskedImage = (meanAttention*image)

            if thresholdedMask is None:
                axes[0].set_title("Original")
                axes[0].imshow(image)

                axes[1].set_title("Attention Map")
                axes[1].imshow((maskedImage))
                
                axes[2].set_title("Head mean")
                axes[2].imshow(meanAttention,cmap='inferno')

                for maskIdx, mask in enumerate(masks):
                    axes[3+maskIdx].set_title("Head {}".format(maskIdx+1))
                    axes[3+maskIdx].imshow(mask,cmap='inferno')
            plt.savefig(os.path.join(epochPlotsLocation,'attention_map_{}.png'.format(sampleIdx)),dpi=300,bbox_inches='tight')
            plt.close()

    class EmbeddingSaverCallback(tf.keras.callbacks.Callback):
        def __init__(self, dataGenerator, tensorboardDir):
            super(Callbacks.EmbeddingSaverCallback, self).__init__()
            self.layerNames = ['Decoder_depth_bottleneck_dinoMappingNetwork']
            self.log_dir = tensorboardDir
            self.dataGenerator = dataGenerator

        def find_layer(self, model, layer_name):
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    sub_layer = self.find_layer(layer, layer_name)
                    if sub_layer is not None:
                        return sub_layer
                elif layer.name == layer_name:
                    return layer
            return None

        def on_epoch_begin(self, epoch, logs=None):
            # Get the model
            generatorIndices = self.dataGenerator.indices
            mappingLayer = self.model.generator.decoder.layers[0].get_layer('Decoder_depth_bottleneck_dinoMappingNetwork')
            embeddings = []
            images = []
            for batch in self.dataGenerator:
                dataX = batch[0]
                dinoEmbedding,_ = self.model.generator.dinoEncoder(dataX)
                dinoFeatures = self.model.generator.featuresReductionLayer(dinoEmbedding)
                latentZ = self.model.generator.flattenLayer(dinoFeatures)
                mappingEmbedding = mappingLayer(latentZ)[:,0,:]
                embeddings.extend(mappingEmbedding)
                images.extend(self.dataGenerator.ConvertToImage(dataX))

            # Save the embeddings to a file
            np.save(os.path.join(self.log_dir, 'embeddings_epoch_layer_styleMapping.npy'),embeddings)
            metadata = self.dataGenerator.getMetadata(generatorIndices)  # Function to generate metadata
            frequencies = metadata[2]
            np.save(os.path.join(self.log_dir, 'metadata.npy'), frequencies)
            np.save(os.path.join(self.log_dir, 'images.npy'), images)

    class TerminateOnNaN(tf.keras.callbacks.Callback):
        # TODO: cite source
        """Callback that terminates training when a NaN loss is encountered."""
        def __init__(self):
            super().__init__()
            self._supports_tf_logs = True

        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            loss = logs.get("loss")
            if loss is not None:
                loss = tf_utils.sync_to_numpy_or_python_type(loss)
                if np.isnan(loss) or np.isinf(loss):
                    print(f"Batch {batch}: Invalid loss, terminating training")
                    self.model.stop_training = True

class layers():
    class DiscLayer5(tf.keras.layers.Layer):
        def __init__(self, size1, size2, lamb, name=None, **kwargs):
            super(layers.DiscLayer5, self).__init__(name=name)
            self.size1 = size1
            self.size2 = size2
            self.lamb = lamb
            super(layers.DiscLayer5, self).__init__(**kwargs)

        def build(self, input_shape):
            # w5 = tf.get_variable(name='d_w5', regularizer=l2_regularizer(scale=self.lamb), initializer=tf.truncated_normal(shape=[size1 * size2, 1], dtype=tf.float32))
            # b5 = tf.Variable(initial_value=tf.random_normal(shape=[1], dtype=tf.float32), name=name)
            # l5 = tf.matmul(l5_flat,w5) + b5

            # Create the trainable weights for the layer
            self.w5 = self.add_weight(name='w5', shape=(self.size1 * self.size2, 8), initializer=tf.initializers.TruncatedNormal(), regularizer=tf.keras.regularizers.l2(self.lamb), trainable=True)
            self.b5 = self.add_weight(name='b5', shape=(8,), initializer=tf.initializers.RandomNormal(), trainable=True)
            super(layers.DiscLayer5, self).build(input_shape)

        def call(self, inputs):
            l5 = tf.matmul(inputs, self.w5) + self.b5
            return l5

        def get_config(self):
            config = super(layers.DiscLayer5, self).get_config()
            config.update({'size1': self.size1, 'size2': self.size2, 'lamb': self.lamb})
            return config

    class UpSampleLayer(tf.keras.layers.Layer):
        def __init__(self, batchNormMomentum, epsilon, featureMuliplier = 1, **kwargs):
            super(layers.UpSampleLayer, self).__init__(**kwargs)
            self.batchNormMomentum = batchNormMomentum
            self.epsilon = epsilon
            self.featureMuliplier = featureMuliplier

        def build(self, input_shape):
            _, _, _, channels_num = input_shape
            channels_num *= self.featureMuliplier
            self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters=channels_num // 2, kernel_size=(2, 2), strides=2, padding='VALID')
            self.batch_norm = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum, epsilon=self.epsilon)
            self.relu = tf.keras.layers.ReLU()

        def call(self, input_data):
            result_up = self.conv_transpose(input_data)
            normed_batch = self.batch_norm(result_up)
            result_relu = self.relu(normed_batch)
            return result_relu
        
        def get_config(self):
            config = {
                'batch_norm_momentum': self.batchNormMomentum,
                'epsilon': self.epsilon,
                'featureMultiplier' : self.featureMuliplier
            }
            base_config = super(layers.UpSampleLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    class ResUnitDown(tf.keras.layers.Layer):
        def __init__(self, batchNormMomentum, epsilon, **kwargs):
            super(layers.ResUnitDown, self).__init__(**kwargs)
            self.batchNormMomentum = batchNormMomentum
            self.eps = epsilon

        def build(self, input_shape):
            _, _, _, channels_num = input_shape

            # conv_0
            self.conv2d0 = tf.keras.layers.Conv2D(filters=2 * channels_num, kernel_size=(1,1), strides=1, padding='SAME')
            self.batchNorm0 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            
            # conv_1
            self.conv2d1 = tf.keras.layers.Conv2D(filters=2 * channels_num, kernel_size=(3,3), strides=1, padding='SAME')
            self.batchNorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            self.relu1 = tf.keras.layers.ReLU()

            # conv_2
            self.conv2d2 = tf.keras.layers.Conv2D(filters=2 * channels_num, kernel_size=(3,3), strides=1, padding='SAME')
            self.batchNorm2 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            self.relu2 = tf.keras.layers.ReLU()

            # conv_3
            self.conv2d3 = tf.keras.layers.Conv2D(filters=2 * channels_num, kernel_size=(3,3), strides=1, padding='SAME')
            self.batchNorm3 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            
            # add short skip
            self.batchNormAdd = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            self.reluAdd = tf.keras.layers.ReLU()      

        def call(self, input_data, training=False):
            result_conv_0 = self.conv2d0(input_data)
            split_from_input = self.batchNorm0(result_conv_0, training=training)
            
            result_conv_2 = self.conv2d1(input_data)
            normed_batch = self.batchNorm1(result_conv_2, training=training)
            result_relu_2 = self.relu1(normed_batch)

            result_conv_2 = self.conv2d2(result_relu_2)
            normed_batch = self.batchNorm2(result_conv_2, training=training)
            result_relu_2 = self.relu2(normed_batch)

            result_conv_1 = self.conv2d3(result_relu_2)
            normed_batch = self.batchNorm3(result_conv_1, training=training)
            
            result_add = tf.add(x=normed_batch, y=split_from_input)
            result_add = self.batchNormAdd(result_add, training=training)
            result_relu_add = self.reluAdd(result_add)

            return result_relu_add
        
        def get_config(self):
            config = {
                'batch_norm_momentum': self.batch_norm_momentum,
                'epsilon': self.epsilon,
            }
            base_config = super(layers.ResUnitUp, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
        
    class ResUnitUp(tf.keras.layers.Layer):
        def __init__(self, batchNormMomentum, epsilon, **kwargs):
            super(layers.ResUnitUp, self).__init__(**kwargs)
            self.batchNormMomentum = batchNormMomentum
            self.eps = epsilon

        def build(self, input_shape):
            _, _, _, channels_num = input_shape

            # split from the input to short connect
            # input
            self.conv2dInput = tf.keras.layers.Conv2D(filters=channels_num // 2, kernel_size=(1,1), strides=1, padding='SAME')
            self.batchNormInput = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)          

            # conv_1
            self.conv2d1 = tf.keras.layers.Conv2D(filters=channels_num // 2, kernel_size=(3,3), strides=1, padding='SAME')
            self.batchNorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            self.relu1 = tf.keras.layers.ReLU()
            
            # conv_2
            self.conv2d2 = tf.keras.layers.Conv2D(filters=channels_num // 2, kernel_size=(3,3), strides=1, padding='SAME')
            self.batchNorm2 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            self.relu2 =  tf.keras.layers.ReLU()
            
            # conv_3
            self.conv2d3 = tf.keras.layers.Conv2D(filters=channels_num // 2, kernel_size=(3,3), strides=1, padding='SAME')
            self.batchNorm3 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            
            # add short skip
            self.batchNormAdd = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
            self.reluAdd = tf.keras.layers.ReLU()

            self.layerList = [self.conv2dInput,self.batchNormInput,
                              self.conv2d1,self.batchNorm1,self.relu1,
                              self.conv2d2,self.batchNorm2,self.relu2,
                              self.conv2d3,self.batchNorm3,
                              self.batchNormAdd,self.reluAdd]

        def call(self, input_data):
            # input
            xSplit = self.conv2dInput(input_data)
            xSplit = self.batchNormInput(xSplit)

            # conv 1
            x1 = self.conv2d1(input_data)
            x1 = self.batchNorm1(x1)
            x1 = self.relu1(x1)

            # conv 2
            x2 = self.conv2d2(x1)
            x2 = self.batchNorm2(x2)
            x2 = self.relu2(x2)
            
            # conv 3
            x3 = self.conv2d3(x2)
            x3 = self.batchNorm3(x3)
            x3 = self.relu2(x3)

            # Add results
            xAdd = tf.add(x=x3, y=xSplit)
            xAdd = self.batchNormAdd(xAdd)
            xAdd = self.reluAdd(xAdd)

            return xAdd
        
        def get_config(self):
            config = {
                'batch_norm_momentum': self.batch_norm_momentum,
                'epsilon': self.epsilon,
            }
            base_config = super(layers.ResUnitUp, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    class ModConvResUnitUp(tf.keras.layers.Layer):
        def __init__(self, batchSize, activationFunction,normalizeAdd = None, **kwargs):
            super(layers.ModConvResUnitUp, self).__init__(**kwargs)
            self.batchSize = batchSize
            self.activationFunction = activationFunction
            self.normalizeAdd = normalizeAdd

        def build(self, input_shape):
            _, _, _, channels_num = input_shape

            # split from the input to short connect
            # xSplit =  AdaIN ? Or else modConv? Since the result is added, they must be similar somehow. ModConv with filter size 1?
            self.modConvSplit = layers.ModulatedConv2d(channels_num // 2,self.batchSize, kernel=1,activationFunction=self.activationFunction)

            # conv_1
            self.modConv1 = layers.ModulatedConv2d(channels_num // 2,self.batchSize, kernel=3,activationFunction=self.activationFunction)

            # conv_2
            self.modConv2 = layers.ModulatedConv2d(channels_num // 2,self.batchSize, kernel=3,activationFunction=self.activationFunction)
            
            # conv_3
            self.modConv3 = layers.ModulatedConv2d(channels_num // 2,self.batchSize, kernel=3,activationFunction=self.activationFunction)
            
            # add short skip normalization
            if self.normalizeAdd is not None:
                if self.normalizeAdd == 'IN':
                    self.layerNormAdd = layers.InstanceNormalization()
                elif self.normalizeAdd == 'IN_bias':
                    self.layerNormAdd = layers.InstanceNormalization(bias=True)
                elif self.normalizeAdd == 'scale':
                    def scaleFunction(x):
                        return x * (1 / np.sqrt(2))
                    self.layerNormAdd = scaleFunction

            if self.activationFunction == 'relu':
                self.activationAdd = tf.keras.layers.ReLU()
            elif self.activationFunction == 'leaky_relu':
                self.activationAdd = tf.keras.layers.LeakyReLU()

        def call(self, x, y):
            xSplit = self.modConvSplit(x,y[:,0,:])
            
            x = self.modConv1(x,y[:,1,:])
            x = self.modConv2(x,y[:,2,:])
            x = self.modConv3(x,y[:,3,:])
            
            # ReLU
            xAdd = tf.add(x=x, y=xSplit)
            if self.normalizeAdd is not None:
                xAdd = self.layerNormAdd(xAdd)
            if self.activationFunction is not None:
                xAdd = self.activationAdd(xAdd)
            return xAdd

    class ModConvResUnitDown(tf.keras.layers.Layer):
        def __init__(self, batchSize, activationFunction,normalizeAdd = None, **kwargs):
            super(layers.ModConvResUnitDown, self).__init__(**kwargs)
            self.batchSize = batchSize
            self.activationFunction = activationFunction            
            self.normalizeAdd = normalizeAdd
            self.activationAdd = None

        def build(self, input_shape):
            _, _, _, channels_num = input_shape

            # split from the input to short connect
            self.modConvSplit = layers.ModulatedConv2d(channels_num * 2,self.batchSize, kernel=1,activationFunction=self.activationFunction)

            self.modConv1 = layers.ModulatedConv2d(channels_num * 2,self.batchSize, kernel=3,activationFunction=self.activationFunction)
            self.modConv2 = layers.ModulatedConv2d(channels_num * 2,self.batchSize, kernel=3,activationFunction=self.activationFunction)
            self.modConv3 = layers.ModulatedConv2d(channels_num * 2,self.batchSize, kernel=3,activationFunction=self.activationFunction)
            
            addActivationFunction = self.activationFunction
            # add short skip normalization
            if self.normalizeAdd is not None:
                if self.normalizeAdd == 'IN':
                    self.layerNormAdd = layers.InstanceNormalization()
                elif self.normalizeAdd == 'IN_bias':
                    self.layerNormAdd = layers.InstanceNormalization(bias=True)
                elif self.normalizeAdd == 'modConv':
                    self.layerNormAdd = layers.ModulatedConv2d(channels_num * 2,self.batchSize, kernel=1,activationFunction=self.activationFunction)
                    addActivationFunction = None # modConv has its own activation function
                elif self.normalizeAdd == 'scale':
                    def scaleFunction(x):
                        return x * (1 / np.sqrt(2))
                    self.layerNormAdd = scaleFunction
            
            if addActivationFunction is not None:
                if addActivationFunction == 'relu':
                    self.activationAdd = tf.keras.layers.ReLU()
                elif addActivationFunction == 'leaky_relu':
                    self.activationAdd = tf.keras.layers.LeakyReLU()

        def call(self, x, y):
            xSplit = self.modConvSplit(x,y[:,0,:])
 
            x = self.modConv1(x,y[:,1,:])
            x = self.modConv2(x,y[:,2,:])
            x = self.modConv3(x,y[:,3,:])

            # ReLU
            xAdd = tf.add(x=x, y=xSplit)
            if self.normalizeAdd is not None:
                if self.normalizeAdd == 'modConv':
                    xAdd = self.layerNormAdd(xAdd,y[:,4,:])
                elif self.normalizeAdd == 'scale':
                    xAdd *= (1 / np.sqrt(2))
                else:
                    xAdd = self.layerNormAdd(xAdd)

            if self.activationAdd is not None:
                xAdd = self.activationAdd(xAdd)
            return xAdd

    class InstanceNormalization(tf.keras.layers.Layer):
        # TODO: cite that it is from stylegan?
        def __init__(self, bias=False, epsilon=1e-8, **kwargs):
            super(layers.InstanceNormalization, self).__init__(**kwargs)
            self.epsilon = epsilon
            self.bias = bias

        def build(self, input_shape):
            self.biasWeights = self.add_weight(name='bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
           
        def call(self, x):
            # x format: NHWC
            x -= tf.reduce_mean(x, axis=[1,2], keepdims=True)
            x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True) + self.epsilon)
            if self.bias:
                x += self.biasWeights
            return x

    class ModulatedConv2d(tf.keras.layers.Layer):
        def __init__(self, nFeaturesW, batchSize, up=False, down=False, kernel=3, demodulate=True, lrmul=1, activationFunction = None, **kwargs):
            super(layers.ModulatedConv2d, self).__init__(**kwargs)
            self.demodulate = demodulate
            self.nFeaturesW = nFeaturesW
            self.kernel=kernel
            self.lrmul = lrmul
            self.batchSize = batchSize
            self.activationFunction = activationFunction
            self.up=up
            self.down=down

            # Only used for upsample, when up=True
            self.upsampleFactor = 2
            self.downsampleFactor = 2
            self.resample_kernel=None

        def build(self, shapeX):
            # x.shape[1] are the number of filters, which start at 512 and can go to 1
            self.nFeaturesX = shapeX[-1]
            self.latentToStyleLayer = tf.keras.layers.Dense(self.nFeaturesX, kernel_initializer='he_normal')

            convFilterWeightShape = (self.kernel, self.kernel, self.nFeaturesX, self.nFeaturesW)
            gain=1
            fan_in = np.prod(convFilterWeightShape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
            he_std = gain / np.sqrt(fan_in) # He init
            initStd = 1.0 / self.lrmul
            self.weightRuntimeCoef = he_std*self.lrmul

            initValue = tf.random.normal(convFilterWeightShape,mean=0,stddev=initStd)
            self.weightConvFilter = tf.Variable(initValue, trainable=True)
            self.bias = tf.Variable(tf.zeros([self.nFeaturesW], dtype=tf.float32))

            if self.up:
                self.convH = self.kernel
                self.convW = self.kernel
                self.inC = self.nFeaturesX 
                self.outC = self.nFeaturesW*self.batchSize #1280
                self.stride = [1, 1, self.upsampleFactor, self.upsampleFactor] # Stride in standard upsample layer is 2
                self.convTranspose_outputShape = [1, self.outC, (shapeX[1] - 1) * self.upsampleFactor + self.convH, (shapeX[2] - 1) * self.upsampleFactor + self.convW]
                self.convTranspose_numGroups = self.batchSize

        def call(self, x,y):
            x = tf.transpose(x, [0, 3, 1, 2]) # NHWC -> NCHW

            # Get weight.
            w = self.weightConvFilter*self.weightRuntimeCoef
            ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

            # Modulate.
            s=self.latentToStyleLayer(y)
            ww *= s[:, np.newaxis, np.newaxis, :, np.newaxis]

            # Demodulate.
            if self.demodulate:
                d = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) 
                ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :]

            # Reshape/scale input.
            # fused_modconv:
            x = tf.reshape(x, [1, x.shape[1]*self.batchSize, x.shape[2], x.shape[3]], name="minibatchToConvGroup") # Fused => reshape minibatch to convolution groups.
            wTransposed = tf.transpose(ww, [1, 2, 3, 0, 4])
            w = tf.reshape(wTransposed, [ww.shape[1], ww.shape[2], ww.shape[3], self.batchSize*ww.shape[4]], name="convGroupToWeightFilter") # Fused => reshape convolution groups to minibatch.

            # Convolution with optional up/downsampling.
            if self.up:
                # Transpose weights.
                w = tf.reshape(w, [self.convH, self.convW, self.inC, self.convTranspose_numGroups, -1])
                w = tf.transpose(w[::-1, ::-1], [0, 1, 4, 3, 2])
                w = tf.reshape(w, [self.convH, self.convW, -1, self.convTranspose_numGroups * self.inC]) 
                x = tf.nn.conv2d_transpose(x, w, output_shape=self.convTranspose_outputShape, strides=self.stride, padding='VALID', data_format='NCHW')

            elif self.down:
                s = [1, 1, self.downsampleFactor, self.downsampleFactor]
                x = tf.nn.conv2d(x, w, strides=s, padding='SAME', data_format='NCHW')
            else:
                x = tf.nn.conv2d(x, w, data_format='NCHW', strides=[1,1,1,1], padding='SAME')

            # Reshape/scale output.
            x = tf.reshape(x, [self.batchSize, self.nFeaturesW, x.shape[2], x.shape[3]], name="ConvGroupToMinibatch") # Fused => reshape convolution groups back to minibatch.
            x = tf.transpose(x, [0, 2, 3, 1]) # NCHW -> NHWC
            x = tf.nn.bias_add(x, self.bias*self.lrmul)

            if self.activationFunction is not None:
                if self.activationFunction == 'relu':
                    x = tf.nn.relu(x)
                elif self.activationFunction == 'leaky_relu':
                    x = tf.nn.leaky_relu(x)
            return x
    
        def get_config(self):
            config = {
                'wieghtConvFilter': self.weightConvFilter,
            }
            base_config = super(layers.ModulatedConv2d, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
    
    class LrMulDenseLayer(tf.keras.layers.Layer):
        def __init__(self, units, mapping_lrmul, activation, **kwargs):
            super(layers.LrMulDenseLayer, self).__init__(**kwargs)
            self.units = units
            self.lrmul = mapping_lrmul
            self.activation = activation

        def build(self, input_shape):
            gain=1

            weightShape = (input_shape[1], self.units)

            fan_in = np.prod(weightShape[:-1]) 
            he_std = gain / np.sqrt(fan_in) # He init 
            initStd = 1.0 / self.lrmul
            self.weightRuntimeCoef = he_std*self.lrmul

            weightInit = tf.initializers.random_normal(0, initStd)
            self.weight = self.add_weight(name='weight', shape=weightShape, dtype=tf.float32, initializer=weightInit, trainable=True)
            self.bias = self.add_weight(name='bias', shape=(self.units,), dtype=tf.float32, initializer='zeros', trainable=True)

            if self.activation is not None:
                if self.activation == 'relu':
                    self.activationLayer = tf.keras.layers.ReLU()
                elif self.activation == 'leaky_relu':
                    self.activationLayer = tf.keras.layers.LeakyReLU()
                else:
                    raise ValueError('Activation function not recognized')
  
        def call(self, x):
            scaledWeight = self.weight*self.lrmul
            scaledBias = self.bias*self.lrmul
            x = tf.matmul(x, scaledWeight)
            x = tf.nn.bias_add(x, scaledBias)
            if self.activation is not None:
                x = self.activationLayer(x)
            return x

    class DinoMappingNetwork(tf.keras.layers.Layer):
        def __init__(self,
                     nFeaturesW,                                #512    # Disentangled latent (W) dimensionality.
                     dlatent_broadcast       = None,                    # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
                     nLayers                 = 3,               # 8     # Number of mapping layers.
                     nUnitsHiddenLayers      = 320,             #512,   # Number of activations in the mapping layers.
                     mapping_lrmul           = 0.01,            #0.01   # Learning rate multiplier for the mapping layers.
                     activation              = 'leaky_relu',    #lrelu  # Activation function: 'relu', 'lrelu', etc.
                     normalizeInput          = True,            #True   # Normalize latent vectors (Z) before feeding them to the mapping layers?
                     **kwargs):
            super(layers.DinoMappingNetwork, self).__init__(**kwargs)
            self.nFeaturesW = nFeaturesW
            self.dlatent_broadcast = dlatent_broadcast
            self.nLayers = nLayers
            self.nUnitsHiddenLayers = nUnitsHiddenLayers
            self.mapping_lrmul = mapping_lrmul
            self.activation = activation
            self.normalizeInput = normalizeInput
            
        def build(self, shapeX):
            self.denseLayers = []

            if self.nLayers>0:
                if self.mapping_lrmul is not None:
                    for layer_idx in range(self.nLayers-1):
                        self.denseLayers.append(layers.LrMulDenseLayer(self.nUnitsHiddenLayers, self.mapping_lrmul, activation=self.activation))
                    self.denseLayers.append(layers.LrMulDenseLayer(self.nFeaturesW, self.mapping_lrmul, activation=self.activation))
                else:
                    for layer_idx in range(self.nLayers-1):
                        self.denseLayers.append(tf.keras.layers.Dense(units=self.nUnitsHiddenLayers, kernel_initializer='he_normal', activation=self.activation))
                    self.denseLayers.append(tf.keras.layers.Dense(units=self.nFeaturesW, kernel_initializer='he_normal', activation=self.activation))


        def call(self, x):
            # Normalize latents.
            if self.normalizeInput:
                x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)

            # Mapping layers.
            for layer_idx in range(self.nLayers):
                x = self.denseLayers[layer_idx](x)

            # Broadcast.
            if self.dlatent_broadcast is not None:
                x = tf.tile(x[:, np.newaxis], [1, self.dlatent_broadcast, 1])

            # Output.
            return tf.identity(x, name='dlatents_out')

# class Architectures():
#     class EncoderRfiGan(tf.keras.layers.Layer):
#         def __init__(self, batchNormMomentum=0.9, eps=10e-5, dropoutRate=0.0, **kwargs):
#             super(Architectures.EncoderRfiGan, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate

#         def build(self, input_shape):
#             # Input
#             self.batchnorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)

#             # Layer 1
#             self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='SAME')
#             self.batchNorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.relu1 = tf.keras.layers.ReLU()

#             self.resDown1 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxPool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout1 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 2
#             self.resDown2 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout2 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 3
#             self.resDown3 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout3 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 4
#             self.resDown4 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout4 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             self.resDown5 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)

#             #up sample
#             self.upsample5 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout5 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#         def call(self, inputs):
#             result_from_contract_layer = {}
#             # Input
#             x1 = self.batchnorm1(inputs)

#             # Layer 1
#             x1 = self.conv1(x1)
#             x1 = self.batchNorm1(x1)
#             x1 = self.relu1(x1)

#             x1 = self.resDown1(x1)
#             result_from_contract_layer[1]=x1
#             x1 = self.maxPool1(x1)
#             x1 = self.dropout1(x1)

#             # layer 2
#             x2 = self.resDown2(x1)
#             result_from_contract_layer[2]=x2
#             x2 = self.maxpool2(x2)
#             x2 = self.dropout2(x2)

#             # layer 3
#             x3 = self.resDown3(x2)
#             result_from_contract_layer[3]=x3
#             x3 = self.maxpool3(x3)
#             x3 = self.dropout3(x3)
            
#             # layer 4
#             x4 = self.resDown4(x3)
#             result_from_contract_layer[4]=x4
#             x4 = self.maxpool4(x4)
#             x4 = self.dropout4(x4)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             x5 = self.resDown5(x4)
#             x5 = self.upsample5(x5) #up sample
#             x5 = self.dropout5(x5)

#             return [x5, result_from_contract_layer]
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate
#             }
#             base_config = super(Architectures.EncoderRfiGan, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))
       
#     class DecoderRfiGan(tf.keras.layers.Layer):
#         def __init__(self, outputChannels, batchNormMomentum, eps, dropoutRate, **kwargs):
#             super(Architectures.DecoderRfiGan, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate
#             self.outputChannels = outputChannels

#         def build(self, input_shape):
#             # layer 6
#             self.concat6 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm6 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp6 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample6 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout6 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 7
#             self.concat7 =  tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm7 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp7 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample7 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout7 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 8
#             self.concat8 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm8 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp8 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample8 =utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout8 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 9
#             self.concat9 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm9 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resup9 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)

#             # To output
#             self.conv2dOut = tf.keras.layers.Conv2D(filters=self.outputChannels, kernel_size=(1,1), strides=1, padding='VALID', name='conv_3')
#             self.batchnormOut = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.sigmoidOut = tf.keras.layers.Activation('sigmoid')

#         def call(self, inputs):
#             [latentInputs, result_from_contract_layer] = inputs
            
#             # layer 6
#             x6 = self.concat6([result_from_contract_layer[4], latentInputs])
#             x6 = self.batchnorm6(x6)
#             x6 = self.resUp6(x6)
#             x6 = self.upsample6(x6)
#             x6 = self.dropout6(x6)

#             # layer 7
#             x7 = self.concat7([result_from_contract_layer[3], x6])
#             x7 = self.batchnorm7(x7)
#             x7 = self.resUp7(x7)
#             x7 = self.upsample7(x7)
#             x7 = self.dropout7(x7)
            
#             # layer 8
#             x8 = self.concat8([result_from_contract_layer[2], x7])
#             x8 = self.batchnorm8(x8)
#             x8 = self.resUp8(x8)
#             x8 = self.upsample8(x8)
#             x8 = self.dropout8(x8)

#             # layer 9
#             x9 = self.concat9([result_from_contract_layer[1], x8])
#             x9 = self.batchnorm9(x9)
#             x9 = self.resup9(x9)

#             # To output
#             xOut = self.conv2dOut(x9)
#             xOut = self.batchnormOut(xOut)
#             xOut = self.sigmoidOut(xOut)
#             return xOut
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate
#             }
#             base_config = super(Architectures.DecoderRfiGan, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))

#     class EncoderRfiGanDense_v1(tf.keras.layers.Layer):
#         def __init__(self, batchNormMomentum=0.9, eps=10e-5, dropoutRate=0.0, **kwargs):
#             super(Architectures.EncoderRfiGanDense_v1, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate

#         def build(self, input_shape):
#             # Input
#             self.batchnorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)

#             # Layer 1
#             self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='SAME')
#             self.batchNorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.relu1 = tf.keras.layers.ReLU()

#             self.resDown1 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxPool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout1 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 2
#             self.resDown2 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout2 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 3
#             self.resDown3 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout3 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 4
#             self.resDown4 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout4 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             self.resDown5 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)

#             #self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1)
            
#             # self.flatten5 = tf.keras.layers.Flatten()
#             # self.dense5 = tf.keras.layers.Dense(524288)
#             # self.reshape = tf.keras.layers.Reshape((8,4,16,1024))


                        
#             self.layerList = [self.batchnorm1,
#                     self.conv1,self.batchNorm1,self.relu1,
#                     self.resDown1,self.maxPool1,self.dropout1,
#                     self.resDown2,self.maxpool2,self.dropout2,
#                     self.resDown3,self.maxpool3,self.dropout3,
#                     self.resDown4,self.maxpool4,self.dropout4,
#                     self.resDown5]#,self.conv5]

#         def call(self, inputs):
#             result_from_contract_layer = {}
#             # Input
#             x1 = self.batchnorm1(inputs)

#             # Layer 1
#             x1 = self.conv1(x1)
#             x1 = self.batchNorm1(x1)
#             x1 = self.relu1(x1)

#             x1 = self.resDown1(x1)
#             result_from_contract_layer[1]=x1
#             x1 = self.maxPool1(x1)
#             x1 = self.dropout1(x1)

#             # layer 2
#             x2 = self.resDown2(x1)
#             result_from_contract_layer[2]=x2
#             x2 = self.maxpool2(x2)
#             x2 = self.dropout2(x2)

#             # layer 3
#             x3 = self.resDown3(x2)
#             result_from_contract_layer[3]=x3
#             x3 = self.maxpool3(x3)
#             x3 = self.dropout3(x3)
            
#             # layer 4
#             x4 = self.resDown4(x3)
#             result_from_contract_layer[4]=x4
#             x4 = self.maxpool4(x4)
#             x4 = self.dropout4(x4)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             x5 = self.resDown5(x4)
#             #x5 = self.conv5(x5)
#             # x5 = self.flatten5(x5)
#             # x5 = self.dense5(x5)
#             # x5 = self.reshape(x5)


#             return [x5, result_from_contract_layer]
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate
#             }
#             base_config = super(Architectures.EncoderRfiGanDense_v1, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))
       
#     class DecoderRfiGanDense_v1(tf.keras.layers.Layer):
#         def __init__(self, outputChannels, batchNormMomentum, eps, dropoutRate, skipConnections = True, **kwargs):
#             super(Architectures.DecoderRfiGanDense_v1, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate
#             self.outputChannels = outputChannels
#             self.skipConnections = skipConnections

#         def build(self, input_shape):
#             # layer 5
#             self.upsample5 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout5 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 6
#             self.concat6 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm6 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp6 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample6 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout6 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 7
#             self.concat7 =  tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm7 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp7 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample7 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout7 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 8
#             self.concat8 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm8 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp8 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample8 =utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout8 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 9
#             self.concat9 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm9 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resup9 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)

#             # To output
#             self.conv2dOut = tf.keras.layers.Conv2D(filters=self.outputChannels, kernel_size=(1,1), strides=1, padding='VALID', name='conv_3')
#             self.batchnormOut = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.sigmoidOut = tf.keras.layers.Activation('sigmoid')

#             self.layerList = [self.upsample5, self.dropout5,
#                     self.batchnorm6, self.resUp6, self.upsample6, self.dropout6,
#                     self.batchnorm7, self.resUp7, self.upsample7, self.dropout7,
#                     self.batchnorm8, self.resUp8, self.upsample8, self.dropout8,
#                     self.batchnorm9, self.resup9, 
#                     self.conv2dOut,self.batchnormOut,self.sigmoidOut]

#         def call(self, inputs):
#             [latentInputs, result_from_contract_layer] = inputs


#             # layer 5
#             x5 = self.upsample5(latentInputs) #up sample
#             x5 = self.dropout5(x5)
            
#             # layer 6
#             x6 = self.concat6([result_from_contract_layer[4], x5])
#             x6 = self.batchnorm6(x6)
#             x6 = self.resUp6(x6)
#             x6 = self.upsample6(x6)
#             x6 = self.dropout6(x6)

#             # layer 7
#             x7 = self.concat7([result_from_contract_layer[3], x6])
#             x7 = self.batchnorm7(x7)
#             x7 = self.resUp7(x7)
#             x7 = self.upsample7(x7)
#             x7 = self.dropout7(x7)
            
#             # layer 8
#             x8 = self.concat8([result_from_contract_layer[2], x7])
#             x8 = self.batchnorm8(x8)
#             x8 = self.resUp8(x8)
#             x8 = self.upsample8(x8)
#             x8 = self.dropout8(x8)

#             # layer 9
#             x9 = self.concat9([result_from_contract_layer[1], x8])
#             x9 = self.batchnorm9(x9)
#             x9 = self.resup9(x9)

#             # To output
#             xOut = self.conv2dOut(x9)
#             xOut = self.batchnormOut(xOut)
#             xOut = self.sigmoidOut(xOut)
#             return xOut
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate
#             }
#             base_config = super(Architectures.DecoderRfiGanDense_v1, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))

#     class EncoderRfiGanDense_v2(tf.keras.layers.Layer):
#         def __init__(self, batchNormMomentum=0.9, eps=10e-5, dropoutRate=0.0, embeddingRestriction = 1, **kwargs):
#             super(Architectures.EncoderRfiGanDense_v2, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate
#             self.embeddingRestriction = embeddingRestriction

#         def build(self, input_shape):
#             # Input
#             self.batchnorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)

#             # Layer 1
#             self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='SAME')
#             self.batchNorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.relu1 = tf.keras.layers.ReLU()

#             self.resDown1 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxPool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout1 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 2
#             self.resDown2 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout2 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 3
#             self.resDown3 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout3 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 4
#             self.resDown4 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout4 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             self.resDown5 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             channels_num = 1024
#             #_, _, _, channels_num = self.resDown5.input_shape
#             #channels_num *= 2
#             nEmbeddingFilters = channels_num/self.embeddingRestriction
#             self.conv5 = tf.keras.layers.Conv2D(filters=nEmbeddingFilters, kernel_size=(1,1), strides=1, padding='SAME')

#             #self.conv5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1)
            
#             # self.flatten5 = tf.keras.layers.Flatten()
#             # self.dense5 = tf.keras.layers.Dense(524288)
#             # self.reshape = tf.keras.layers.Reshape((8,4,16,1024))

                  
#             self.layerList = [self.batchnorm1,
#                     self.conv1,self.batchNorm1,self.relu1,
#                     self.resDown1,self.maxPool1,self.dropout1,
#                     self.resDown2,self.maxpool2,self.dropout2,
#                     self.resDown3,self.maxpool3,self.dropout3,
#                     self.resDown4,self.maxpool4,self.dropout4,
#                     self.resDown5,self.conv5]

#         def call(self, inputs, training=False):
#             result_from_contract_layer = {}
#             # Input
#             x1 = self.batchnorm1(inputs, training=training)

#             # Layer 1
#             x1 = self.conv1(x1)
#             x1 = self.batchNorm1(x1, training=training)
#             x1 = self.relu1(x1)

#             x1 = self.resDown1(x1, training=training)
#             result_from_contract_layer[1]=x1
#             x1 = self.maxPool1(x1)
#             x1 = self.dropout1(x1, training=training)

#             # layer 2
#             x2 = self.resDown2(x1, training=training)
#             result_from_contract_layer[2]=x2
#             x2 = self.maxpool2(x2)
#             x2 = self.dropout2(x2, training=training)

#             # layer 3
#             x3 = self.resDown3(x2, training=training)
#             result_from_contract_layer[3]=x3
#             x3 = self.maxpool3(x3)
#             x3 = self.dropout3(x3, training=training)
            
#             # layer 4
#             x4 = self.resDown4(x3, training=training)
#             result_from_contract_layer[4]=x4
#             x4 = self.maxpool4(x4)
#             x4 = self.dropout4(x4, training=training)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             x5 = self.resDown5(x4, training=training)
#             x5 = self.conv5(x5)
#             return [x5, result_from_contract_layer]
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate,
#                 'embeddingRestriction' :self.embeddingRestriction
#             }
#             base_config = super(Architectures.EncoderRfiGanDense_v2, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))
       
#     class DecoderRfiGanDense_v2(tf.keras.layers.Layer):
#         def __init__(self, outputChannels, batchNormMomentum, eps, dropoutRate, skipConnections = True, embeddingRestriction = 1, **kwargs):
#             super(Architectures.DecoderRfiGanDense_v2, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate
#             self.outputChannels = outputChannels
#             self.skipConnections = skipConnections
#             self.embeddingRestriction = embeddingRestriction

#         def build(self, input_shape):
#             # layer 5
#             self.upsample5 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps, self.embeddingRestriction)
#             self.dropout5 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 6
#             self.concat6 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm6 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp6 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample6 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout6 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 7
#             self.concat7 =  tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm7 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp7 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample7 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout7 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 8
#             self.concat8 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm8 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp8 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample8 =utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout8 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 9
#             self.concat9 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm9 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resup9 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)

#             # To output
#             self.conv2dOut = tf.keras.layers.Conv2D(filters=self.outputChannels, kernel_size=(1,1), strides=1, padding='VALID', name='conv_3')
#             self.batchnormOut = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.sigmoidOut = tf.keras.layers.Activation('sigmoid')

#             self.layerList = [self.upsample5, self.dropout5,
#                     self.batchnorm6, self.resUp6, self.upsample6, self.dropout6,
#                     self.batchnorm7, self.resUp7, self.upsample7, self.dropout7,
#                     self.batchnorm8, self.resUp8, self.upsample8, self.dropout8,
#                     self.batchnorm9, self.resup9, 
#                     self.conv2dOut,self.batchnormOut,self.sigmoidOut]

#         def call(self, inputs):
#             [latentInputs, result_from_contract_layer] = inputs

#             # layer 5
#             x5 = self.upsample5(latentInputs) #up sample
#             x5 = self.dropout5(x5)
            
#             # layer 6
#             x6 = self.concat6([result_from_contract_layer[4], x5])
#             x6 = self.batchnorm6(x6)
#             x6 = self.resUp6(x6)
#             x6 = self.upsample6(x6)
#             x6 = self.dropout6(x6)

#             # layer 7
#             x7 = self.concat7([result_from_contract_layer[3], x6])
#             x7 = self.batchnorm7(x7)
#             x7 = self.resUp7(x7)
#             x7 = self.upsample7(x7)
#             x7 = self.dropout7(x7)
            
#             # layer 8
#             x8 = self.concat8([result_from_contract_layer[2], x7])
#             x8 = self.batchnorm8(x8)
#             x8 = self.resUp8(x8)
#             x8 = self.upsample8(x8)
#             x8 = self.dropout8(x8)

#             # layer 9
#             x9 = self.concat9([result_from_contract_layer[1], x8])
#             x9 = self.batchnorm9(x9)
#             x9 = self.resup9(x9)

#             # To output
#             xOut = self.conv2dOut(x9)
#             xOut = self.batchnormOut(xOut)
#             xOut = self.sigmoidOut(xOut)
#             return xOut
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate,
#                 'embeddingRestriction' :self.embeddingRestriction
#             }
#             base_config = super(Architectures.DecoderRfiGanDense_v2, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))
  
#     class EncoderRfiGanDense_v3(tf.keras.layers.Layer):
#         def __init__(self, batchNormMomentum=0.9, eps=10e-5, dropoutRate=0.0, embeddingRestriction = 1, **kwargs):
#             super(Architectures.EncoderRfiGanDense_v3, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate
#             self.embeddingRestriction = embeddingRestriction

#         def build(self, input_shape):
#             # Input
#             self.batchnorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)

#             # Layer 1
#             self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='SAME')
#             self.batchNorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.relu1 = tf.keras.layers.ReLU()

#             self.resDown1 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxPool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout1 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 2
#             self.resDown2 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout2 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 3
#             self.resDown3 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout3 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 4
#             self.resDown4 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout4 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 5
#             self.resDown5 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout5 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 6 (bottom 16 * 8 * 1024)
#             self.resDown6 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             channels_num = 2048
#             nEmbeddingFilters = channels_num/self.embeddingRestriction
#             self.conv6 = tf.keras.layers.Conv2D(filters=nEmbeddingFilters, kernel_size=(1,1), strides=1, padding='SAME')

                  
#             self.layerList = [self.batchnorm1,
#                     self.conv1,self.batchNorm1,self.relu1,
#                     self.resDown1,self.maxPool1,self.dropout1,
#                     self.resDown2,self.maxpool2,self.dropout2,
#                     self.resDown3,self.maxpool3,self.dropout3,
#                     self.resDown4,self.maxpool4,self.dropout4,
#                     self.resDown5,self.maxpool5,self.dropout5,
#                     self.resDown6,self.conv6]

#         def call(self, inputs, training=False):
#             result_from_contract_layer = {}
#             # Input
#             x1 = self.batchnorm1(inputs, training=training)

#             # Layer 1
#             x1 = self.conv1(x1)
#             x1 = self.batchNorm1(x1, training=training)
#             x1 = self.relu1(x1)

#             x1 = self.resDown1(x1, training=training)
#             result_from_contract_layer[1]=x1
#             x1 = self.maxPool1(x1)
#             x1 = self.dropout1(x1, training=training)

#             # layer 2
#             x2 = self.resDown2(x1, training=training)
#             result_from_contract_layer[2]=x2
#             x2 = self.maxpool2(x2)
#             x2 = self.dropout2(x2, training=training)

#             # layer 3
#             x3 = self.resDown3(x2, training=training)
#             result_from_contract_layer[3]=x3
#             x3 = self.maxpool3(x3)
#             x3 = self.dropout3(x3, training=training)
            
#             # layer 4
#             x4 = self.resDown4(x3, training=training)
#             result_from_contract_layer[4]=x4
#             x4 = self.maxpool4(x4)
#             x4 = self.dropout4(x4, training=training)

#             # layer 5       4 * 16 * 1024
#             x5 = self.resDown5(x4, training=training)
#             result_from_contract_layer[5]=x5
#             x5 = self.maxpool5(x5)
#             x5 = self.dropout5(x5, training=training)


#             # layer 6 (bottom 2 * 8 * nFilters)
#             x6 = self.resDown6(x5, training=training)
#             x6 = self.conv6(x6)

#             return [x6, result_from_contract_layer]
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate,
#                 'embeddingRestriction' :self.embeddingRestriction
#             }
#             base_config = super(Architectures.EncoderRfiGanDense_v3, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))
       
#     class DecoderRfiGanDense_v3(tf.keras.layers.Layer):
#         def __init__(self, outputChannels, batchNormMomentum, eps, dropoutRate, skipConnections = True, embeddingRestriction = 1, **kwargs):
#             super(Architectures.DecoderRfiGanDense_v3, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate
#             self.outputChannels = outputChannels
#             self.skipConnections = skipConnections # TODO: remove line, not used
#             self.embeddingRestriction = embeddingRestriction

#         def build(self, input_shape):
#             # layer 5
#             self.upsample5 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps, self.embeddingRestriction)
#             self.dropout5 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 6
#             self.concat6 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm6 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp6 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample6 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout6 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 7
#             self.concat7 =  tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm7 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp7 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample7 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout7 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 8
#             self.concat8 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm8 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp8 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample8 =utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout8 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 9
#             self.concat9 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm9 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp9 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample9 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout9 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 10
#             self.concat10 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm10 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resup10 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)

#             # To output
#             self.conv2dOut = tf.keras.layers.Conv2D(filters=self.outputChannels, kernel_size=(1,1), strides=1, padding='VALID', name='conv_3')
#             self.batchnormOut = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.sigmoidOut = tf.keras.layers.Activation('sigmoid')

#             self.layerList = [self.upsample5, self.dropout5,
#                     self.batchnorm6, self.resUp6, self.upsample6, self.dropout6,
#                     self.batchnorm7, self.resUp7, self.upsample7, self.dropout7,
#                     self.batchnorm8, self.resUp8, self.upsample8, self.dropout8,
#                     self.batchnorm9, self.resUp9, self.upsample9, self.dropout9,
#                     self.batchnorm10, self.resup10, 
#                     self.conv2dOut,self.batchnormOut,self.sigmoidOut]

#         def call(self, inputs):
#             [latentInputs, result_from_contract_layer] = inputs

#             # layer 5
#             x5 = self.upsample5(latentInputs) #up sample
#             x5 = self.dropout5(x5)
            
#             # layer 6
#             x6 = self.concat6([result_from_contract_layer[5], x5])
#             x6 = self.batchnorm6(x6)
#             x6 = self.resUp6(x6)
#             x6 = self.upsample6(x6)
#             x6 = self.dropout6(x6)

#             # layer 7
#             x7 = self.concat7([result_from_contract_layer[4], x6])
#             x7 = self.batchnorm7(x7)
#             x7 = self.resUp7(x7)
#             x7 = self.upsample7(x7)
#             x7 = self.dropout7(x7)
            
#             # layer 8
#             x8 = self.concat8([result_from_contract_layer[3], x7])
#             x8 = self.batchnorm8(x8)
#             x8 = self.resUp8(x8)
#             x8 = self.upsample8(x8)
#             x8 = self.dropout8(x8)

#             # layer 9
#             x9 = self.concat9([result_from_contract_layer[2], x8])
#             x9 = self.batchnorm9(x9)
#             x9 = self.resUp9(x9)
#             x9 = self.upsample9(x9)
#             x9 = self.dropout9(x9)

#             # layer 10
#             x10 = self.concat10([result_from_contract_layer[1], x9])
#             x10 = self.batchnorm10(x10)
#             x10 = self.resup10(x10)

#             # To output
#             xOut = self.conv2dOut(x10)
#             xOut = self.batchnormOut(xOut)
#             xOut = self.sigmoidOut(xOut)
#             return xOut
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate,
#                 'embeddingRestriction' :self.embeddingRestriction
#             }
#             base_config = super(Architectures.DecoderRfiGanDense_v3, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))

#     class EncoderRfiGanDense_v4(tf.keras.layers.Layer):
#         def __init__(self, batchNormMomentum=0.9, eps=10e-5, dropoutRate=0.0, **kwargs):
#             super(Architectures.EncoderRfiGanDense_v4, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate

#         def build(self, input_shape):
#             # Input
#             self.batchnorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)

#             # Layer 1
#             self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='SAME')
#             self.batchNorm1 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.relu1 = tf.keras.layers.ReLU()

#             self.resDown1 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxPool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout1 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 2
#             self.resDown2 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout2 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 3
#             self.resDown3 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout3 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 4
#             self.resDown4 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)
#             self.maxpool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
#             self.dropout4 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             self.resDown5 = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)

                  
#             self.layerList = [self.batchnorm1,
#                     self.conv1,self.batchNorm1,self.relu1,
#                     self.resDown1,self.maxPool1,self.dropout1,
#                     self.resDown2,self.maxpool2,self.dropout2,
#                     self.resDown3,self.maxpool3,self.dropout3,
#                     self.resDown4,self.maxpool4,self.dropout4,
#                     self.resDown5]

#         def call(self, inputs, training=False):
#             result_from_contract_layer = {}
#             # Input
#             x1 = self.batchnorm1(inputs, training=training)

#             # Layer 1
#             x1 = self.conv1(x1)
#             x1 = self.batchNorm1(x1, training=training)
#             x1 = self.relu1(x1)

#             x1 = self.resDown1(x1, training=training)
#             result_from_contract_layer[1]=x1
#             x1 = self.maxPool1(x1)
#             x1 = self.dropout1(x1, training=training)

#             # layer 2
#             x2 = self.resDown2(x1, training=training)
#             result_from_contract_layer[2]=x2
#             x2 = self.maxpool2(x2)
#             x2 = self.dropout2(x2, training=training)

#             # layer 3
#             x3 = self.resDown3(x2, training=training)
#             result_from_contract_layer[3]=x3
#             x3 = self.maxpool3(x3)
#             x3 = self.dropout3(x3, training=training)
            
#             # layer 4
#             x4 = self.resDown4(x3, training=training)
#             result_from_contract_layer[4]=x4
#             x4 = self.maxpool4(x4)
#             x4 = self.dropout4(x4, training=training)

#             # Originaly: 16 * 8 * 1024
#             # Originaly swap: 8 * 16 * 1024
#             # current 1 chan: 4 * 16 * 1024     128 freq to 64 freq
#             # current 8 chan: 4 * 16 * 1024

#             # layer 5 (bottom 16 * 8 * 1024)
#             x5 = self.resDown5(x4, training=training)
#             return [x5, result_from_contract_layer]
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate,
#             }
#             base_config = super(Architectures.EncoderRfiGanDense_v4, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))
       
#     class DecoderRfiGanDense_v4(tf.keras.layers.Layer):
#         def __init__(self, outputChannels, batchNormMomentum, eps, dropoutRate, **kwargs):
#             super(Architectures.DecoderRfiGanDense_v4, self).__init__(**kwargs)
#             self.batchNormMomentum = batchNormMomentum
#             self.eps = eps
#             self.dropoutRate = dropoutRate
#             self.outputChannels = outputChannels

#         def build(self, input_shape):
#             # layer 5
#             self.upsample5 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout5 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 6
#             self.concat6 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm6 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp6 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample6 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout6 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 7
#             self.concat7 =  tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm7 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp7 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample7 = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout7 = tf.keras.layers.Dropout(rate = self.dropoutRate)
            
#             # layer 8
#             self.concat8 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm8 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resUp8 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)
#             self.upsample8 =utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps)
#             self.dropout8 = tf.keras.layers.Dropout(rate = self.dropoutRate)

#             # layer 9
#             self.concat9 = tf.keras.layers.Concatenate(axis=-1)
#             self.batchnorm9 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.resup9 = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps)

#             # To output
#             self.conv2dOut = tf.keras.layers.Conv2D(filters=self.outputChannels, kernel_size=(1,1), strides=1, padding='VALID', name='conv_3')
#             self.batchnormOut = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)
#             self.sigmoidOut = tf.keras.layers.Activation('sigmoid')

#             self.layerList = [self.upsample5, self.dropout5,
#                     self.batchnorm6, self.resUp6, self.upsample6, self.dropout6,
#                     self.batchnorm7, self.resUp7, self.upsample7, self.dropout7,
#                     self.batchnorm8, self.resUp8, self.upsample8, self.dropout8,
#                     self.batchnorm9, self.resup9, 
#                     self.conv2dOut,self.batchnormOut,self.sigmoidOut]

#         def call(self, inputs):
#             [latentInputs, result_from_contract_layer] = inputs

#             # layer 5
#             x5 = self.upsample5(latentInputs) #up sample
#             x5 = self.dropout5(x5)
            
#             # layer 6
#             x6 = self.concat6([result_from_contract_layer[4], x5])
#             x6 = self.batchnorm6(x6)
#             x6 = self.resUp6(x6)
#             x6 = self.upsample6(x6)
#             x6 = self.dropout6(x6)

#             # layer 7
#             x7 = self.concat7([result_from_contract_layer[3], x6])
#             x7 = self.batchnorm7(x7)
#             x7 = self.resUp7(x7)
#             x7 = self.upsample7(x7)
#             x7 = self.dropout7(x7)
            
#             # layer 8
#             x8 = self.concat8([result_from_contract_layer[2], x7])
#             x8 = self.batchnorm8(x8)
#             x8 = self.resUp8(x8)
#             x8 = self.upsample8(x8)
#             x8 = self.dropout8(x8)

#             # layer 9
#             x9 = self.concat9([result_from_contract_layer[1], x8])
#             x9 = self.batchnorm9(x9)
#             x9 = self.resup9(x9)

#             # To output
#             xOut = self.conv2dOut(x9)
#             xOut = self.batchnormOut(xOut)
#             xOut = self.sigmoidOut(xOut)
#             return xOut
        
#         def get_config(self):
#             config = {
#                 'batchNormMomentum': self.batchNormMomentum,
#                 'epsilon': self.eps,
#                 'dropoutRate' : self.dropoutRate,
#                 'embeddingRestriction' :self.embeddingRestriction
#             }
#             base_config = super(Architectures.DecoderRfiGanDense_v4, self).get_config()
#             return dict(list(base_config.items()) + list(config.items()))

#     class DecoderClassificationHead_v1(tf.keras.layers.Layer):
#         def __init__(self, nClasses, **kwargs):
#             super(Architectures.DecoderClassificationHead_v1, self).__init__(**kwargs)
#             self.nClasses = nClasses

#         def build(self, input_shape):
#             self.flatten = tf.keras.layers.Flatten()
#             self.dense1 = tf.keras.layers.Dense(20)
#             self.dense2 = tf.keras.layers.Dense(self.nClasses,activation='softmax') # TODO: change to sigmoid when multioutput

#         def call(self, inputs):
#             # layer 5
#             x5 = self.flatten(inputs)
#             x5 = self.dense1(x5)
#             x5 = self.dense2(x5)
#             return x5

class Models():
    class Gan_Encoder_v1(tf.keras.Model):
        def __init__(self, inputShape, contractingResults = True, batchNormMomentum=0.9, eps=10e-5, dropoutRate=0.0, depth = 4, scaleInputShape = None, modConv = False, batchSize=10, modConvActivation=None,modConvResDownNormAdd=None,nFeaturesW=None):
            super(Models.Gan_Encoder_v1, self).__init__()
            self.batchNormMomentum = batchNormMomentum
            self.eps = eps
            self.dropoutRate = dropoutRate
            self.contractingResults = contractingResults
            self.depth = depth
            self.scaleInput = scaleInputShape is not None
            self.inputShape=inputShape
            self.modConv = modConv
            self.batchSize = batchSize
            self.modConvActivation = modConvActivation
            self.modConvResDownNormAdd = modConvResDownNormAdd
            self.nFeaturesW = nFeaturesW

            if self.modConv:
                # nLatentWTiles = self.depth*5 # all residual units
                # nLatentWTiles += 4 # input modConv2d
                self.nLatentWTiles=24
            else:
                self.nLatentWTiles=0

            self.encoder = self.buildEncoder(inputShape)#, scaleInputShape)
            self.encoder.build((None,) + inputShape)


        def call(self, inputs):
            if self.scaleInput:
                encoderIn = tf.keras.layers.experimental.preprocessing.Resizing(self.inputShape[0],self.inputShape[1])(inputs)
            else:
                encoderIn = inputs
            return self.encoder(encoderIn)
        
        def finishBuild(self,inputs,result_from_contract_layer, lastLayer):
            outputs = lastLayer
            if self.contractingResults:
                return tf.keras.Model(inputs, [outputs, result_from_contract_layer])
            else:
                return tf.keras.Model(inputs, outputs)
       
        def buildEncoder(self,inputShape):
            result_from_contract_layer = {}
            obervationInput = tf.keras.layers.Input(inputShape)

            if self.modConv:
                latentW = tf.keras.layers.Input((self.nLatentWTiles,self.nFeaturesW))
                inputs = [obervationInput,latentW]
                latentIndex = 0
            else:
                inputs = obervationInput

            # Layer 0, 64 x 256 x n
            if self.modConv:
                #x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(obervationInput)
                #x = layers.ModulatedConv2d(32,self.batchSize,activationFunction=self.modConvActivation, name='Encoder_input_ModConv')(x,latentW[:,latentIndex+4,:])
                #x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
                x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(obervationInput)
                x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='SAME')(x)
                x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(x)
                x = tf.keras.layers.ReLU()(x)
                x = utils.models.layers.ModConvResUnitDown(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResDownNormAdd, name='Encoder_depth_input_resDownModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                latentIndex += 4
            else:
                x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(obervationInput)
                x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='SAME')(x)
                x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(x)
                x = tf.keras.layers.ReLU()(x)
                x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
            if self.depth == 0:
                return self.finishBuild(inputs,result_from_contract_layer, x)
            if self.contractingResults: result_from_contract_layer[0]=x


            # Layer 1, 32 x 128 x n
            if self.modConv:
                x = layers.ModulatedConv2d(64,self.batchSize,down=True,activationFunction=self.modConvActivation, name='Encoder_depth_1_downSample_ModConv')(x,latentW[:,latentIndex,:])
                latentIndex += 1
                #x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
                x = utils.models.layers.ModConvResUnitDown(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResDownNormAdd, name='Encoder_depth_1_resDownModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                latentIndex += 4
            else:
                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(x)
                x = tf.keras.layers.Dropout(rate = self.dropoutRate)(x)
                x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
            if self.depth == 1:
                return self.finishBuild(inputs,result_from_contract_layer, x)
            if self.contractingResults: result_from_contract_layer[1]=x

            
            # layer 2, 16 x 64 x n
            if self.modConv:
                x = layers.ModulatedConv2d(128,self.batchSize,down=True,activationFunction=self.modConvActivation, name='Encoder_depth_2_downSample_ModConv')(x,latentW[:,latentIndex,:])
                latentIndex += 1
                #x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
                x = utils.models.layers.ModConvResUnitDown(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResDownNormAdd, name='Encoder_depth_2_resDownModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                latentIndex += 4
            else:
                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(x)
                x = tf.keras.layers.Dropout(rate = self.dropoutRate)(x)
                x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
            if self.depth == 2:
                return self.finishBuild(inputs,result_from_contract_layer, x)
            if self.contractingResults: result_from_contract_layer[2]=x

                
            # layer 3, 8 x 32 x n
            if self.modConv:
                x = layers.ModulatedConv2d(256,self.batchSize,down=True,activationFunction=self.modConvActivation, name='Encoder_depth_3_downSample_ModConv')(x,latentW[:,latentIndex,:])
                latentIndex += 1
                #x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
                x = utils.models.layers.ModConvResUnitDown(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResDownNormAdd, name='Encoder_depth_3_resDownModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                latentIndex += 4
            else:
                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(x)
                x = tf.keras.layers.Dropout(rate = self.dropoutRate)(x)
                x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
            if self.depth == 3:
                return self.finishBuild(inputs,result_from_contract_layer, x)
            if self.contractingResults: result_from_contract_layer[3]=x

   
            # layer 4, 4 x 16 x n
            if self.modConv:
                x = layers.ModulatedConv2d(512,self.batchSize,down=True,activationFunction=self.modConvActivation, name='Encoder_depth_4_downSample_ModConv')(x,latentW[:,latentIndex,:])
                latentIndex += 1
                #x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
                x = utils.models.layers.ModConvResUnitDown(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResDownNormAdd, name='Encoder_depth_4_resDownModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                latentIndex += 4
            else:
                x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')(x)
                x = tf.keras.layers.Dropout(rate = self.dropoutRate)(x)
                x = utils.models.layers.ResUnitDown(self.batchNormMomentum,self.eps)(x)
            return self.finishBuild(inputs,result_from_contract_layer, x)

    class Gan_DecoderMask_v1(tf.keras.Model):
        def __init__(self, inputShape, randomContractingLayers, nOutputChannels,batchSize, batchNormMomentum=0.9, eps=10e-5, dropoutRate=0.0, depth = 4, convertInputFilters=None, selfSupervised = False, concatDinoEncoder=False,nFeaturesZ=None, nFeaturesW = 512, modConv=False, modConvActivation=None, modConvResUpNormAdd=None, styleMappingLrMul = None, nMappingLayers= 3, styleMappingNormInput = True,externalMappingNetwork = False,**kwargs):
            super(Models.Gan_DecoderMask_v1, self).__init__()
            self.batchNormMomentum = batchNormMomentum
            self.eps = eps
            self.dropoutRate = dropoutRate
            self.result_from_contract_layer = randomContractingLayers
            self.depth = depth
            self.convertInputFilters = convertInputFilters
            self.selfSupervised = selfSupervised
            self.concatDinoEncoder = concatDinoEncoder
            #self.nFeaturesZ is not None = nFeaturesZ is not None
            self.nFeaturesZ = nFeaturesZ
            self.nFeaturesW = nFeaturesW
            self.batchSize = batchSize
            self.modConvActivation = modConvActivation
            self.modConv = modConv
            self.modConvResUpNormAdd = modConvResUpNormAdd
            self.styleMappingLrMul = styleMappingLrMul
            self.nMappingLayers = nMappingLayers
            self.styleMappingNormInput=styleMappingNormInput
            self.externalMappingNetwork = externalMappingNetwork

            self.decoder = self.buildDecoder(inputShape, nOutputChannels)
            self.decoder.build((None,) + inputShape)

        def call(self, inputs):
            return self.decoder(inputs)
        
        def buildDecoder(self,inputShape,nOutputChannels):    
            latentIndex = 0
            inputsContractingLayers = []
            for depthIdx, shape in enumerate(self.result_from_contract_layer.keys()):
                if depthIdx < self.depth:
                    contractingInput = tf.keras.layers.Input(shape=self.result_from_contract_layer[shape].shape[1:])
                    inputsContractingLayers.append(contractingInput)

            if self.nFeaturesZ is not None:
                nLatentWTiles = 0
                if self.modConv:
                    nLatentWTiles = self.depth*4 # all residual units
                    nLatentWTiles += self.depth # all upsample units
                    nLatentWTiles += 1 # last modConv2d

                if self.externalMappingNetwork:
                    inputsZ = tf.keras.layers.Input((nLatentWTiles,self.nFeaturesW))
                    latentW = inputsZ
                else:
                    inputsZ = tf.keras.layers.Input(self.nFeaturesZ)
                    latentW = layers.DinoMappingNetwork(nFeaturesW=self.nFeaturesW,dlatent_broadcast=nLatentWTiles, nLayers=self.nMappingLayers, mapping_lrmul = self.styleMappingLrMul, activation=self.modConvActivation, normalizeInput=self.styleMappingNormInput, name='Decoder_depth_bottleneck_dinoMappingNetwork')(inputsZ)
                
            # layer 5: starts with 4x16x1024
            inputs = tf.keras.layers.Input(inputShape)

            if self.modConv:
                x = layers.ModulatedConv2d(int(1024/2),self.batchSize, up=True, kernel=2,activationFunction=self.modConvActivation, name='Decoder_depth_bottleneck_upSampleModConv')(inputs,latentW[:,latentIndex,:])
                latentIndex +=1
            else:
                x = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps, name='Decoder_depth_bottleneck_upSample')(inputs)
            x = tf.keras.layers.Dropout(rate = self.dropoutRate, name='Decoder_depth_bottleneck_dropout')(x)
            
            if self.concatDinoEncoder: # Keep the original implementation completely intact
                dinoInputs = tf.keras.layers.Input(inputShape)
                xDino = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps, name='Decoder_depth_bottleneck_upSampleDino')(dinoInputs)
                xDino = tf.keras.layers.Dropout(rate = self.dropoutRate, name='Decoder_depth_bottleneck_dropoutDino')(xDino) # Out: 8,32,512

                x = tf.keras.layers.Concatenate(axis=-1, name='Decoder_depth_bottleneck_concatDino')([xDino, x]) # Out: 8,32,1024
                x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps, name='Decoder_depth_bottleneck_batchNormDino')(x)
                x = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps, name='Decoder_depth_bottleneck_resUpDino')(x) # Out: 8,32,512
            
            # layer 6: starts with 8,32,512
            if self.depth >= 4:
                x = tf.keras.layers.Concatenate(axis=-1, name='Decoder_depth_4_concat')([inputsContractingLayers[3], x])
                if self.modConv:
                    x = utils.models.layers.ModConvResUnitUp(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResUpNormAdd, name='Decoder_depth_4_resUpModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                    x = layers.ModulatedConv2d(int(512/2),self.batchSize, up=True, kernel=2,activationFunction=self.modConvActivation, name='Decoder_depth_4_upSampleModConv')(x,latentW[:,latentIndex+4,:])
                    latentIndex += 5
                else:
                    x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps, name='Decoder_depth_4_batchNorm')(x)
                    x = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps, name='Decoder_depth_4_resUp')(x)
                    x = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps, name='Decoder_depth_4_upSample')(x)
                x = tf.keras.layers.Dropout(rate = self.dropoutRate, name='Decoder_depth_4_dropout')(x)

            # layer 7: starts with 16,64,256
            if self.depth >= 3:
                x = tf.keras.layers.Concatenate(axis=-1, name='Decoder_depth_3_concat')([inputsContractingLayers[2], x])
                if self.modConv:
                    x = utils.models.layers.ModConvResUnitUp(self.batchSize,activationFunction=self.modConvActivation, normalizeAdd=self.modConvResUpNormAdd,name='Decoder_depth_3_resUpModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                    x = layers.ModulatedConv2d(int(256/2),self.batchSize, up=True, kernel=2,activationFunction=self.modConvActivation, name='Decoder_depth_3_upSampleModConv')(x,latentW[:,latentIndex+4,:])
                    latentIndex += 5
                else:
                    x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps, name='Decoder_depth_3_batchNorm')(x)
                    x = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps, name='Decoder_depth_3_resUp')(x)
                    x = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps, name='Decoder_depth_3_upSample')(x)
                x = tf.keras.layers.Dropout(rate = self.dropoutRate, name='Decoder_depth_3_dropout')(x)
            
            # layer 8: starts with 32,128,128
            if self.depth >= 2:
                x = tf.keras.layers.Concatenate(axis=-1, name='Decoder_depth_2_concat')([inputsContractingLayers[1], x])
                if self.modConv:
                    x = utils.models.layers.ModConvResUnitUp(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResUpNormAdd, name='Decoder_depth_2_resUpModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                    x = layers.ModulatedConv2d(int(128/2),self.batchSize, up=True, kernel=2,activationFunction=self.modConvActivation, name='Decoder_depth_2_upSampleModConv')(x,latentW[:,latentIndex+4,:])
                    latentIndex += 5
                else:
                    x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps, name='Decoder_depth_2_batchNorm')(x)
                    x = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps, name='Decoder_depth_2_resUp')(x)
                    x = utils.models.layers.UpSampleLayer(self.batchNormMomentum,self.eps, name='Decoder_depth_2_upSample')(x)
                x = tf.keras.layers.Dropout(rate = self.dropoutRate, name='Decoder_depth_2_dropout')(x)

            # layer 9: starts with 64,256,64
            if self.depth >= 1:
                x = tf.keras.layers.Concatenate(axis=-1, name='Decoder_depth_1_concat')([inputsContractingLayers[0], x])
                if self.modConv:
                    x = utils.models.layers.ModConvResUnitUp(self.batchSize,activationFunction=self.modConvActivation,normalizeAdd=self.modConvResUpNormAdd, name='Decoder_depth_1_resUpModConv')(x,latentW[:,latentIndex:latentIndex+4,:])
                    latentIndex += 4
                else:
                    x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps, name='Decoder_depth_1_batchNorm')(x)
                    x = utils.models.layers.ResUnitUp(self.batchNormMomentum, self.eps, name='Decoder_depth_1_resUp')(x)

            # To output
            if self.modConv:
                #x = layers.ModulatedConv2d(nOutputChannels,self.batchSize, kernel=1,activationFunction=None, name='Decoder_depth_0_modConv')(x,latentW[:,latentIndex,:])
                x = layers.ModulatedConv2d(nOutputChannels,self.batchSize, kernel=1,activationFunction=self.modConvActivation, name='Decoder_depth_0_modConv')(x,latentW[:,latentIndex,:])
                latentIndex +=1
            else:
                x = tf.keras.layers.Conv2D(filters=nOutputChannels, kernel_size=(1,1), strides=1, padding='VALID', name='Decoder_depth_0_conv2d')(x)
                x = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps, name='Decoder_depth_0_batchNorm')(x)

            if self.selfSupervised:
                outputs = tf.keras.layers.Activation('tanh', name='Decoder_depth_0_activation')(x)
            else:
                outputs = tf.keras.layers.Activation('sigmoid', name='Decoder_depth_0_activation')(x)
            
            if self.concatDinoEncoder and self.nFeaturesZ is not None:
                return tf.keras.Model([inputs,inputsContractingLayers, dinoInputs,inputsZ],outputs, name='Decoder')
            elif self.concatDinoEncoder:
                return tf.keras.Model([inputs,inputsContractingLayers, dinoInputs],outputs, name='Decoder')
            elif self.nFeaturesZ is not None:
                return tf.keras.Model([inputs,inputsContractingLayers, inputsZ],outputs, name='Decoder')
            else:
                return tf.keras.Model([inputs,inputsContractingLayers],outputs, name='Decoder')
                
    # class Gan_DecoderClass_v1(tf.keras.Model):
    #     def __init__(self, inputShape, nClasses,batchNormMomentum=0.9, eps=10e-5, dropoutRate = 0.1):
    #         super(Models.Gan_DecoderClass_v1, self).__init__()
    #         self.nClasses = nClasses
    #         self.batchNormMomentum = batchNormMomentum
    #         self.eps = eps
    #         self.dropoutRate = dropoutRate
    #         self.result_from_contract_layer = {}
    #         self.decoder = self.buildDecoder(inputShape)

    #     def call(self, inputs):
    #         return self.decoder(inputs)
        
    #     def buildDecoder(self,inputShape):    
    #         inputs = tf.keras.layers.Input(inputShape)

    #         # conv_1
    #         x5 = tf.keras.layers.Conv2D(filters=int(inputShape[2]/2), kernel_size=(3,3), strides=1, padding='SAME')(inputs)
    #         x5 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(x5)
    #         x5 = tf.keras.layers.ReLU()(x5)
    #         x5 = tf.keras.layers.Dropout(rate = self.dropoutRate)(x5)

    #         # conv_2
    #         x5 = tf.keras.layers.Conv2D(filters=int(inputShape[2]/2), kernel_size=(3,3), strides=1, padding='SAME')(x5)
    #         x5 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(x5)
    #         x5 = tf.keras.layers.ReLU()(x5)
    #         x5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(x5)
    #         x5 = tf.keras.layers.Dropout(rate = self.dropoutRate)(x5)

    #         x5 = tf.keras.layers.Conv2D(filters=int(inputShape[2]/4), kernel_size=(1,1), strides=1, padding='SAME')(x5)
    #         x5 = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(x5)
    #         x5 = tf.keras.layers.ReLU()(x5)
    #         x5 = tf.keras.layers.Dropout(rate = self.dropoutRate)(x5)

    #         x5 = tf.keras.layers.Flatten()(x5)
    #         x5 = tf.keras.layers.Dense(20)(x5)
    #         x5 = tf.keras.layers.Dense(20)(x5)
    #         outputs = tf.keras.layers.Dense(self.nClasses,activation='softmax')(x5) # TODO: change to sigmoid when multioutp
    #         return tf.keras.Model(inputs,outputs)

    class Discriminator_v1(tf.keras.Model):
        def __init__(self, inputShape, batchNormMomentum, eps, dropoutRate, lamb):
            super(Models.Discriminator_v1, self).__init__()
            self.batchNormMomentum = batchNormMomentum
            self.eps = eps
            self.dropoutRate = dropoutRate
            self.lamb = lamb
            #self.nClasses = nClasses
            self.discriminator = self.buildDiscriminator(inputShape)

        def call(self, inputs):
            return self.discriminator(inputs)
        
        def buildDiscriminator(self, inputShape):
            inputs = tf.keras.layers.Input(inputShape)

            #layer_1_expand
            result_conv=tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1,padding='SAME',name='d_conv1')(inputs)
            normed_batch = tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(result_conv)
            h1 = tf.keras.layers.ReLU()(normed_batch)
            l1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(h1)
            #(128,64,32)

            #layer_2_expand
            result_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='SAME', name='d_conv2')(l1)
            normed_batch =tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(result_conv)
            h2 = tf.keras.layers.ReLU()(normed_batch)
            l2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(h2)
            #(64,32,64)

            # layer_3_expand
            result_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='SAME', name='d_conv3')(l2)
            normed_batch =tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(result_conv)
            h3 = tf.keras.layers.ReLU()(normed_batch)
            l3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(h3)
            # (32,16,128)

            # layer_4_expand
            result_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=1, padding='SAME', name='d_conv4')(l3)
            normed_batch =tf.keras.layers.BatchNormalization(momentum=self.batchNormMomentum,epsilon=self.eps)(result_conv)
            h4 = tf.keras.layers.ReLU()(normed_batch)
            l4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(h4)
            
            shape=l4.shape
            #size0= int(shape[0]) # 8
            size1 = int(shape[1]) # 4
            size2 = int(shape[2]) # 16
            #l5_flat = tf.reshape(l4, [size0,-1])
            l5_flat = tf.keras.layers.Flatten()(l4)

            # labels = tf.keras.layers.Input(self.nClasses,batchSize)
            # concatenated = tf.keras.layers.Concatenate(axis=1)([l5_flat,labels])
            # denseOne = tf.keras.layers.Dense(20)(concatenated)
            # l5 = tf.keras.layers.Dense(8)(denseOne)
            # return tf.keras.Model([inputs,labels], [tf.nn.sigmoid(l5), l5])
        
            l5 = utils.models.layers.DiscLayer5(size1, size2, self.lamb, name='custom_layer')(l5_flat) # 8x8
            return tf.keras.Model(inputs, [tf.nn.sigmoid(l5), l5])

class ModelToFeatures():
    def __init__(self, encoder, modelDir):
        self.encoder = encoder
        checkpoint = tf.train.Checkpoint(generatorEncoder = self.encoder)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, modelDir, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
                allVariables = tf.train.list_variables(ckpt_manager.latest_checkpoint)
                encoderVariables = [variable[0] for variable in allVariables if variable[0].startswith('generatorEncoder')]
                loaded_variables = {}
                for variable_name in encoderVariables:
                    variable_value = tf.train.load_variable(ckpt_manager.latest_checkpoint, variable_name)
                    loaded_variables[variable_name] = variable_value

                for variable_name, variable_value in loaded_variables.items():
                    setattr(self.encoder, variable_name, variable_value)
        else:
            raise Exception("No checkpoint found")
        
    def predict(self, dataX):
        features = self.encoder(dataX)
        return features

def printLayerSummary(layer):
    line_length = 65
    print('_' * line_length)
    layers = layer.layerList
    for nestedLayer in layers:
        appendSpace = 20 - len(nestedLayer.name)

        printLine = nestedLayer.name
        for _ in range(appendSpace):
            printLine += " "
        printLine += '\t'
        
        for dimIndex, dimension in enumerate(nestedLayer._build_input_shape.dims):
            printLine +=str(dimension.value)
            if dimIndex != (len(nestedLayer._build_input_shape)-1):
                printLine += " x "
        print(printLine)