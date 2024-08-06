import os
import pickle
import random
import re
import warnings
import cv2
import numpy as np
import hashlib
import base64

import h5py
import tensorflow as tf

import utils as utils

linearRepresentation = ['XX', 'XY', 'YX','YY']

class PretextGenerator():
    def __init__(self, pretextTask, normalizationMethod):
        self.normalizationMethod = normalizationMethod
        if pretextTask == 0:
            self.pretextFunction = self.pretext0_xIsy
            self.shapeX = (64, 256, 8)
            self.shapeY = (64, 256, 8)
        elif pretextTask == 1:
            self.pretextFunction = self.pretext1_PredictMagnitude
            self.shapeX = (64, 256, 8)
            self.shapeY = (64, 256, 1)
        else:
            raise Exception("Invalid pretext task.")

    def __call__(self, batchX):
        pretextX, pretextY = self.pretextFunction(batchX)
        return pretextX, pretextY
    
    def getDataShape(self):
        return [self.shapeX, self.shapeY]
    
    def pretext0_xIsy(self, batchX):
        # Pretext task 0: x is y
        batchX = NormalizeConvertChannels(batchX, self.normalizationMethod, 8)
        return batchX, batchX
    
    def pretext1_PredictMagnitude(self, batchX):
        # Pretext task 1: predict the magnitude of the visibility
        magnitudeX = np.abs(batchX)
        squareX = np.square(magnitudeX)
        sumX = np.sum(squareX,axis=-1,keepdims=True)
        combinedMagnitudeX = np.sqrt(sumX)

        batchX = NormalizeConvertChannels(batchX, self.normalizationMethod, 8)
        combinedMagnitudeX = NormalizeConvertChannels(combinedMagnitudeX, self.normalizationMethod, 1)
        return batchX, combinedMagnitudeX

def NormalizeComplex(dataX, normalizationMethod):
    # Convert it to a batch format is single image is given
    removeFirstAxis = False
    if len(dataX.shape)==3:
        dataX = np.expand_dims(dataX, axis=0)
        removeFirstAxis=True

    magnitudeX = np.abs(dataX)
    normalizedMagnitudeX = magnitudeX.copy()
    if normalizationMethod is None:
        warnings.warn('No normalization method is set')
        #return dataX
    elif normalizationMethod == 0:
        warnings.warn('No normalization method is set')
        #return dataX
    elif normalizationMethod == 5:  # 5 median and mad
        raise Exception('Normalization method without clipping should not be used')
        flattenedPerChannel = np.reshape(normalizedMagnitudeX, (-1, normalizedMagnitudeX.shape[-1]))
        medianIntensity = np.median(flattenedPerChannel)
        mad = np.median(np.abs(normalizedMagnitudeX - medianIntensity))
        normalizedMagnitudeX = (normalizedMagnitudeX - medianIntensity) / (1.4826 * mad)       
    elif normalizationMethod == 6:  # 6 median and mad per channel
        raise Exception('Normalization method without clipping should not be used')
        flattenedBatchPerChannel = np.reshape(normalizedMagnitudeX, (normalizedMagnitudeX.shape[0], -1, normalizedMagnitudeX.shape[-1]))
        medianIntensity = np.median(flattenedBatchPerChannel, axis=1)
        mad = np.median(np.abs(flattenedBatchPerChannel - medianIntensity[:,np.newaxis,:]),axis=1)
        normalizedMagnitudeX = (normalizedMagnitudeX - medianIntensity[:,np.newaxis,np.newaxis,:]) / (mad[:,np.newaxis,np.newaxis,:])
    elif normalizationMethod == 10: # 10 mean and std per channel
        warnings.warn('Normalization method without clipping should not be used')
        mean = np.mean(normalizedMagnitudeX, axis=(1,2))
        std = np.std(normalizedMagnitudeX, axis=(1,2))
        normalizedMagnitudeX -= mean[:,np.newaxis,np.newaxis,:]
        normalizedMagnitudeX /= std[:,np.newaxis,np.newaxis,:]
    # elif normalizationMethod == 11: # 11 rank transform
        from scipy.special import erfinv
        flattenedImage = np.reshape(magnitudeX, (1, -1))[0]
        N = flattenedImage.shape[0]
        temp = flattenedImage.argsort()
        rank_x = temp.argsort() / N
        rank_x -= rank_x.mean()
        rank_x *= 2
        efi_x = erfinv(rank_x)
        efi_x -= efi_x.mean()
        normalizedMagnitudeX = efi_x.reshape(normalizedMagnitudeX.shape)
    elif normalizationMethod == 12: # median and mad per channel clipped
        flattenedBatchPerChannel = np.reshape(normalizedMagnitudeX, (normalizedMagnitudeX.shape[0], -1, normalizedMagnitudeX.shape[-1]))
        medianIntensity = np.median(flattenedBatchPerChannel, axis=1)
        mad = np.median(np.abs(flattenedBatchPerChannel - medianIntensity[:,np.newaxis,:]),axis=1)
        normalizedMagnitudeX = (normalizedMagnitudeX - medianIntensity[:,np.newaxis,np.newaxis,:]) / (mad[:,np.newaxis,np.newaxis,:])
        clipStd = 3
        stdValue = 4.5
        normalizedMagnitudeX = np.clip(normalizedMagnitudeX, 0, stdValue)
        normalizedMagnitudeX *= (clipStd/stdValue)
    elif normalizationMethod == 13: # mean and std per channel clipped
        mean = np.mean(normalizedMagnitudeX, axis=(1,2))
        std = np.std(normalizedMagnitudeX, axis=(1,2))
        normalizedMagnitudeX -= mean[:,np.newaxis,np.newaxis,:]
        normalizedMagnitudeX /= std[:,np.newaxis,np.newaxis,:]
        clipStd = 3
        normalizedMagnitudeX = np.clip(normalizedMagnitudeX, 0, clipStd)
    elif normalizationMethod == 14: # median and mad per pixel-wise clipped
        flattenedBatchPerChannel = np.reshape(normalizedMagnitudeX, (normalizedMagnitudeX.shape[0], -1, normalizedMagnitudeX.shape[-1]))
        medianIntensity = np.median(flattenedBatchPerChannel)#, axis=1)
        mad = np.median(np.abs(flattenedBatchPerChannel - medianIntensity))#[:,np.newaxis,:]))#,axis=1)
        normalizedMagnitudeX = (normalizedMagnitudeX - medianIntensity) / (mad)
        #normalizedMagnetudeX = (normalizedMagnetudeX - medianIntensity[:,np.newaxis,np.newaxis,:]) / (mad[:,np.newaxis,np.newaxis,:])
        clipStd = 3
        stdValue = 4.5
        normalizedMagnitudeX = np.clip(normalizedMagnitudeX, 0, stdValue)
        normalizedMagnitudeX *= (clipStd/stdValue)
    elif normalizationMethod == 15: # 10 mean and std pixel-wise clipped
        mean = np.mean(normalizedMagnitudeX)#, axis=(1,2))
        std = np.std(normalizedMagnitudeX)#, axis=(1,2))
        normalizedMagnitudeX -= mean#[:,np.newaxis,np.newaxis,:]
        normalizedMagnitudeX /= std#[:,np.newaxis,np.newaxis,:]
        clipStd = 3
        normalizedMagnitudeX = np.clip(normalizedMagnitudeX, 0, clipStd)
    else:
        raise Exception('Invalid normalization method')

    normalizeFactor = np.divide(normalizedMagnitudeX, magnitudeX, out=np.zeros_like(magnitudeX,dtype = np.float32), where=magnitudeX!=0)
    normalizedX = np.multiply(dataX,normalizeFactor)

    if removeFirstAxis:
        normalizedX = normalizedX[0]
    return normalizedX

class PreprocessDino:
    def __init__(self,global_crops_scale,local_crops_scale,local_crops_number,global_image_size=[224, 224],local_image_size=[96, 96],nChannels = 3,normalizationMethod=None, mode=None, augmentation = 'f'):
        self.local_image_size = local_image_size
        self.global_image_size = global_image_size
        self.local_crops_scale = local_crops_scale
        self.nLocalCrops = local_crops_number
        self.nGlobalCrops = 2
        self.global_crops_scale = global_crops_scale
        self.height = global_image_size[0]
        self.width = global_image_size[1]
        self.nChannels = nChannels
        self.normalizationMethod = normalizationMethod
        self.mode = mode
        
        self.augmentFlip = False
        self.augmentColor = False
        if augmentation == 'f' or augmentation == 'fc':
            self.augmentFlip = True
        if augmentation == 'c' or augmentation == 'fc':
            self.augmentColor = True

        self.datasetMeans = [6.21184, 6.2020483, 6.1998925, 6.200046]
        self.datasetStd =  [0.3797695, 0.3832017, 0.3883769, 0.38410324]

        scaling_hw = np.stack([self.height, self.width], axis=0)
        self.globalCropSize = (int(scaling_hw[0]*self.global_crops_scale[0]),int(scaling_hw[1]*self.global_crops_scale[1]),self.nChannels) 
        self.localCropSize = (int(scaling_hw[0]*self.local_crops_scale[0]),int(scaling_hw[1]*self.local_crops_scale[1]),self.nChannels)

    def _augmentFlip(self, image, horizontal = True, vertical = True):
        if horizontal:
            if random.random()<0.5:
                image = np.flip(image, axis=1)
            else:
                image = image
        if vertical:
            if random.random()<0.5:
                image = np.flip(image, axis=0)
            else:
                image = image
        return image
    
    def _augmentCropGlobal(self, image):
        top = np.random.randint(0, self.height - self.globalCropSize[0] + 1)
        left = np.random.randint(0, self.width - self.globalCropSize[1] + 1)
        image = image[top:top+self.globalCropSize[0], left:left+self.globalCropSize[1], :]
        image = tf.image.resize(image, self.global_image_size, method="bicubic")
        return image

    def _augmentCropLocal(self, image):
        top = np.random.randint(0, self.height - self.localCropSize[0] + 1)
        left = np.random.randint(0, self.width - self.localCropSize[1] + 1)
        image = image[top:top+self.localCropSize[0], left:left+self.localCropSize[1], :]
        image = tf.image.resize(image, self.local_image_size, method="bicubic")
        return image
    
    def _augmentMagnitude(self, magnitude):
        randomBrightness = np.random.uniform(0.6, 1.4)
        randomChannels = np.random.uniform(0.8, 1.2, 4)

        augmented = magnitude*randomBrightness
        augmented *= randomChannels
        return augmented

    def _augmentColor(self, image):
        if self.nChannels == 4:
            augmented = self._augmentMagnitude(image)
        elif self.nChannels == 8:
            dataReal = image[:,:,:4]
            dataImag = image[:,:,4:]
            magnitude = np.sqrt(dataReal**2 + dataImag**2)
            augmentedMagnitude = self._augmentMagnitude(magnitude)

            augmentFactor = np.divide(augmentedMagnitude, magnitude, out=np.zeros_like(magnitude,dtype = np.float32), where=magnitude!=0)
            dataReal = np.multiply(dataReal,augmentFactor)
            dataImag = np.multiply(dataImag,augmentFactor)
            augmented = np.concatenate([dataReal,dataImag],axis=-1)
        else:
            raise Exception('Color augmentation not implemnted for 8 channels')
        return augmented

    def __call__(self, batchX):
        # the input is complex. If the final result should be 3 or 4 channels, already take the absolute value to reduce the augmentation load
        normalizedComplexX = NormalizeComplex(batchX, self.normalizationMethod)

        if self.nChannels == 4:
            normalizedX = np.abs(normalizedComplexX)
        elif self.nChannels == 8:
            dataReal = normalizedComplexX.real
            dataImag = normalizedComplexX.imag
            normalizedX = np.concatenate([dataReal,dataImag],axis=-1)
        else:
            raise Exception('Invalid number of channels')

        if self.mode is None:
            raise Exception('Mode is not set')
        
        if self.mode == 'train' or self.mode == 'val':
            batch_global, batch_local = [], []
            for image in normalizedX:
                globalCrops, localCrops = [], []
                for crop in range(self.nGlobalCrops+ self.nLocalCrops):
                    augmentedImage = image.copy()
                    if self.mode == 'train':
                        if self.augmentFlip:
                            augmentedImage = self._augmentFlip(augmentedImage)
                        if self.augmentColor:
                            augmentedImage = self._augmentColor(augmentedImage)

                    if crop < self.nGlobalCrops:
                        augmentedImage = self._augmentCropGlobal(augmentedImage)
                        globalCrops.append(augmentedImage)
                    else:
                        augmentedImage = self._augmentCropLocal(augmentedImage)
                        localCrops.append(augmentedImage)

                batch_local.append(localCrops)
                batch_global.append(globalCrops)
            return batch_global, batch_local
        elif self.mode == 'original' or self.mode == 'test':
            return normalizedX
        else:
            raise Exception('Invalid mode')

class Generators():
    class UniversalDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, h5SetsLocation, modelType, mode, nChannels, samples_file = None, dinoSettings=None, ganParameters=None, dataSettings=None, bufferAll = False, cacheDataset = False, nSamples = None, returnDataY=False, offsetSamples = None):
            
            self.h5SetsLocation = h5SetsLocation
            self.modelType = modelType
            self.mode = mode
            self.nChannels = nChannels
            self.returnDataY=returnDataY
            
            self.dataSettings = dataSettings

            self.batch_size = dataSettings['batchSize']
            self.normalizationMethod = dataSettings['normalizationMethod']
            self.augmentation = dataSettings['augmentation']
            
            self.nSamples = nSamples
            self.offsetSamples = offsetSamples # can only be used when nSamples is not None. Then it does not start at 0 but at the offset
            self.dataX = None
            self.loadMetadata = False
            self.cacheDataset = None
            self.samplesHash = None
            self.dinoEmbedding = None
            self.styleInput = None
            self.classLabels = None
            self.classValIndices = None
            self.classTrainIndices = None

            if self.modelType == 'dino':
                self.sampleDim = dinoSettings['sampleDim']
                if self.sampleDim is not None:
                    raise Exception("Sample dim is not implemented in UniversalDataGenerator yet.")
                self.dinoSettings = dinoSettings
                localScale = dinoSettings['localScale']
                globalScale = dinoSettings['globalScale']
                self.loadMetadata = (dinoSettings["customPositionalEncoding"] is not None or dinoSettings["metadataEncoding"] is not None)
                self.nSubbands = dataSettings['nSubbands']               

                self.preprocess = PreprocessDino(globalScale,localScale, self.dinoSettings['nLocalCrops'], self.dinoSettings['teacherGlobalSize'], self.dinoSettings['studentLocalSize'], nChannels, self.normalizationMethod, mode, self.augmentation )
                self.dataLoader = self.loadDino
            elif self.modelType == 'gan':
                self.dataLoader = self.loadGan
            elif self.modelType == 'gan_ss':
                self.pretext = PretextGenerator(ganParameters['pretextTask'], self.normalizationMethod)
                self.dataLoader = self.loadGanSs
            elif self.modelType == 'dimensionReduction':
                self.dataLoader = self.loadDimensionReduction

            if samples_file is None:
                self.loadh5(h5SetsLocation)
            else:
                self.loadSamplesFile(samples_file, bufferAll, cacheDataset)

        def loadSamplesFile(self, samples_file, bufferAll, cacheDataset):
            print()
            print('Load samples: file {}'.format(samples_file))
            # Read the samples file
            if isinstance(samples_file, str):
                samplesFile = pickle.load(open(samples_file, "rb"))
                if os.environ.get('OS','') == "Windows_NT":
                    cacheLocation = os.path.dirname(samples_file)
                else:
                    cacheLocation =  './tempDatasets/LOFAR_L2014581 (recording)/preprocessed/dataset250k'
            else:
                samplesFile = samples_file

            # If the samples file contain the frequency mapping, frequency is random. In that case use it
            self.samples = samplesFile[0]
            if len(samplesFile) > 1:
                self.frequencyMap = samplesFile[1]
            else:
                self.frequencyMap = None
            
            # Retrieve the samples with one or more subbands from the data if desired
            if self.dataSettings['subbands'] is not None:
                if self.frequencyMap is not None:
                    raise Exception("Overlapping subbands are not supported with frequency mapping.")
                if self.dataSettings['subbands'] == 'habrok':
                    print("Load samples: extract samples which are available on Habrok")
                    # Get all filenames in h5SetsLocation
                    availableSubbands = os.listdir(self.h5SetsLocation)
                    self.samples = [sample for sample in self.samples if sample[0] in availableSubbands]
                else:
                    print("Load samples: extract the samples of the requested subbands")
                    subbands = [self.dataSettings['subbandPrefix']+str(subband)+self.dataSettings['subbandPostfix'] for subband in self.dataSettings['subbands']]
                    self.samples = [sample for sample in self.samples if sample[0] in subbands]

            # Limit the number of samples if desired
            if self.offsetSamples is not None and self.nSamples is None:
                raise Exception("OffsetSamples can only be used when nSamples is not None.")
            
            if self.nSamples is not None:
                print("Load samples: limit the number of samples")
                np.random.seed(0)
                np.random.shuffle(self.samples)
                if self.offsetSamples is None:
                    startIdx = 0
                else:
                    startIdx = self.offsetSamples
                self.samples = self.samples[startIdx:startIdx+self.nSamples]

            print("Load samples: Number of samples: {}".format(len(self.samples)))

            # Create the indices list for sampling
            self.indices = np.arange(len(self.samples))
            if len(self.indices)>45000 and bufferAll:
                print("Load samples: more than 45000 samples. Buffering is disabled.")
                bufferAll = False

            # Always create a hash such that the samples are verifiable in the pipeline
            # Hash the samples data, and remove all non-alphanumeric characters from the hash so it can be used as a filename
            if self.loadMetadata: # Chached files without metadata are not loaded again
                str_data = str(self.samples) + "_metadata"
            else:
                str_data = str(self.samples)
            hash_object = hashlib.md5()
            hash_object.update(str_data.encode('utf-8'))
            self.samplesHash = base64.b64encode(hash_object.hexdigest().encode('utf-8')).decode('utf-8')
            self.samplesHash = ''.join(e for e in self.samplesHash if e.isalnum())

            # If the data has to be buffered, use cach to store the data. This makes it possible to load the data faster
            if bufferAll or cacheDataset:
                self.cacheFilename = os.path.join(cacheLocation, self.samplesHash+'.h5')   
                if os.path.exists(self.cacheFilename):
                    self.cacheDataset = h5py.File(self.cacheFilename, 'r')
                    
                    if bufferAll:
                        self.dataX = self.cacheDataset['/dataX'][:]
                    else:
                        self.dataX = self.cacheDataset['/dataX']
                    self.dataY = self.cacheDataset['/dataY']
                    
                    if self.loadMetadata:
                        self.metadata = self.cacheDataset['/metadata']
                else:
                    if cacheDataset:
                        print("Load samples: cache file does not exist, create: "+self.cacheFilename)
                    else:
                        print("Load samples: cache file does not exist, buffering: "+self.cacheFilename)
                    names,positions,frequencies,times,uvws,timeStartStep,self.dataX,self.dataY,setMetadata = utils.functions.sampleFromH5(self.h5SetsLocation, self.samples, self.frequencyMap, verbose=True, standardize=False)
                    self.metadata = list(zip(frequencies, timeStartStep))

                    if cacheDataset:
                        cacheSet = h5py.File(self.cacheFilename, 'w')
                        cacheSet.create_dataset('dataX',data=self.dataX)
                        cacheSet.create_dataset('dataY',data=self.dataY)
                        if self.loadMetadata:
                            cacheSet.create_dataset('metadata',data=self.metadata)
                        cacheSet.close()
                        self.cacheDataset = h5py.File(self.cacheFilename, 'r')
                    else:
                        self.cacheDataset = {}
                        self.cacheDataset['/dataX'] = self.dataX
                        self.cacheDataset['/dataY'] = self.dataY
                        if self.loadMetadata:
                            self.cacheDataset['/metadata'] = self.metadata

        def loadh5(self, h5SetsLocation):
            if self.loadMetadata:
                raise Exception("Load metadata is not implemented for h5 files.")
            
            # Open the HDF5 file
            self.h5f = h5py.File(h5SetsLocation, 'r')
            indices = np.arange(len(self.h5f['/dataX/names']))
            self.dataX = self.h5f['/dataX/observations'][indices]
            self.dataY = self.h5f['/dataY'][indices].astype(int)

            self.indices = np.arange(len(self.dataX))

        def loadIndices(self, sampleIndices, loadY = True):
            dataY=None
            batchDinoFeatures = None
            if self.dinoEmbedding is not None:
                batchDinoFeatures = self.dinoEmbedding[sampleIndices]

            if self.dataX is None:
                if self.cacheDataset is not None:
                    dataX = self.cacheDataset['/dataX'][sampleIndices]
                    if loadY:
                        dataY = self.cacheDataset['/dataY'][sampleIndices]

                    if batchDinoFeatures is not None:
                        dataX = [dataX, batchDinoFeatures]
                    
                    if self.loadMetadata:
                        metadata = self.cacheDataset['/metadata'][sampleIndices]
                        return dataX, dataY, metadata
                    else:
                        return dataX, dataY
                else:
                    sampleList = [self.samples[index] for index in sampleIndices]
                    if loadY:
                        dataX,dataY = utils.functions.sampleFromH5(self.h5SetsLocation, sampleList, self.frequencyMap, verbose=False, standardize=False, loadComponents=['observations','labels'])
                    else:
                        dataX = utils.functions.sampleFromH5(self.h5SetsLocation, sampleList, self.frequencyMap, verbose=False, standardize=False, loadComponents=['observations'])[0]
                    if batchDinoFeatures is not None:
                        dataX = [dataX, batchDinoFeatures]
                    return dataX, dataY
            else:
                dataX = self.dataX[sampleIndices]
                if batchDinoFeatures is not None:
                        dataX = [dataX, batchDinoFeatures]
                if loadY:
                    dataY = self.dataY[sampleIndices]

                if self.loadMetadata:
                    metadata = self.metadata[sampleIndices]
                    return dataX, dataY, metadata
                else:
                    return dataX, dataY

        def AppendDinoFeatures(self, features):
            self.dinoEmbedding = features

        def setMetadataStyleInput(self, metadataType = None):
            if metadataType is None:
                self.styleInput = None
                return
            self.styleInput = []
            if metadataType == 'subband':
                for sample in self.samples:
                    subbandName = sample[0]
                    # Define the regex pattern to match the number
                    pattern = r"SB(\d+)_"
                    match = re.search(pattern, subbandName)
                    if match:
                        number = int(match.group(1))
                        self.styleInput.append(number)

                    else:
                        raise Exception("Subband number not found in the subband name.")          
                self.styleInput -= np.min(self.styleInput)
                self.nStyleSubbands = np.max(self.styleInput)
            return self.nStyleSubbands

        @staticmethod
        def timeToEmbedding(timeSteps):
            return timeSteps
        
        def freqToEmbedding(self, freqSteps):
            fZero = self.dataSettings['startFreq']# 130.56640625
            subbandWidth = self.dataSettings['freqStepSize'] #0.1953125
            maxTimeSteps = 2400/self.dinoSettings['patchSize']

            # Calculate how much the step size must be to make sure time and frequency are uniquely encoded
            nRowPatches = self.dinoSettings['teacherGlobalSize'][0]/self.dinoSettings['patchSize']
            subbandStepSize = maxTimeSteps*nRowPatches
            subbandIndices = (freqSteps - fZero)/subbandWidth
            freqEncoding = subbandIndices*subbandStepSize
            return freqEncoding     

        def encodingToEmbedding(self,embedding, hiddenSize):
            if 'bin' in self.dinoSettings["customPositionalEncoding"]:
                # Each value in embedding must be presented as a binary number
                n=10000
                posEmbedding = np.tile(embedding[:,:,np.newaxis], (hiddenSize)).astype(np.float32)
                halfDepth = np.tile(embedding[:,:,np.newaxis], (int(hiddenSize/2))).astype(np.float32)
                i = np.arange(int(hiddenSize/2))
                fraction = halfDepth/np.power(n, 2*i/hiddenSize)
                posEmbedding[:, :, ::2] = np.sin(fraction)
                posEmbedding[:, :, 1::2] = np.cos(fraction)
            else:
                n=10000
                posEmbedding = np.tile(embedding[:,:,np.newaxis], (hiddenSize))
                halfDepth = np.tile(embedding[:,:,np.newaxis], (int(hiddenSize/2)))
                i = np.arange(int(hiddenSize/2))
                fraction = halfDepth/np.power(n, 2*i/hiddenSize)
                posEmbedding[:, :, ::2] = np.sin(fraction)
                posEmbedding[:, :, 1::2] = np.cos(fraction)
            return posEmbedding
        
        def cropMetadataToEmbedding(self, metadata, cropSize):     
            nSamples = metadata.shape[0]
            rows = cropSize[0]
            columns = cropSize[1]

            fStart = metadata[:,0]
            tStart = metadata[:,1]

            fChannelIndex = ((fStart-self.dataSettings['startFreq'])/self.dataSettings['freqStepSize'])*64

            tPatchIndex = (tStart/self.dinoSettings['patchSize']).astype(np.int32)
            fPatchIndex = (fChannelIndex/self.dinoSettings['patchSize']).astype(np.int32)

            nTotalTimePatches = int(2400/self.dinoSettings['patchSize'])
            
            nColumnPatches = int(columns/self.dinoSettings['patchSize'])
            nRowPatches = int(rows/self.dinoSettings['patchSize'])

            patchesPerGlobalImage = int((256/self.dinoSettings['patchSize'])*(64/self.dinoSettings['patchSize']))

            relEncoding = np.arange(0,nColumnPatches*nRowPatches,dtype=np.int32)
            batchRelEncoding = np.tile(relEncoding, (nSamples, 1))
            batchAbsEncoding = batchRelEncoding + tPatchIndex*patchesPerGlobalImage + (fPatchIndex*nTotalTimePatches)*patchesPerGlobalImage

            batchEncoding = batchAbsEncoding
            clsEncoding = batchEncoding[:,0]
            batchEncodingWithCls = np.insert(batchEncoding,0,clsEncoding,axis=1)
            raise Exception("Hidden size not implemented ")
            posEmbedding = self.encodingToEmbedding(batchEncodingWithCls, self.hiddenSize)
            return posEmbedding
            
        def cropMetadataEncoding(self, metadata, cropSize):
            encodingType = self.dinoSettings['metadataEncoding']
            if encodingType == 'f_nn':
                freqMetadata = metadata[:,0]
                freqOnehot = tf.one_hot(freqMetadata,depth=self.nSubbands)
                return freqOnehot

        def ConvertToImage(self, dataX, std=2, calcMagnitude=True, calcPhase = False):
            if isinstance(dataX,tf.Tensor):
                dataX = dataX.numpy()
            elif isinstance(dataX,list):
                dataX = np.asarray(dataX)

            removeFirstAxis = False
            if len(dataX.shape)==3:
                dataX = np.expand_dims(dataX, axis=0)
                removeFirstAxis=True

            if dataX.shape[-1] == 8:
                imag = dataX[:,:,:,0:4]
                real = dataX[:,:,:,4:]
                magnitudeX = np.sqrt(imag**2+real**2)
                phaseX = np.arctan2(imag,real)
            elif dataX.dtype == np.complex64:
                magnitudeX = np.abs(dataX)
                phaseX = np.angle(dataX)
            else:
                magnitudeX = dataX
            
            if calcMagnitude:
                # Clip each value and normalize to 0-1
                if np.min(magnitudeX)<0:
                    magnitudeX = np.clip(magnitudeX,-std,std)
                    magnitudeX = (magnitudeX+std)/(2*std)
                    warnings.warn("Negative valued data plotted to an image. Data from dataset should be amplitudes")
                else:
                    magnitudeX = np.clip(magnitudeX,0,std)
                    magnitudeX = magnitudeX/(std)

                # Convert to color
                if magnitudeX.shape[-1] == 3:
                    magnitudeImage = magnitudeX
                elif magnitudeX.shape[-1] == 4:
                    magnitudeImage = np.zeros((magnitudeX.shape[0],magnitudeX.shape[1],magnitudeX.shape[2],3),dtype=np.float32)
                    magnitudeImage[:,:,:,0] = magnitudeX[:,:,:,0]
                    magnitudeImage[:,:,:,1] = 0.5*magnitudeX[:,:,:,1] + 0.5*magnitudeX[:,:,:,2]
                    magnitudeImage[:,:,:,2] = magnitudeX[:,:,:,3]

                if removeFirstAxis:
                    magnitudeImage = magnitudeImage[0,...]

            if calcPhase:
                phaseX = (phaseX+np.pi)/(2*np.pi)

                differenceOne = (phaseX[:,:,:,0]-phaseX[:,:,:,1])**2
                differenceFour = (phaseX[:,:,:,1]-phaseX[:,:,:,2])**2
                differenceSix = (phaseX[:,:,:,2]-phaseX[:,:,:,3])**2
                errorOne = np.sqrt(differenceOne)
                errorTwo = np.sqrt(differenceFour)
                errorThree = np.sqrt(differenceSix)
                phaseX = np.stack([errorOne,errorTwo,errorThree],axis=-1)

                if phaseX.shape[-1] == 4:
                    phaseImage = np.zeros((phaseX.shape[0],phaseX.shape[1],phaseX.shape[2],3),dtype=np.float32)
                    phaseImage[:,:,:,0] = phaseX[:,:,:,0]
                    phaseImage[:,:,:,1] = 0.5*phaseX[:,:,:,1] + 0.5*phaseX[:,:,:,2]
                    phaseImage[:,:,:,2] = phaseX[:,:,:,3]
                elif phaseX.shape[-1] == 3:
                    phaseImage = phaseX
                elif len(phaseX.shape) == 3:
                    phaseImage = np.zeros((phaseX.shape[0],phaseX.shape[1],phaseX.shape[2],3),dtype=np.float32)
                    phaseImage[:,:,:,0] = phaseX
                    phaseImage[:,:,:,1] = phaseX
                    phaseImage[:,:,:,2] = phaseX
                else:
                    raise Exception("Phase can only be calculated for complex-valued data")

                if removeFirstAxis:
                    phaseImage = phaseImage[0,...]

            if calcMagnitude and calcPhase:
                return magnitudeImage, phaseImage
            elif calcPhase:
                return phaseImage
            elif calcMagnitude:
                return magnitudeImage
            else:
                raise Exception("No data to return")

        def loadDino(self, sampleIndices):
            if self.loadMetadata:
                dataX, dataY, metadata = self.loadIndices(sampleIndices, loadY=self.returnDataY)
            else:
                dataX, dataY = self.loadIndices(sampleIndices, loadY=self.returnDataY)

            preprocessedX = self.preprocess(dataX)
            if self.returnDataY:
                return preprocessedX, dataY
            else:
                return preprocessedX

        def loadGan(self, sampleIndices):
            if self.dataX is None:
                sampleList = [self.samples[index] for index in sampleIndices]
                dataX,dataY = utils.functions.sampleFromH5(self.h5SetsLocation, sampleList, self.frequencyMap, standardize=False, loadComponents = ['observations','labels'])
            else:
                dataX = self.dataX[sampleIndices]
                dataY = self.dataY[sampleIndices]

            normalizedComplexX = NormalizeComplex(dataX, self.normalizationMethod)

            if self.nChannels == 4:
                normalizedX = np.abs(normalizedComplexX)
            else:
                normalizedX = np.concatenate([normalizedComplexX.real,normalizedComplexX.imag],axis=-1)
            
            dataY = tf.one_hot(dataY,depth=2)
            if self.styleInput is not None:
                batchStyleInput = self.styleInput[sampleIndices]
                batchStyleInput = tf.cast(tf.one_hot(batchStyleInput,depth=self.nStyleSubbands), tf.float32)
                normalizedX = [normalizedX, batchStyleInput]

            result = (normalizedX, dataY)
            if self.mode=='original':
                raise Exception("Not implemented anymore. Use normalizedX and convert it to color")
            return result
      
        def loadGanSs(self, sampleIndices):
            if self.dataX is None:
                sampleList = [self.samples[index] for index in sampleIndices]
                names,positions,frequencies,times,uvws,sinTimes,dataX,dataY,setMetadata = utils.functions.sampleFromH5(self.h5SetsLocation, sampleList, self.frequencyMap, verbose=False, standardize=False)
            else:
                dataX = self.dataX[sampleIndices]

            pretextX, pretextY = self.pretext(dataX)
            return pretextX, pretextY    

        def getLabels(self):
            if self.dataX is None:
                sampleList = self.samples
                names,positions,frequencies,times,uvws,sinTimes,scaledX,dataY,setMetadata = utils.functions.sampleFromH5(self.h5SetsLocation, sampleList, self.frequencyMap, verbose=False, standardize=True)
            else:
                dataY = self.dataY

            return dataY
        
        # For classification task with external labels
        def setClassLabels(self,labels):
            self.classLabels = labels

        def makeTrainValSplit(self, nValSamples):
            nTrainSamples = len(self.classLabels)-nValSamples
            classes, samplesPerClass = np.unique(self.classLabels, return_counts=True)
            classMostSamples = classes[np.argmax(samplesPerClass)]

            valFactor = nValSamples/len(self.classLabels)
            valIndices = []
            trainIndices = []
            for classNumber in classes:
                if classNumber == classMostSamples: continue
                classIndices = np.where(self.classLabels == classNumber)[0]
                nValSamplesClass = int(round(valFactor*len(classIndices)))
                classValIndices = np.random.choice(classIndices, nValSamplesClass, replace=False)
                classTrainIndices = np.setdiff1d(classIndices,classValIndices)
                valIndices.extend(classValIndices)  
                trainIndices.extend(classTrainIndices)
            
            # Fill the last samples by the samples of the largest cluster
            nValSamplesLargestCluster = nValSamples-len(valIndices)
            classIndices = np.where(self.classLabels == classMostSamples)[0]
            classValIndices = np.random.choice(classIndices, nValSamplesLargestCluster, replace=False)
            classTrainIndices = np.setdiff1d(classIndices, classValIndices)
            valIndices.extend(classValIndices)
            trainIndices.extend(classTrainIndices)
            self.classValIndices = np.asarray(valIndices)
            self.classTrainIndices = np.asarray(trainIndices)

        def getClassTrainValSamples(self, subset):
            if subset == 'train':
                data = self.dataLoader(self.classTrainIndices)
                labels = self.classLabels[self.classTrainIndices]
            elif subset == 'val':
                data = self.dataLoader(self.classValIndices)
                labels = self.classLabels[self.classValIndices]
            else:
                raise Exception("Subset not recognized")
            return data, labels

        def getSamplesFile(self):
            return self.samples
        
        def loadDimensionReduction(self, sampleIndices):
            batchX, batchY = self.loadIndices(sampleIndices, loadY=True)
            normalizedComplexX = NormalizeComplex(batchX, self.normalizationMethod)

            if self.nChannels == 4:
                normalizedX = np.abs(normalizedComplexX)
            elif self.nChannels == 8:
                dataReal = normalizedComplexX.real
                dataImag = normalizedComplexX.imag
                normalizedX = np.concatenate([dataReal,dataImag],axis=-1)
            else:
                raise Exception('Invalid number of channels')
            return normalizedX, batchY

        def getSsDataShape(self, pretextTask):
            if self.modelType == 'gan_ss':
                dummyPretext = self.pretext
            else:
                dummyPretext = PretextGenerator(pretextTask, 0)
            return dummyPretext.getDataShape()

        def getDataShape(self):
            if self.modelType == 'gan':
                return self.dataSettings['inputShape'], self.dataSettings['outputShape']
            elif self.modelType == 'gan_ss':
                return self.pretext.getDataShape()
            else:
                raise Exception("Not implemented")

        def getMetadata(self, sampleIndices = None):
            if sampleIndices is None:
                sampleList = self.samples
            else:
                sampleList = [self.samples[index] for index in sampleIndices]

            # loadNames = 'names' in loadComponents
            # loadPositions = 'positions' in loadComponents
            # loadFrequencies = 'frequencies' in loadComponents
            # loadTimes = 'times' in loadComponents
            # loadUvws = 'uvws' in loadComponents
            # loadTimeStartSteps = 'timeStartStep' in loadComponents
            # loadObservations = 'observations' in loadComponents
            # loadLabels = 'labels' in loadComponents
            # loadMetadata = 'setMetadata' in loadComponents
            loadComponents = ['names', 'positions', 'frequencies', 'times']

            names, positions, frequencies, times = utils.functions.sampleFromH5(self.h5SetsLocation, sampleList, self.frequencyMap, verbose=False, standardize=False, loadComponents=loadComponents)
            names = np.asarray(names)
            metadata = [names,positions, frequencies,times]
            return metadata

        def __len__(self):
            return int(np.floor(len(self.indices) / self.batch_size))

        def __getitem__(self, index):
            startIndex = index*self.batch_size
            batchIndices = sorted(self.indices[startIndex:startIndex+self.batch_size])
            data = self.dataLoader(batchIndices)           
            return data

        def on_epoch_end(self):
            np.random.shuffle(self.indices)

    class FakeDinoGenerator(tf.keras.utils.Sequence):
        def __init__(self, batchSize, nChannels, globalSize, localSize, nGlobalCrops, nLocalCrops, nBatches):
            self.nBatches = nBatches
            self.batchSize = batchSize
            self.nChannels = nChannels
            self.globalSize = globalSize
            self.localSize = localSize
            self.nGlobalCrops = nGlobalCrops
            self.nLocalCrops = nLocalCrops
            
        def __len__(self):
            return self.nBatches

        def __getitem__(self, index):
            fakeGlobalImage = np.random.normal(size=(self.globalSize[0],self.globalSize[1], self.nChannels))
            fakeLocalImage = np.random.normal(size=(self.localSize[0],self.localSize[1], self.nChannels))
                
            localBatch = []
            globalBatch = []
            for sampleIdx in range(self.batchSize):
                localSamples = []
                globalSamples = []
                for localIdx in range(self.nLocalCrops):
                    localSamples.append(fakeLocalImage)
                
                for globalIdx in range(2):
                    globalSamples.append(fakeGlobalImage)
            
                localBatch.append(localSamples)
                globalBatch.append(globalSamples)
            fakeBatch = [globalBatch,localBatch]
            return fakeBatch

# Functions on detecting RFI/statistics
def calcRfiTypesPerSample(dataY, returnDebugImages = False): #debugPlotLoc=None):
        dataY = dataY[:,1:,:]
        
        dataRfiY = []
        dataRfiX = []
        dataRfiWeak = []
        cumRfiRatio = 0.5 # the higher, the more rfi in both x and y is allowed
        maxRfiLineLength = 10
        for imageIdx, image in enumerate(dataY): 
            rfiLineLength = 4
            while True:
                # Make kernels
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rfiLineLength))
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rfiLineLength, 1))

                # Apply morphology operations
                temp_img = cv2.erode(image, vertical_kernel, iterations=3)
                rfiY = cv2.dilate(temp_img, vertical_kernel, iterations=3)
                temp_img = cv2.erode(image, horizontal_kernel, iterations=3)
                rfiX = cv2.dilate(temp_img, horizontal_kernel, iterations=3)

                if rfiLineLength >= maxRfiLineLength:
                    break
                
                xAndY = np.logical_and(rfiY, rfiX)
                sumX = np.sum(rfiX)
                sumY = np.sum(rfiY)
                sumXAndY = np.sum(xAndY)
                ratioX = 1
                ratioY = 1
                if sumX > 0:
                    ratioX = sumXAndY/sumX
                if sumY > 0:
                    ratioY = sumXAndY/sumY
                if ratioX > cumRfiRatio or ratioY > cumRfiRatio:
                    rfiLineLength += 1
                else:
                    break
            rfiX = np.logical_and(rfiX, image).astype(np.uint8)
            rfiY = np.logical_and(rfiY, image).astype(np.uint8)
            weakRfi = np.logical_and(image,np.logical_not(rfiX))
            weakRfi = np.logical_and(weakRfi,np.logical_not(rfiY)).astype(np.uint8)
            dataRfiX.append(rfiX)
            dataRfiY.append(rfiY)
            dataRfiWeak.append(weakRfi)

        dataRfiX = np.asarray(dataRfiX)
        dataRfiY = np.asarray(dataRfiY)
        dataRfiWeak = np.asarray(dataRfiWeak)

        # Calculate the total RFI, and the ratios of RFI wrt total RFI per sample
        totalRfi = np.sum(dataY,axis=(1,2),dtype=np.int32)
        totalRfiX = np.sum(dataRfiX,axis=(1,2),dtype=np.int32)
        totalRfiY = np.sum(dataRfiY,axis=(1,2),dtype=np.int32)
        totalMaxRfi = np.max(totalRfi)
        totalWeakRfi = np.sum(dataRfiWeak,axis=(1,2),dtype=np.int32) # totalRfi-totalRfiX-totalRfiY

        ratioTotalRfi = totalRfi/totalMaxRfi
        ratioRfiX = totalRfiX/totalRfi
        ratioRfiY = totalRfiY/totalRfi
        ratioWeakRfi = totalWeakRfi/totalRfi

        rfiResults = np.stack((totalRfi,totalWeakRfi,totalRfiY, totalRfiX),axis=1)
        rfiRatioResults = np.stack((ratioTotalRfi,ratioWeakRfi, ratioRfiY, ratioRfiX),axis=1)
        rfiCategories = ['Total amount of RFI','Weak RFI','RFI local in time', 'RFI local in frequency']#,'Single RFI']

        if returnDebugImages:
            return rfiResults, rfiRatioResults, rfiCategories, dataY, dataRfiWeak, dataRfiY, dataRfiX

        return rfiResults, rfiRatioResults, rfiCategories