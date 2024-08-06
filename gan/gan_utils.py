import os
import datetime
import math
import re

import numpy as np
import tensorflow as tf

import utils as utils

class Generator(tf.keras.Model):   
    def __init__(self, inputShape, outputShape, dataSettings, ganModelSettings, dinoModelSettings = None, useDino = False, trainSS = False, metadataStyleVectorSize = None):
        super(Generator, self).__init__()
        zeroInputBatch = np.zeros((dataSettings['batchSize'], inputShape[0], inputShape[1], inputShape[2]))
        self.trainEncoderInInferenceMode = False # When true, batch normalization will not be used while training
        self.ganModelSettings=ganModelSettings
        self.trainSS = trainSS
        self.batchSize = dataSettings['batchSize']
        self.enableSkipConnections = ganModelSettings['skipsEnabled']
        self.metadataSyleInput = ganModelSettings['metadataStyleInput']
        self.metadataStyleVectorSize = metadataStyleVectorSize

        self.encoderDepth = ganModelSettings['generatorDepth']
        self.decoderDepth = ganModelSettings['generatorDepth']
        self.decoderInputShape = (int(inputShape[0]/2**(self.decoderDepth)),int(inputShape[1]/2**(self.decoderDepth)),64*2**(self.decoderDepth))


        self.encoder = utils.models.Models.Gan_Encoder_v1(inputShape,dropoutRate = ganModelSettings['dropoutRate'],depth=self.encoderDepth, 
                                                          modConv = ganModelSettings['modConvEncoder'],
                                                          batchSize=dataSettings['batchSize'], 
                                                          modConvActivation=ganModelSettings['modulatedConvActivation'],
                                                          modConvResDownNormAdd=ganModelSettings['modConvResUpNormAdd'],
                                                          nFeaturesW=ganModelSettings['nFeaturesW']
                                                          )

        self.nEncoderTiles = 0
        self.nDecoderTiles = 0
        if ganModelSettings['modConvEncoder']:
            self.nEncoderTiles = self.encoder.nLatentWTiles
            self.nDecoderTiles = 21
            nLatentWTiles = self.nEncoderTiles+self.nDecoderTiles
            #self.mappingNetworkEnc = utils.models.layers.DinoMappingNetwork(nFeaturesW=ganModelSettings['nFeaturesW'],dlatent_broadcast=self.nEncoderTiles, nLayers=nMappingLayers, mapping_lrmul = ganModelSettings['styleMappingLrMul'], activation=ganModelSettings['modulatedConvActivation'], name='External_dinoMappingNetwork_Enc')
            #self.mappingNetworkDec = utils.models.layers.DinoMappingNetwork(nFeaturesW=ganModelSettings['nFeaturesW'],dlatent_broadcast=self.nDecoderTiles, nLayers=nMappingLayers, mapping_lrmul = ganModelSettings['styleMappingLrMul'], activation=ganModelSettings['modulatedConvActivation'], name='External_dinoMappingNetwork_Dec')
            self.mappingNetwork = utils.models.layers.DinoMappingNetwork(nFeaturesW=ganModelSettings['nFeaturesW'],dlatent_broadcast=nLatentWTiles, nLayers=ganModelSettings['nMappingLayers'], mapping_lrmul = ganModelSettings['styleMappingLrMul'], activation=ganModelSettings['modulatedConvActivation'], normalizeInput=ganModelSettings['styleMappingNormInput'], name='External_dinoMappingNetwork')
            
            zeroMappingBatch = np.zeros((dataSettings['batchSize'],self.nEncoderTiles,ganModelSettings['nFeaturesW']))
            zeroEncoderIn = [zeroInputBatch,zeroMappingBatch]
        else:
            zeroEncoderIn = zeroInputBatch
        
        randomEmbedding, randomContractingLayers = self.encoder(zeroEncoderIn)

        self.concatDinoEncoder = ganModelSettings['concatDino']
        self.flattenFeaturesVector = ganModelSettings['modConv']
        self.cachedDinoEmbedding = False

        self.maskStyleInput = ganModelSettings['maskStyleInput']
        
        nFeaturesZ = None
        if ganModelSettings['concatDino'] or ganModelSettings['modConv']:
            self.dinoEncoder = utils.models.Models.Gan_Encoder_v1(inputShape,depth=self.encoderDepth)
            if ganModelSettings['modConv']:
                if self.metadataSyleInput is None:
                    nFeaturesZ = dinoModelSettings['reducedGanDimension']
                    encoderOutputDimension = randomEmbedding.shape[1]*randomEmbedding.shape[2]*randomEmbedding.shape[3] # TODO: Get this from the encoder, randomEmbedding
                    nFiltersReductionLayer = round((nFeaturesZ/encoderOutputDimension)*randomEmbedding.shape[-1])
                    self.featuresReductionLayer = tf.keras.layers.Conv2D(filters=nFiltersReductionLayer,kernel_size=1,strides=1,padding="valid")
                    self.flattenLayer = tf.keras.layers.Flatten()
                else:
                    nFeaturesZ = int(self.metadataStyleVectorSize)

        self.decoder = utils.models.Models.Gan_DecoderMask_v1(self.decoderInputShape, randomContractingLayers,outputShape[2],dataSettings['batchSize'], 
                                                              dropoutRate = ganModelSettings['dropoutRate'],
                                                              depth=self.decoderDepth, 
                                                              selfSupervised=self.trainSS, 
                                                              concatDinoEncoder=self.concatDinoEncoder, 
                                                              nFeaturesZ=nFeaturesZ, 
                                                              nFeaturesW=ganModelSettings['nFeaturesW'],
                                                              modConv=ganModelSettings['modConv'], 
                                                              modConvActivation=ganModelSettings['modulatedConvActivation'],
                                                              modConvResUpNormAdd=ganModelSettings['modConvResUpNormAdd'], 
                                                              styleMappingLrMul = ganModelSettings['styleMappingLrMul'],
                                                              nMappingLayers = ganModelSettings['nMappingLayers'],
                                                              styleMappingNormInput = ganModelSettings['styleMappingNormInput'],
                                                              externalMappingNetwork = ganModelSettings['modConvEncoder'],
                                                              )


        if self.enableSkipConnections==False:
            self.contractingLayersAsZeros = {}
            for layerKey in randomContractingLayers.keys():
                self.contractingLayersAsZeros[layerKey] = tf.zeros_like(randomContractingLayers[layerKey])
    
    def build(self, input_shape):
        pass

    def call(self, inputs):
        if self.metadataSyleInput is not None:
            [inputs, styleInputs] = inputs
        
        # Infer the latent Z space from the DINO encoder, and if required, mix it to a style
        if self.concatDinoEncoder or self.flattenFeaturesVector:
            if self.concatDinoEncoder or self.maskStyleInput==False:
                dinoEmbedding, _ = self.dinoEncoder(inputs)

            
            if self.flattenFeaturesVector:
                if self.maskStyleInput: # To test the effect of the style input. Will prevent the network from learning the style
                    # make a random noise vector of the same size as the latentZ
                    flattenSize = 320 # latentZ.shape[1] # TODO: fix this
                    latentZ = tf.random.truncated_normal((self.batchSize,flattenSize))
                elif self.metadataSyleInput is None:
                    dinoFeatures = self.featuresReductionLayer(dinoEmbedding)
                    latentZ = self.flattenLayer(dinoFeatures)
                else:
                    latentZ = styleInputs

                
                if self.ganModelSettings['modConvEncoder']:
                    # latentEnc = self.mappingNetworkEnc(latentZ)
                    # latentDec = self.mappingNetworkDec(latentZ)
                    latentW = self.mappingNetwork(latentZ)
                    latentEnc = latentW[:,:self.nEncoderTiles,:]
                    latentDec = latentW[:,self.nEncoderTiles:,:]
                else:
                    latentDec = latentZ

        # Forward pass of the encoder
        if self.ganModelSettings['modConvEncoder']:
            encoderIn = (inputs,latentEnc)
        else:
            encoderIn = inputs
        if self.trainEncoderInInferenceMode:
            embedding, contractingLayersResult = self.encoder(encoderIn, training=False)
        else:
            embedding, contractingLayersResult = self.encoder(encoderIn)

        if self.enableSkipConnections==False:
            contractingLayersResult = self.contractingLayersAsZeros

        # Forward pass of the decoder
        if self.concatDinoEncoder and self.flattenFeaturesVector:
            #decoderIn = (embedding, contractingLayersResult, dinoEmbedding,latentZ[:,self.nEncoderTiles:,:])
            decoderIn = (embedding, contractingLayersResult, dinoEmbedding,latentDec)
        elif self.concatDinoEncoder:
            decoderIn = (embedding, contractingLayersResult, dinoEmbedding)
        elif self.flattenFeaturesVector:
            #decoderIn = (embedding, contractingLayersResult, latentZ[:,self.nEncoderTiles:,:])
            decoderIn = (embedding, contractingLayersResult, latentDec)
        else:
            decoderIn = (embedding,contractingLayersResult)

        # if self.concatDinoEncoder:
        #     dinoEmbedding, _ = self.dinoEncoder(inputs)
        #     rfiMask = self.decoder((embedding, contractingLayersResult, dinoEmbedding))
        # elif self.flattenFeaturesVector:
        #     dinoEmbedding, _ = self.dinoEncoder(inputs)
        #     dinoFeatures = self.featuresReductionLayer(dinoEmbedding)
        #     latentZ = self.flattenLayer(dinoFeatures)
        #     rfiMask = self.decoder((embedding, contractingLayersResult,latentZ))
        # else:
        #     rfiMask = self.decoder((embedding, contractingLayersResult))

        rfiMask = self.decoder(decoderIn)
        return rfiMask
    
class GANModel(tf.keras.Model):
    def __init__(self, dataSettings, ganModelSettings, dataShape=None, loadSS=False, dinoModelSettings=None,metadataStyleVectorSize=None):
        super().__init__()
        self.dataSettings = dataSettings
        self.ganModelSettings = ganModelSettings
        self.metadataStyleVectorSize = metadataStyleVectorSize

        # GAN
        if loadSS:
            nSamples = dataSettings['nSsTrainingSamples']
        else:
            nSamples = dataSettings['nTrainingSamples']
        
        if dataShape is None:
            self.inputShape = dataSettings['inputShape']
            self.outputShape = dataSettings['outputShape']
        else:
            self.inputShape = dataShape[0]
            self.outputShape = dataShape[1]

        batchSize = dataSettings['batchSize']
        nBatches = math.floor(nSamples/batchSize)

        lrStart = self.ganModelSettings['lrStart']
        lrDecay = (self.ganModelSettings['lrEnd'] / self.ganModelSettings['lrStart'])**(1/self.ganModelSettings['scheduledEpochs'])

        # Generator
        lrSchedulerGen = tf.keras.optimizers.schedules.ExponentialDecay(lrStart, decay_steps=nBatches*3, decay_rate=lrDecay, staircase=True)
        self.generator = Generator(self.inputShape, self.outputShape, dataSettings, ganModelSettings,dinoModelSettings,metadataStyleVectorSize=metadataStyleVectorSize)
        self.generator.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedulerGen))
       
        # Discriminator
        lrSchedulerDisc = tf.keras.optimizers.schedules.ExponentialDecay(lrStart, decay_steps=nBatches, decay_rate=lrDecay, staircase=True)
        discriminatorInputShape = (self.inputShape[0], self.inputShape[1], self.inputShape[2]+self.outputShape[2])
        self.discriminator = utils.models.Models.Discriminator_v1(discriminatorInputShape, self.ganModelSettings['batchNormMomentum'], self.ganModelSettings['eps'], self.ganModelSettings['dropoutRate'], ganModelSettings['lamb'])
        self.discriminator.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lrSchedulerDisc))

        # Loss and additional metrics        
        self.discriminatorLossTracker = tf.keras.metrics.Mean(name="disc_loss")
        self.maskLossTracker = tf.keras.metrics.Mean(name="gen_loss")
        self.trainBatchSize = batchSize

        if loadSS:
            self.compileMetrics=[utils.models.loss.ssim_loss]
        else:
            self.compileMetrics=[utils.models.metrics.accuracy, utils.models.metrics.precision, utils.models.metrics.recall]

        if metadataStyleVectorSize is not None:
            fakeSample = np.zeros((batchSize, self.inputShape[0], self.inputShape[1], self.inputShape[2]))
            fakeStyle = np.zeros((batchSize, metadataStyleVectorSize))
            fakeInput = [fakeSample,fakeStyle]
            _ = self.call(fakeInput)

        self.compile(metrics = self.compileMetrics)
        
    def compile(self, **kwargs):# metrics):
        super().compile(**kwargs)
        if self.metadataStyleVectorSize is None:
            self.build(input_shape = [self.trainBatchSize, self.inputShape[0],self.inputShape[1],self.inputShape[2]])
    
    def predict(self, observations):      
        predictedLabels = self.generator(observations, training=False)   
        return predictedLabels

    def train_step(self,xyPair):
        if self.metadataStyleVectorSize is not None:
            dataX = xyPair[0][0]
            metaX = xyPair[0][1]
            dataY = xyPair[1]
        else:
            dataX = tf.cast(xyPair[0],dtype=tf.float32)
            dataY = tf.cast(xyPair[1],tf.float32)

        with tf.GradientTape() as disc_tape:
            if self.metadataStyleVectorSize is None:
                predictedY = self.generator(dataX, training=True)
            else:
                predictedY = self.generator([dataX,metaX], training=True)
            
            # Make a true pair and predicted pair
            yNoise=tf.random.truncated_normal([self.trainBatchSize,self.dataSettings['outputShape'][0],self.dataSettings['outputShape'][1], predictedY.shape[-1]])
            truePair = tf.concat([dataX, tf.add(dataY,yNoise)], 3) 
            predictedPair = tf.concat([dataX, tf.add(predictedY,yNoise)], 3)

            predictedDiscriminator, predictedDiscriminatorLogits = self.discriminator(predictedPair, training=True)
            trueDiscriminator, trueDiscriminatorLogits = self.discriminator(truePair, training=True)
            
            # Calculate discriminator loss
            trueDiscriminatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(trueDiscriminator)*(1-self.ganModelSettings['smooth']),logits=trueDiscriminatorLogits))
            predictedDiscriminatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictedDiscriminator),logits=predictedDiscriminatorLogits))
            discriminatorLoss = trueDiscriminatorLoss + predictedDiscriminatorLoss
        discriminatorGradients = disc_tape.gradient(discriminatorLoss, self.discriminator.trainable_variables)
        self.discriminator.optimizer.apply_gradients(zip(discriminatorGradients, self.discriminator.trainable_variables))

        # Calculate and update generator gradients 
        for genUpdate in range(3):
            with tf.GradientTape() as gen_tape:
                if self.metadataStyleVectorSize is None:
                    predictedY = self.generator(dataX, training=True)
                else:
                    predictedY = self.generator([dataX,metaX], training=True)

                yNoise=tf.random.truncated_normal([self.trainBatchSize,self.dataSettings['outputShape'][0],self.dataSettings['outputShape'][1], predictedY.shape[-1]])
                predictedPair = tf.concat([dataX, tf.add(predictedY,yNoise)], 3)
                predictedDiscriminator, predictedDiscriminatorLogits = self.discriminator(predictedPair, training=True)
                maskLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predictedDiscriminator)*(1-self.ganModelSettings['smooth']), logits=predictedDiscriminatorLogits)) + self.ganModelSettings['L1_lambda'] * tf.reduce_mean(tf.abs(dataY - predictedY))

            # RFI Mask: Compute gradients separately for encoder and decoder
            generatorGradients = gen_tape.gradient(maskLoss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(generatorGradients, self.generator.trainable_variables))     

            # Update loss functions
            self.discriminatorLossTracker.update_state(discriminatorLoss)
            self.maskLossTracker.update_state(maskLoss)

        self.compiled_metrics.update_state(dataY, predictedY)

        # Merge the metrics and learning rates
        logNames = [m.name for m in self.metrics]
        logValues = [m.result() for m in self.metrics]

        # Return logs as dictionary
        return {m: n for m, n in zip(logNames, logValues)}

    def test_step(self, xyPair):
        if self.metadataStyleVectorSize is not None:
            dataX = xyPair[0][0]
            metaX = xyPair[0][1]
            dataY = xyPair[1]
        else:
            dataX = tf.cast(xyPair[0],dtype=tf.float32)
            dataY = tf.cast(xyPair[1],tf.float32)

        if self.metadataStyleVectorSize is None:
            predictedY = self.generator(dataX, training=False)
        else:
            predictedY = self.generator([dataX,metaX], training=False)

        # make pairs for discriminator
        truePair = tf.concat([dataX, dataY], axis=-1)
        predictedPair = tf.concat([dataX, predictedY], axis=-1)

        # Predict outputs of the discriminator and calculate the discriminator loss
        predictedDiscriminator, predictedDiscriminatorLogits = self.discriminator(predictedPair, training=True)
        trueDiscriminator, trueDiscriminatorLogits = self.discriminator(truePair, training=True)
        trueDiscriminatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(trueDiscriminator)*(1-self.ganModelSettings['smooth']),logits=trueDiscriminatorLogits))
        predictedDiscriminatorLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictedDiscriminator),logits=predictedDiscriminatorLogits))
        discriminatorLoss = trueDiscriminatorLoss + predictedDiscriminatorLoss
        
        # calc generator loss
        generatorMaskLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predictedDiscriminator), logits=predictedDiscriminatorLogits))
        
        # Update all variables
        self.compiled_metrics.update_state(dataY, predictedY)
        self.discriminatorLossTracker.update_state(discriminatorLoss)
        self.maskLossTracker.update_state(generatorMaskLoss)
            
        return {m.name: m.result() for m in self.metrics}

    def call(self,inputs):
        rfiMask = self.generator(inputs, training=False) 
        return rfiMask

    def printSummary(self, printEncoder=True, printDecoder=True):
        if printEncoder: self.generator.encoder.encoder.summary()
        if printDecoder: self.generator.decoder.decoder.summary()
        self.generator.summary()

        self.summary()

class GanLoader():
    def __init__(self, dataSettings, ganModelSettings, plotsLocation=None, modelName=None, dataShape=None, run=None, dinoModelSettings=None):
        self.dataSettings = dataSettings
        self.ganModelSettings = ganModelSettings
        self.dinoModelSettings = dinoModelSettings

        if dataShape is None:
            self.dataShape = dataSettings['inputShape']
        else:
            self.dataShape = dataShape

        if run is None:
            self.logName = tf.Variable("run_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            self.trainModelDir, self.logDir = utils.functions.getModelLocation(os.path.join('gan',modelName))
        else:
            self.logName = tf.Variable("run_{}_{}".format(run, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            self.trainModelDir, self.logDir = utils.functions.getModelLocation(os.path.join('gan',modelName),'run_'+str(run))

        if plotsLocation is None:
            self.plotsLocationSupervised = None
        else:
            self.plotsLocationSupervised = os.path.join(plotsLocation,'supervised')
            os.makedirs(self.plotsLocationSupervised, exist_ok=True)
    
    def makeModel(self,metadataStyleVectorSize=None):
        self.gan = GANModel(self.dataSettings, self.ganModelSettings, self.dataShape,dinoModelSettings=self.dinoModelSettings,metadataStyleVectorSize=metadataStyleVectorSize)

    def summary(self):
        self.gan.generator.encoder.encoder.summary()
        if self.ganModelSettings['concatDino']:
            self.gan.generator.dinoEncoder.encoder.summary()
        self.gan.generator.decoder.decoder.summary()
        self.gan.summary()

    def getLatestCheckpoint(self):
        dummyCheckpoint = tf.train.Checkpoint(logName = self.logName)       
        ckpt_manager = tf.train.CheckpointManager(dummyCheckpoint, self.trainModelDir, max_to_keep=1)

        latestCheckpoint = 0
        if ckpt_manager.latest_checkpoint:
            # Check using the name of the checkpoint if it already was finished
            pattern = re.compile(r'-(\d+)$')
            match = pattern.search(ckpt_manager.latest_checkpoint)
            if match:
                latestCheckpoint = int(match.group(1))
        return latestCheckpoint
        
    def restoreWeights(self, restorePartial=False):
        checkpoint = tf.train.Checkpoint(logName = self.logName, generator = self.gan.generator, discriminator = self.gan.discriminator)           
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, self.trainModelDir, max_to_keep=1)

        loadedEpoch = 0
        if self.ckpt_manager.latest_checkpoint:
            # Check using the name of the checkpoint if it already was finished
            pattern = re.compile(r'-(\d+)$')
            match = pattern.search(self.ckpt_manager.latest_checkpoint)
            if match:
                loadedEpoch = int(match.group(1))

            if loadedEpoch<=self.ganModelSettings['freezeEncoderEpochs']:
                print("Loading checkpoint: freeze encoder")
                self.gan.generator.encoder.trainable = False
                self.gan.compile(metrics = self.gan.compileMetrics)
            elif self.ganModelSettings['concatDino'] or self.ganModelSettings['modConv']:
                self.gan.generator.dinoEncoder.trainable = False
                self.gan.compile(metrics = self.gan.compileMetrics)

            # Fit the model once
            inputShape = self.dataShape[0]
            outputShape = self.dataShape[1]
            dummyIn = np.zeros((self.dataSettings['batchSize'], inputShape[0], inputShape[1], inputShape[2]))
            dummyOut = np.zeros((self.dataSettings['batchSize'], outputShape[0], outputShape[1], outputShape[2]))
        
            self.gan.fit(dummyIn, dummyOut, batch_size=self.dataSettings['batchSize'], epochs=1,verbose=2)
            status = checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            
            if not restorePartial:
                status.assert_consumed()
            else:
                status.expect_partial()
        return loadedEpoch
    
    def SetDinoEncoderWeights(self, weights):
        if self.ganModelSettings['concatDino'] or self.ganModelSettings['modConv']:
            encoderWeights = weights
            self.gan.generator.dinoEncoder.set_weights(encoderWeights)
            self.gan.generator.dinoEncoder.trainable = False
        else:
            self.gan.generator.encoder.set_weights(weights)
        self.gan.compile(metrics = self.gan.compileMetrics)

    def ssToSl(self, ssDataShape):
        ssGan = GANModel(self.dataSettings, self.ganModelSettings, loadSS=True, dataShape=ssDataShape)
        ssLogName = tf.Variable("run={}_{}".format(self.run,datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        
        checkpoint = tf.train.Checkpoint(logName = ssLogName, generator = ssGan.generator, discriminator = ssGan.discriminator)  
        checkpointManager = tf.train.CheckpointManager(checkpoint, self.ssModelDir, max_to_keep=1)        

        ssEpoch = 0
        if checkpointManager.latest_checkpoint:
            checkpointPath = checkpointManager.latest_checkpoint
            pattern = re.compile(r'-(\d+)$')
            match = pattern.search(checkpointPath)
            if match:
                ssEpoch = int(match.group(1))

            # Fit the model once
            inputShape = ssDataShape[0]
            outputShape = ssDataShape[1]
            dummyIn = np.zeros((self.dataSettings['batchSize'], inputShape[0], inputShape[1], inputShape[2]))
            dummyOut = np.zeros((self.dataSettings['batchSize'], outputShape[0], outputShape[1], outputShape[2]))
        
            ssGan.fit(dummyIn, dummyOut, batch_size=self.dataSettings['batchSize'], epochs=1,verbose=2)
            status = checkpoint.restore(checkpointPath).expect_partial()
            status.assert_consumed()

            sumWeights = tf.reduce_sum([tf.reduce_sum(variable) for variable in self.gan.generator.encoder.weights])
            print ('Sum of weights before loading: {}'.format(sumWeights))

            sourceWeights = ssGan.generator.encoder.encoder.weights
            sourceWeightKeys = [weight.name for weight in sourceWeights]

            for weightIdx, targetWeight in enumerate(self.gan.generator.encoder.encoder.weights):
                sourceWeightIndex = weightIdx
                sourceValue = sourceWeights[sourceWeightIndex]
                targetWeight.assign(sourceValue)

            sumWeights = tf.reduce_sum([tf.reduce_sum(variable) for variable in self.gan.generator.encoder.weights])
            print ('Sum of weights after loading: {}'.format(sumWeights))
        
        self.gan.generator.encoder.trainable = (self.ganModelSettings['freezeEncoder'] == False)
        self.gan.compile(metrics = self.gan.compileMetrics)
    
        return ssEpoch
  
    def getModel(self):
        return self.gan
    
    def loadCallbacks(self, dataGenerator = None, tensorBoardModelStatistics=False,tensorBoardModelPerformance=False, tensorBoardModelEmbeddingGenerator=None):
        callbacks = []
        
        # Tensorboard callback
        tensorboardDir = os.path.join(self.logDir, self.logName.numpy().decode('utf-8'))

        if tensorBoardModelPerformance:
            profile_batch = '5,45'
        else:
            profile_batch = 0

        if tensorBoardModelStatistics:
            if tensorBoardModelEmbeddingGenerator is None:
                tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=tensorboardDir, histogram_freq=1,write_graph=True, profile_batch=profile_batch)
            else:
                tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=tensorboardDir, histogram_freq=1,write_graph=True, profile_batch=profile_batch,
                                                                     embeddings_freq=1,
                                                                     embeddings_layer_names=['Decoder_depth_bottleneck_dinoMappingNetwork'],
                                                                     embeddings_metadata='metadata.npy',
                                                                     embeddings_data='embeddings_epoch_layer_styleMapping.npy')

        else:
            tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=tensorboardDir, histogram_freq=0,write_graph=False)
        
        callbacks.append(tensorboardCallback)
        
        # Checkpoint callback
        checkpointCallback = utils.models.Callbacks.CheckpointSaver(self.ckpt_manager)
        callbacks.append(checkpointCallback)

        # Plotter callback
        if dataGenerator is not None:
            if self.plotsLocationSupervised is None:
                raise ValueError("Plots location is not defined.")
            plotterCallback = utils.models.Callbacks.Plotter(dataGenerator, self.plotsLocationSupervised, plotPretext = False, plotPredictions = False)
            callbacks.append(plotterCallback)

        if tensorBoardModelEmbeddingGenerator is not None:
            embeddingCallback = utils.models.Callbacks.EmbeddingSaverCallback(tensorBoardModelEmbeddingGenerator, tensorboardDir)
            callbacks.append(embeddingCallback)
        return callbacks