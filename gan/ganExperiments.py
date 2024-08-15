import os
        
import utils as utils

def GetExperiment(experiment, debug=False):
    # RFI-GAN reference experiments
    if experiment == 0: # Skips disabled
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 6000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'skipsEnabled':False,
                'scheduledEpochs':120, 
            },
        }
    
    elif experiment == 1: # Normalization method 12
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 6000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':120,
            },
        }
    elif experiment == 2: # Normalization method 13
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 6000,
                'normalizationMethod': 13,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':120,
            },
        }

    elif experiment == 3: # 500 samples
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':700,
            },
        }
    elif experiment == 4: # 1000 samples
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 1000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':350,
            },
        }
    elif experiment == 5: # 2000 samples
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 2000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':300,
            },
        }
    elif experiment == 6: # 3000 samples
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 3000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':250,
            },
        }
    elif experiment == 7: # 5000 samples
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 5000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
            },
        }

    # DiMoGAN experiments
    elif experiment == 8: # 500 samples, pretrained
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':1000,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'freezeEncoderEpochs': 500,
            },
        }
    elif experiment == 9: # 500 samples, concat
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':700,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'concatDino':True,
            },
        }

    # Default and single experiments
    elif experiment == 10: # 500 samples, modConv default
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
            },
        }
    elif experiment == 11: # 500 samples, modConv relu activation
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'relu',
            },
        }
    elif experiment == 12: # 500 samples, modConv random style input. To measure if the style has an effect
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'maskStyleInput': True, # style input zeros leading to unstable training. Truncated_normal
            },
        }
    elif experiment == 13: # 500 samples, modConv default no style input normalization and one mapping layer
       userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'nMappingLayers':1,
                'styleMappingNormInput':False,
            },
       }
    elif experiment == 14: # 500 samples, modConv default no style input normalization and no mapping layers
       userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'nMappingLayers':0,
                'styleMappingNormInput':False,
            },
       }

    elif experiment == 15: # 500 samples, modConv default no mapping layers but with style input normalization
       userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'nMappingLayers':0,
                'styleMappingNormInput':True,
            },
       }

    # add layer normalization methods
    elif experiment == 20: # 500 samples, modConv addNorm = IN
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'modConvResUpNormAdd':'IN'
            },
        }
    elif experiment == 21: # 500 samples, modConv addNorm = IN with bias
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'modConvResUpNormAdd':'IN_bias'
            },
        }

    # 30-34: dropout experiments
    elif experiment == 30: # 500 samples, modConv, dropout=0.2
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'dropoutRate':0.2,
            },
        }   
    elif experiment == 31: # 500 samples, modConv, dropout=0.4
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'dropoutRate':0.4,
            },
        }
    
    # 35-39: lr experiments
    elif experiment == 35: # 500 samples, modConv lr. Previously experiment 24
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'lrStart': 1e-4,        # 'lrStart': 1e-4,
                'lrEnd': 5e-6,        # 'lrEnd': 3.5e-5,
            },
        }

    # 40-44: style mapping lr multiplier experiments
    elif experiment == 40: # 500 samples, modConv lr multiplier=0.01
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'styleMappingLrMul': 0.01,
            },
        }
    elif experiment == 41: # 500 samples, modConv lr multiplier=0.1
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'styleMappingLrMul': 0.1,
            },
        }
    elif experiment == 42: # 500 samples, modConv lr multiplier=0.2
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'styleMappingLrMul': 0.2,
            },
        }

    elif experiment == 50: # 500 samples, modConv using metadata as style input
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'metadataStyleInput':'subband',
            },
        }

    # 55: also encoder modConv
    elif experiment == 55: # 500 samples, modConv default but also in encoder
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modConvEncoder':True,
                           
                'modulatedConvActivation': 'leaky_relu',
                # In this model, the downsampling is done, but the resDown is disabled in the code. Still functioning
            },
        }
    elif experiment == 56: # 500 samples, modConv default but also in encoder
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 500,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':200,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modConvEncoder':True,
                
                
                
                'modulatedConvActivation': 'leaky_relu',
                'modConvResUpNormAdd':'scale',

                # Downsampling is enabled, 4 tiles
                # ResDown only in layer 3 and 4, +8 tiles   FAIL
                # ResDown only in layer 4, +4 tiles OK
                # ResDown only in layer 3, +4 tiles OK but maybe not as good
                # ResDown in layer 2,3,4 with modConv normalizing after add


                # 'styleMappingLrMul': 0.1,
                # 'dropoutRate':0.4,
                'styleMappingLrMul': 0.01,
                #'lrStart': 0.002,        # 'lrStart': 1e-4,
                #'lrEnd': 5e-4,        # 'lrEnd': 3.5e-5,
                'skipsEnabled':False,
            },
        }

    # 60+ all experiments with 6000 training samples
    elif experiment == 60: # 6000 samples, modConv default
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 6000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':120,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
            },
        }
    elif experiment == 61: # 6000 samples, modConv default with two mapping layers
        userSettings = {
            'dataSettings' : {
                'nTrainingSamples' : 6000,
                'normalizationMethod': 12,
                'batchSize':10,
            },
            'ganSettings':{
                'scheduledEpochs':70,
                'loadDinoEncoder':132,
                'loadDinoEpoch':9,
                'modConv':True,
                'modulatedConvActivation': 'leaky_relu',
                'nMappingLayers':2,
                'styleMappingNormInput':True,
            },
        }


    dataSettings = utils.models.defaultDataSettings.copy()
    ganModelSettings = utils.models.defaultGanModelSettings.copy()
    dinoModelSettings = utils.models.defaultDinoModelSettings.copy()
    dataSettings.update(userSettings['dataSettings'])
    ganModelSettings.update(userSettings['ganSettings'])
    if 'dinoSettings' in userSettings:
        dinoModelSettings.update(userSettings['dinoSettings'])

    ganModelSettings['experiment']=experiment

    if debug:
        ganModelSettings['modelBaseName'] = 'debug_' + ganModelSettings['modelBaseName']
    return dataSettings, ganModelSettings, dinoModelSettings

def GetGanModelDir(experiment, run=None):
    dataSettings, ganModelSettings, dinoModelSettings = GetExperiment(experiment)
    modelName = utils.models.getGanModelName(ganModelSettings, dataSettings, dinoModelSettings)
    if run is None:
        mdelDir, _= utils.functions.getModelLocation(os.path.join('gan',modelName))
    else:
        mdelDir, _= utils.functions.getModelLocation(os.path.join('gan',modelName),'run_'+str(run))
    return mdelDir