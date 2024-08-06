import os

import utils as utils


def GetExperiment(experiment,mixedPrecision=False,debug=False):
    # TODO: remove mixedPrecision

    if experiment == 1:
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'batchSize': 20,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'learningRate': 0.005,
                'freeze_last_layer': 2,
            },
        }
    elif experiment == 2:
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'batchSize': 14,
            },
            'dinoSettings':{
                'architecture': 'vit-s', # default
                'learningRate': 0.005,
                'freeze_last_layer': 2,
            },
        }
    elif experiment == 3:
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'normalizationMethod': 6,
                'batchSize': 20,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'freeze_last_layer': 2,
            },
        }
    elif experiment == 4: # greater local scale
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'normalizationMethod': 6,
                'batchSize': 16,
            },
            'dinoSettings':{
                'studentLocalSize': (32, 112),
                'localScale': (0.25, 0.328125),
                'architecture': 'vit-xs',
                'freeze_last_layer': 2,
            },
        }
    elif experiment == 5:
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'batchSize': 20,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'freeze_last_layer': 2,
            },
        }
    elif experiment == 6:
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'batchSize': 14,
            },
            'dinoSettings':{
                'architecture': 'vit-s', # default
                'freeze_last_layer': 2,
            },
        }

    elif experiment == 7: # Base 3,4: Teacher temp lower
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'normalizationMethod': 6,
                'batchSize': 16,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'studentLocalSize': (32, 112),
                'localScale': (0.25, 0.328125),
                'teacher_temp':0.01,        # default: 0.04
            },
        }
    elif experiment == 8: # Base 3,4: Higher weight decay
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'normalizationMethod': 6,
                'batchSize': 16,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'studentLocalSize': (32, 112),
                'localScale': (0.25, 0.328125),
                'weightDecay': 0.0001,      # default: 0.00001
                'weight_decay_end':0.0001,  # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            },
        }
    elif experiment == 9: # Base 3,4: lr higher
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'normalizationMethod': 6,
                'batchSize': 16,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'studentLocalSize': (32, 112),
                'localScale': (0.25, 0.328125),
                'learningRate': 0.001,      # default: 0.0005
            },
        }
    elif experiment == 10: # Base 3,4: output dimension higher
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'normalizationMethod': 6,
                'batchSize': 16,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'studentLocalSize': (32, 112),
                'localScale': (0.25, 0.328125),
                'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
            },
        }

    elif experiment == 11: # Base of 8,9,10 (which were dirived from 3,4)
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 6,
            'batchSize': 16,
        },
        'dinoSettings':{
            'architecture': 'vit-xs',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'learningRate': 0.001,      # default: 0.0005
            'weightDecay': 0.0001,     # default: 0.00001
            'weight_decay_end':0.0001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
        },
    }   
    elif experiment == 12: # Base 11, local scale same ratio af global. Width increased a bit
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 6,
            'batchSize': 16,
        },
        'dinoSettings':{
            'architecture': 'vit-xs',
            'studentLocalSize': (32, 128),
            'localScale': (0.25, 0.375),
            'learningRate': 0.001,      # default: 0.0005
            'weightDecay': 0.0001,     # default: 0.00001
            'weight_decay_end':0.0001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
        },
    }
    elif experiment == 13: # Base 11, Greater LR
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 6,
            'batchSize': 16,
        },
        'dinoSettings':{
            'architecture': 'vit-xs',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'learningRate': 0.01,      # default: 0.0005
            'weightDecay': 0.0001,     # default: 0.00001
            'weight_decay_end':0.0001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
        },
    }  
    elif experiment == 14: # Base 11, Greater WD
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 6,
            'batchSize': 16,
        },
        'dinoSettings':{
            'architecture': 'vit-xs',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'learningRate': 0.001,      # default: 0.0005
            'weightDecay': 0.001,     # default: 0.00001
            'weight_decay_end':0.001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
        },
    } 

    elif experiment == 15: # Base 14, teacher temp 0.001
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 16,
        },
        'dinoSettings':{
            'architecture': 'vit-xs',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),

            'learningRate': 0.001,      # default: 0.0005
            'weightDecay': 0.001,     # default: 0.00001
            'weight_decay_end':0.001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
            'teacher_temp': 0.001,
        },
    } 
    elif experiment == 16: # Base 14, teacher temp 0.0001
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 16,
        },
        'dinoSettings':{
            'architecture': 'vit-xs',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'learningRate': 0.001,      # default: 0.0005
            'weightDecay': 0.001,     # default: 0.00001
            'weight_decay_end':0.001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
            'teacher_temp': 0.0001,
        },
    } 
        
    
    # GAN encoder as backbone
    elif experiment == 21: # GAN, teacher temp 0.001
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'optimizer': 'adamw',
            'normLastLayer':False,
            'use_bn_in_head': False,
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'teacher_temp' : 0.001,
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
            'reducedGanDimension':None,
        },
    } 
    elif experiment == 22: # GAN, teacher temp 0.0001
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        # Loss is always 9.0102
        'dinoSettings':{
            'architecture': 'gan',
            'optimizer': 'adamw',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'teacher_temp' : 0.0001,
            'normLastLayer':False,
            'use_bn_in_head': False,
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
            'reducedGanDimension':None,
        },
    } 

    elif experiment == 23: # Collapse GAN all subbands, temp 0.0001
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'teacher_temp' : 0.0001,
            'optimizer': 'adamw',
            'normLastLayer':False,
            'use_bn_in_head': False,
            'reducedGanDimension':None,

            
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
        },
    }
    elif experiment == 24: # GAN 5 subbands experiment 15 with different normalization
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'optimizer': 'adamw',
            'normLastLayer':False,
            'use_bn_in_head': False,

            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
            'reducedGanDimension':None,
        },
    } 
    elif experiment == 25: # GAN, base 14, reducedGanDim, temp 0.0001
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'optimizer': 'adamw',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'outputDim': 8192,
            'normLastLayer':False,
            'use_bn_in_head': False,


            'reducedGanDimension': 320, # 288 is the hidden size of ViT but results in 4.5 features
            'teacher_temp' : 0.0001,
            'learningRate': 0.001,      # default: 0.0005
            'weightDecay': 0.001,     # default: 0.00001
            'weight_decay_end':0.001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
        },
    }
    
    # Adam optimizer
    elif experiment == 26: # GAN 5 subbands, reduced dimension, adam optimizer, temp 0.001. 
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'outputDim': 8192,
            'normLastLayer':False,
            'use_bn_in_head': False,


            'optimizer': 'adam',
            'reducedGanDimension': 320,
            'teacher_temp' : 0.001,    
        },
    } 
    elif experiment == 27: # GAN 5 subbands, reduced dimension, adam optimizer, temp 0.0005
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'outputDim': 8192,
            'normLastLayer':False,
            'use_bn_in_head': False,


            'optimizer': 'adam',
            'reducedGanDimension': 320,
            'teacher_temp' : 0.0005,    
        },
    } 
    elif experiment == 28: # GAN 5 subbands, reduced dimension, adam optimizer, temp 0.0005
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'outputDim': 8192,
            'normLastLayer':False,
            'use_bn_in_head': False,


            'optimizer': 'adam',
            'reducedGanDimension': 320,
            'teacher_temp' : 0.0001,    
        },
    }   
        
    elif experiment == 29: # Temp from 0.0001, towards 0.001 in 5 epochs
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'modelBaseName' : 'dino_v5',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'outputDim': 8192,
            'normLastLayer':False,
            'use_bn_in_head': False,


            'optimizer': 'adam',
            'reducedGanDimension': 320,

            'warmup_teacher_temp': 0.0001,
            'warmup_teacher_temp_epochs': 5,
            'teacher_temp' : 0.001,
        },
    }
    # elif experiment == 30: # Temp from 0.0001, towards 0.0005 in 3 epochs to check if 0.0005 can then be dynamic
    #     userSettings = {
    #     'dataSettings' : {
    #         'subbands':[65,66,67,68,69],
    #         'normalizationMethod': 12,
    #         'batchSize': 10,
    #     },
    #     'dinoSettings':{
    #         'architecture': 'gan',
    #         'studentLocalSize': (32, 112),
    #         'localScale': (0.25, 0.328125),
    #         'nChannels': 8,
    #         'outputDim': 8192,
    #         'normLastLayer':False,
    #         'use_bn_in_head': False,

    #         'optimizer': 'adam',
    #         'reducedGanDimension': 320,

    #         'warmup_teacher_temp': 0.0001,
    #         'warmup_teacher_temp_epochs': 4,
    #         'teacher_temp' : 0.0005,
    #     },
    # }
    
    elif experiment == 31: # According to the resnet parameters
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320,
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            'learningRate':0.2, 
            "warmup_epochs":30, 
            "min_lr":0.0005, 
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            'teacher_temp' : 0.01,
            'momentumTeacher': 0.996,
            'warmup_teacher_temp': 0.0001,
            'warmup_teacher_temp_epochs': 10,
        },
    }
        
    elif experiment == 32: # use batchnorm in head 
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320,
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            'learningRate':0.2, 
            "warmup_epochs":30, 
            "min_lr":0.0005, 
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            'teacher_temp' : 0.01,
            'momentumTeacher': 0.996,
            'warmup_teacher_temp': 0.0005,
            'warmup_teacher_temp_epochs': 10,
        },
    }
    elif experiment == 33: # norm last layer and use batchnorm in head
        userSettings = {
        'dataSettings' : {
            'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320,
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            'learningRate':0.2, 
            "warmup_epochs":30, 
            "min_lr":0.0005, 
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            'teacher_temp' : 0.001,
            'momentumTeacher': 0.996,
            'warmup_teacher_temp': 0.0001,
            'warmup_teacher_temp_epochs': 4,
        },
    }

    
    # All subbands/local experiments
    # elif experiment == 50: # Experiment 23 with temp 0.001
    #     userSettings = {
    #     'dataSettings' : {
    #         'normalizationMethod': 12,
    #         'batchSize': 10,
    #     },
    #     'dinoSettings':{
    #         'architecture': 'gan',
    #         'optimizer': 'adamw',
    #         'normLastLayer':False,
    #         'use_bn_in_head': False,
    #         'studentLocalSize': (32, 112),
    #         'localScale': (0.25, 0.328125),
    #         'nChannels': 8,
    #         'teacher_temp' : 0.001,
    #         'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
    #         'reducedGanDimension':None,
    #     },
    # }
    elif experiment == 51: # temp 0.0005
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 10,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),
            'nChannels': 8,
            'outputDim': 8192,
            'normLastLayer':False,
            'use_bn_in_head': False,

            'optimizer': 'adam',
            'reducedGanDimension': 320, # is default
            'teacher_temp' : 0.0005,    

            'nLocalCrops': 8,
            'momentumTeacher': 0.9999,
            'weightDecay': 1e-05,
            'weight_decay_end': 1e-05,
            'learningRate': 0.0005,
            'warmup_epochs': 0,
            'min_lr': 1e-06,
            'warmup_teacher_temp_epochs': 0,
        },
    }
    elif experiment == 52: # Base resnet parameters, explore the effect of the temperature and momentum
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            

            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            
            'momentumTeacher': 0.9995,

            # Learning rate
            "warmup_epochs":30,
            'learningRate':0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Temperature
            'warmup_teacher_temp_epochs': 4,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.01,           # teacher_temp for the rest of the training
        },
    }
    elif experiment == 53: # Base resnet parameters, explore the effect of the temperature and momentum
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            

            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            
            'momentumTeacher': 0.999,

            # Learning rate
            "warmup_epochs":30,
            'learningRate':0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Temperature
            'warmup_teacher_temp_epochs': 4,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.01,           # teacher_temp for the rest of the training
        },
    }
   
    elif experiment == 54: # According to the resnet parameters
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            

            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            
            'momentumTeacher': 0.996,

            # Learning rate
            "warmup_epochs":30,
            'learningRate':0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Temperature
            'warmup_teacher_temp_epochs': 4,
            'warmup_teacher_temp': 0.01,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.1,           # teacher_temp for the rest of the training
        },
    }
        
    elif experiment == 55: # Base resnet parameters, explore the effect of the temperature and momentum
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            

            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            
            'momentumTeacher': 0.9997,

            # Learning rate
            "warmup_epochs":30,
            'learningRate':0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Temperature
            'warmup_teacher_temp_epochs': 4,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.01,           # teacher_temp for the rest of the training
        },
    }
        
    elif experiment == 56: # Base resnet parameters, explore the effect of the temperature and momentum
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            

            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            
            'momentumTeacher': 0.9995,

            # Learning rate
            "warmup_epochs":30,
            'learningRate':0.005, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000025, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Temperature
            'warmup_teacher_temp_epochs': 4,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.01,           # teacher_temp for the rest of the training
        },
    }
    elif experiment == 57: # Base resnet parameters, explore the effect of the temperature and momentum
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'normLastLayer':True,
            'use_bn_in_head': True,

            'optimizer': 'adamw',
            

            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,
            
            'momentumTeacher': 0.999,

            # Learning rate
            "warmup_epochs":30,
            'learningRate':0.005, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000025, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Temperature
            'warmup_teacher_temp_epochs': 4,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.01,           # teacher_temp for the rest of the training
        },
    }
        
    # All subbands with 20k samples. Starting by base resnet parameters 
    elif experiment == 60: # Resnet default
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            #'nLocalCrops': 6,
            # 'reducedGanDimension': 320, # is default

            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            #'optimizer': 'adamw',
            #'normLastLayer':True,
            #'use_bn_in_head': True,

            # Learning rate
            #"warmup_epochs":10,
            #'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            #"min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            #'weightDecay': 0.000001,   
            #'weight_decay_end':0.000001,

            # Momentum
            #'momentumTeacher': 0.996,

            # Temperature
            #'warmup_teacher_temp_epochs': 50,
            #'warmup_teacher_temp': 0.04,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            #'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }
    elif experiment == 61: # momentum 0.999
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,

            # Momentum
            'momentumTeacher': 0.999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.04,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }    
    elif experiment == 62: # momentum 0.9995
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,

            # Momentum
            'momentumTeacher': 0.9995,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.04,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }     
    elif experiment == 63: # momentum 0.9995, tarmup temp 0.01
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,

            # Momentum
            'momentumTeacher': 0.9995,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.01,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }  
    elif experiment == 64: # momentum 0.9999
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.04,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }     
    elif experiment == 65: # momentum 0.9999, warmup temp 0.01
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.01,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }
    elif experiment == 66: # momentum 0.9999, warmup temp 0.01
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000001,   
            'weight_decay_end':0.000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }     
    elif experiment == 67: # momentum 0.9999, warmup temp 0.01, WD /10
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.0000001,   
            'weight_decay_end':0.0000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }    
    elif experiment == 68: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.07,           # teacher_temp for the rest of the training
        },
    }         
    elif experiment == 69: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.01,           # teacher_temp for the rest of the training
        },
    } 
    elif experiment == 70: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.00000001,   
            'weight_decay_end':0.00000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.01,           # teacher_temp for the rest of the training
        },
    }     
    elif experiment == 71: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training
        },
    }     
    elif experiment == 72: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9995,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training
        },
    }   
    elif experiment == 73: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9995,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0014,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training
        },
    }   
    elif experiment == 74: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0014,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training
        },
    } 
    elif experiment == 75: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0014,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training
            
            'freeze_last_layer':4,
        },
    }      
    elif experiment == 76: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0013,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training
        },
    }  
    elif experiment == 77: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0013,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training

            'clipGradient': 0.3,
        },
    } 
    elif experiment == 78: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.03, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.00048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    }      
    elif experiment == 79: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'lamb',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    }      
    elif experiment == 80: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adam',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.3, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.0048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.00001,   
            'weight_decay_end':0.00001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    }  
    elif experiment == 81: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    }    
    elif experiment == 82: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.00048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0013,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    }  
    elif experiment == 83: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.00048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0013,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    }     
    elif experiment == 84: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.00048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9997,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0013,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    }   
    elif experiment == 85: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.00048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.996,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.013,           # teacher_temp for the rest of the training
            #'clipGradient': 0.3,
        },
    }     
    elif experiment == 86: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.00048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.996,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.01,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.1,           # teacher_temp for the rest of the training

            #'clipGradient': 0.3,
        },
    } 
        
    elif experiment == 87: # 81 but now with 8 local crops. 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0018,           # teacher_temp for the rest of the training
        },
    }    
    elif experiment == 88: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.001,           # teacher_temp for the rest of the training
        },
    }    
        
    elif experiment == 89: # 81 but now with 8 local crops. 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0002,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.002,           # teacher_temp for the rest of the training
        },
    } 

    elif experiment == 90: # 81 but now with 8 local crops. 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0001,           # teacher_temp for the rest of the training
        },
    }    
        
    elif experiment == 91: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0004,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0004,           # teacher_temp for the rest of the training
        },
    }    
    elif experiment == 92: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0003,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0003,           # teacher_temp for the rest of the training
        },
    }  

    elif experiment == 93: 
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.00035,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.00035,           # teacher_temp for the rest of the training
        },
    }   
    elif experiment == 94: # color augmentation is applied!!
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
            'augmentation':'fc', # color augmentation is applied!!
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            
            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.0003,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.0003,           # teacher_temp for the rest of the training
        },
    }     
        
    elif experiment == 95: # color augmentation is applied!!
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
            'augmentation':'fc', # color augmentation is applied!!
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            
            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.000048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.9999,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.00035,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.00035,           # teacher_temp for the rest of the training
        },
    }     
        

    elif experiment == 96: # copy of 85 to expore the histograms when NaN will occur
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 6,
            'reducedGanDimension': 320, # is default
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),

            'optimizer': 'adamw',
            'normLastLayer':True,
            'use_bn_in_head': True,

            # Learning rate
            "warmup_epochs":10,
            'learningRate':0.003, #0.1,  # linear from 0 to min_lr in warmup_epochs steps
            "min_lr":0.00048, # 0.0005,     # cosine from learningRate to min_lr in total_epochs - warmup_epochs steps

            # Weight decay
            'weightDecay': 0.000000001,   
            'weight_decay_end':0.000000001,

            # Momentum
            'momentumTeacher': 0.996,

            # Temperature
            'warmup_teacher_temp_epochs': 50,
            'warmup_teacher_temp': 0.001,    # linear from warmup temp to temp in warmup_teacher_temp_epochs steps
            'teacher_temp' : 0.013,           # teacher_temp for the rest of the training
            #'clipGradient': 0.3,
        },
    }    
    elif experiment == 100: # Base 20, best model according to results with dynamic loss. All subbands.
        userSettings = {
        'dataSettings' : {
            #'subbands':[65,66,67,68,69],
            'normalizationMethod': 12,
            'batchSize': 16,
        },
        'dinoSettings':{
            'architecture': 'vit-xs',
            'studentLocalSize': (32, 112),
            'localScale': (0.25, 0.328125),

            'learningRate': 0.001,      # default: 0.0005
            'weightDecay': 0.001,     # default: 0.00001
            'weight_decay_end':0.001, # default: 0.00001 If not equal to weightDecay, then Habrok will crash
            'outputDim': 8192,          # default: 65536 outputDim must be low. Else k-means will become very slow
            'teacher_temp': 0.0001,
        },
    } 
        
    elif experiment == 101:
        userSettings = {
            'dataSettings' : {
                'batchSize': 5,
                'nSsDinoSamples':100,
                'nDinoValSamples':100,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'learningRate': 0.005,
                'freeze_last_layer': 2,
            },
        }
        

    elif experiment == 102:
        userSettings = {
            'dataSettings' : {
                'batchSize': 5,
                'nSsDinoSamples':100,
                'nDinoValSamples':100,
            },
            'dinoSettings':{
                'architecture': 'gan',
                'learningRate': 0.005,
                'freeze_last_layer': 2,
            },
        }

    ganModelSettings = utils.models.defaultGanModelSettings.copy()
    dataSettings = utils.models.defaultDataSettings.copy()
    if userSettings['dinoSettings']['architecture'] == 'gan':
         dinoModelSettings = utils.models.defaultDinoModelWithGanSettings.copy()
    else:
        dinoModelSettings = utils.models.defaultDinoModelSettings.copy()
    dataSettings.update(userSettings['dataSettings'])
    dinoModelSettings.update(userSettings['dinoSettings'])
    dinoModelSettings['experiment']=experiment

    if 'ganSettings' in userSettings:
        ganModelSettings.update(userSettings['ganSettings'])

    if debug:
        dinoModelSettings['modelBaseName'] = 'debug_' + dinoModelSettings['modelBaseName']

    if mixedPrecision:
        dinoModelSettings['modelBaseName'] = dinoModelSettings['modelBaseName'] + '_fp16'
    return dinoModelSettings,dataSettings,ganModelSettings

def GetDinoModelDir(experiment, run=None):
    dinoModelSettings,dataSettings,ganModelSettings = GetExperiment(experiment,False,False)
    modelName = utils.models.getDinoModelName(dinoModelSettings,dataSettings,ganModelSettings)
    if run is None:
        mdelDir, _= utils.functions.getModelLocation(os.path.join('dino',modelName))
    else:
        mdelDir, _= utils.functions.getModelLocation(os.path.join('dino',modelName),'run_'+str(run))
    return mdelDir