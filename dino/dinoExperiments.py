import os

import utils as utils


def GetExperiment(experiment,debug=False):
    # DINO with ViT as the backbone
    if experiment == 1:
        userSettings = {
            'dataSettings' : {
                'subbands':[65,66,67,68,69],
                'batchSize': 20,
            },
            'dinoSettings':{
                'architecture': 'vit-xs',
                'learningRate': 0.005,
                
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
        
    
    # DINO with CNN as the backbone with 20 samples. Starting by resnet-50 parameters
    elif experiment == 101: # Resnet default
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
            'nSsDinoSamples':20000,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'studentLocalSize': (32,128),# (32, 112),
            'localScale': (0.25, 0.375),#(0.25, 0.328125),
        },
    }
    elif experiment == 102: # momentum 0.999
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
    elif experiment == 103: # momentum 0.9995
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
    elif experiment == 104: # momentum 0.9995, tarmup temp 0.01
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
    elif experiment == 105: # momentum 0.9999
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
    elif experiment == 106: # momentum 0.9999, warmup temp 0.01
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
    elif experiment == 107: # momentum 0.9999, warmup temp 0.01
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
    elif experiment == 108: # momentum 0.9999, warmup temp 0.01, WD /10
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
    elif experiment == 109: 
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
    elif experiment == 110: 
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
    elif experiment == 111: 
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
    elif experiment == 112: 
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
    elif experiment == 113: 
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
    elif experiment == 114: 
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
    elif experiment == 115: 
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
    elif experiment == 116: 
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
    elif experiment == 117: 
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
    elif experiment == 118: 
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
    elif experiment == 119: 
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
    elif experiment == 120: 
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
    elif experiment == 121: 
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
    elif experiment == 122: # now with 8 local crops. 
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
    elif experiment == 123: 
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
    elif experiment == 124:  
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
    elif experiment == 125: 
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
    elif experiment == 126: 
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
    elif experiment == 127: 
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
    elif experiment == 128: 
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
    elif experiment == 129: # color augmentation is applied
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
    elif experiment == 130: # color augmentation is applied
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
    # DINO ViT experiments on the entire dataset
    elif experiment == 131:
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, 
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
    elif experiment == 132:
        userSettings = {
        'dataSettings' : {
            'normalizationMethod': 12,
            'batchSize': 14,
        },
        'dinoSettings':{
            'architecture': 'gan',
            'nLocalCrops': 8,
            'reducedGanDimension': 320, 
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

    return dinoModelSettings,dataSettings,ganModelSettings

def GetDinoModelDir(experiment, run=None):
    dinoModelSettings,dataSettings,ganModelSettings = GetExperiment(experiment,False)
    modelName = utils.models.getDinoModelName(dinoModelSettings,dataSettings,ganModelSettings)
    if run is None:
        mdelDir, _= utils.functions.getModelLocation(os.path.join('dino',modelName))
    else:
        mdelDir, _= utils.functions.getModelLocation(os.path.join('dino',modelName),'run_'+str(run))
    return mdelDir