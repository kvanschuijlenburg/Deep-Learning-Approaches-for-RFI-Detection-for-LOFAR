import os
import random
import pickle

import utils as utils

nTrainingSamples = 250000 # The first 6000 is used as train SL
nValSamples = 25000 # The first 1000 is used as val SL, the first 6000 is used as val SSL, 6000-7000 is used as test set
nTestSamples = 2775
targetTrainTestSubDir = 'dataset250k' # TODO: remove protect in final model
targetTrainSetName = 'trainSamples'
targetValSetName = 'valSamples'

trainValDatasetName = "LOFAR_L2014581 (recording)"
testDatasetName = "Haomin Sun et al. test set"
targetTestSetName = 'testSamples'	

sampleSet = 'test'

def SampleH5():
    # raise ValueError('This script will overwrite training and validation sampling') # TODO remove exception in final model
    h5SetsLocation = utils.functions.getH5SetLocation(trainValDatasetName)
    preprocessedLocation = utils.functions.getDatasetLocation(trainValDatasetName)
    os.makedirs(os.path.join(preprocessedLocation,targetTrainTestSubDir),exist_ok=True)
    trainSamplesFilename = os.path.join(preprocessedLocation,targetTrainTestSubDir, targetTrainSetName)
    valSamplesFilename = os.path.join(preprocessedLocation,targetTrainTestSubDir, targetValSetName)
    if os.path.exists(trainSamplesFilename):
        print("Train samples already exist. Delete the file to sample it again.")
        return
    if os.path.exists(valSamplesFilename):
        print("Val samples already exist. Delete the file to sample it again.")
        return

    print("Start random sampling")
    nSamples = nTrainingSamples + nValSamples
    sampleList = utils.functions.sampleObservations(h5SetsLocation, nSamples, strategy='equalSubbands_equalCorrelation_randomTime')
    random.shuffle(sampleList)

    print('save training sampleList and frequencySubbandMapping as pickle')
    file = open(os.path.join(preprocessedLocation,targetTrainTestSubDir, targetTrainSetName),"wb")
    pickle.dump([sampleList[0:nTrainingSamples]],file)
    file.close()

    print('save validation sampleList and frequencySubbandMapping as pickle')
    file = open(os.path.join(preprocessedLocation,targetTrainTestSubDir, targetValSetName),"wb")
    pickle.dump([sampleList[nTrainingSamples:]],file)
    file.close()