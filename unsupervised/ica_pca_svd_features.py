import os

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

import utils as utils

datasetName = "LOFAR_L2014581 (recording)"
datafolderSubdir = 'dataset250k'
nSamples = 6000

featuresList = list(range(2,63,2))

def ica_fit_transform(flattenedX, nFeatures):
    # Center the data
    centeredX = flattenedX- np.mean(flattenedX, axis=1,keepdims=True)

    # Perform ICA
    ica = FastICA(n_components=nFeatures, whiten="arbitrary-variance", max_iter=400, random_state=1)
    embeddingX = ica.fit_transform(centeredX)
    return embeddingX

def pca_fit_transform(flattenedX, nFeatures, whiten=False):
    pca = PCA(n_components=nFeatures, whiten=whiten)
    embeddingX = pca.fit_transform(flattenedX)
    return embeddingX

def svd_fit_transform(flattenedX, nFeatures):
    centeredX = flattenedX - np.mean(flattenedX,axis=0)
    U, S, Vt = np.linalg.svd(centeredX, full_matrices=False)

    # Keep the first singular values/vectors to reduce dimensionality
    # U_k = U[:, :nFeatures]
    # S_k = np.diag(S[:nFeatures])
    Vt_k = Vt[:nFeatures, :]

    embeddingX = centeredX @ Vt_k.T 
    return embeddingX

def embeddingExists(featuresFilename, nFeatures, samplesHash):
    if os.path.exists(featuresFilename):
        existingnFeatures, existingEmbedding, savedHash = utils.functions.LoadEmbedding(featuresFilename, samplesHash)
        assert existingnFeatures == nFeatures
        return True
    return False

def CalcEmbeddings():
    h5SetsLocation = utils.functions.getH5SetLocation(datasetName)
    trainSamplesFilename = utils.functions.getDatasetLocation(datasetName, 'trainSamples',subdir=datafolderSubdir)
    valSamplesFilename = utils.functions.getDatasetLocation(datasetName, 'valSamples',subdir=datafolderSubdir)

    dataSettings = {}
    dataSettings['normalizationMethod'] = 12
    dataSettings['batchSize'] = 20
    dataSettings['subbands'] = None
    dataSettings['augmentation'] = None

    dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dimensionReduction', 'original', 4, trainSamplesFilename,dataSettings=dataSettings, bufferAll=True, cacheDataset=True, nSamples = nSamples)
    #valGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'gan', 'val', dataSettings['inputShape'][2],valSamplesFilename, ganParameters=ganModelSettings, dataSettings=dataSettings, bufferAll=True, cacheDataset=True, nSamples=dataSettings['nValSamples'])

    samplesHash = dataGenerator.samplesHash

    dataX = []
    for batchX, batchY in dataGenerator:
        dataX.extend(batchX)
    dataX = np.asarray(dataX)
    flattenedX = np.reshape(dataX, (dataX.shape[0], -1))

    icaEmbeddingDir,_ = utils.functions.getModelLocation('ica_norm={}'.format(dataSettings['normalizationMethod']), 'embedding')
    pcaEmbeddingDir,_ = utils.functions.getModelLocation('pca_norm={}'.format(dataSettings['normalizationMethod']), 'embedding')
    svdEmbeddingDir,_ = utils.functions.getModelLocation('svd_norm={}'.format(dataSettings['normalizationMethod']), 'embedding')

    for nFeatures in tqdm(featuresList):
        featuresFilename = 'embedding_nFeatures={}.pkl'.format(nFeatures)
               
        icaEmbeddingFilename = os.path.join(icaEmbeddingDir, featuresFilename)
        if embeddingExists(icaEmbeddingFilename, nFeatures, samplesHash)==False:
            embeddingIca = ica_fit_transform(flattenedX, nFeatures)
            utils.functions.SaveEmbedding(icaEmbeddingFilename, embeddingIca, nFeatures, samplesHash)
        
        pcaEmbeddingFilename = os.path.join(pcaEmbeddingDir, featuresFilename)
        if embeddingExists(pcaEmbeddingFilename, nFeatures, samplesHash)==False:
            embeddingPca = pca_fit_transform(flattenedX, nFeatures)
            utils.functions.SaveEmbedding(pcaEmbeddingFilename, embeddingPca, nFeatures, samplesHash)
        
        svdEmbeddingFilename = os.path.join(svdEmbeddingDir, featuresFilename)
        if embeddingExists(svdEmbeddingFilename, nFeatures, samplesHash)==False:
            embeddingSvd = svd_fit_transform(flattenedX, nFeatures)
            utils.functions.SaveEmbedding(svdEmbeddingFilename, embeddingSvd, nFeatures, samplesHash)