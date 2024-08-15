import os

import numpy as np
from tqdm import tqdm

from gan.ganExperiments import GetGanModelDir
from dino.dinoExperiments import GetDinoModelDir
import unsupervised.cluster_search as cluster_search
import utils as utils
import unsupervised.ica_pca_svd_features as dimensionReduction

# RFI-GAN and DiMoGAN experiment names
ganExperimentNames = [{'exp':0,     'name': 'Skip connections disabled'},
                      {'exp':1,     'name': 'RFI-GAN 6000'},
                      {'exp':2,     'name': 'Mean and SD'}, 
                      {'exp':3,     'name': 'RFI-GAN 500'},
                      {'exp':4,     'name': 'RFI-GAN 1000'},
                      {'exp':5,     'name': 'RFI-GAN 2000'},
                      {'exp':6,     'name': 'RFI-GAN 3000'},
                      {'exp':7,     'name': 'RFI-GAN 5000'},
                      {'exp':10,    'name': 'DiMoGAN 500'},
                      {'exp':11,    'name': 'DiMoGAN 500 ReLU'},
                      {'exp':12,    'name': 'DiMoGAN 500 randomStyle'},
                      {'exp':20,    'name': 'DiMoGAN 500 add norm IN'},
                      {'exp':21,    'name': 'DiMoGAN 500 add norm IN+bias'},
                      {'exp':30,    'name': 'DiMoGAN 500 dropout 0.2'},
                      {'exp':31,    'name': 'DiMoGAN 500 dropout 0.4'},
                      {'exp':35,    'name': 'DiMoGAN 500 lr end 5e-6'},
                      {'exp':40,    'name': 'DiMoGAN 500 style lr 0.01'},
                      {'exp':41,    'name': 'DiMoGAN 500 style lr 0.1'},
                      {'exp':42,    'name': 'DiMoGAN 500 style lr 0.2'},
                      {'exp':60,    'name': 'DiMoGAN 6000'},]

# RFI-GAN experiments
ganExperiments =  range(8)

# DINO experiments
dinoVitExperiments = list(range(1,17))
dinoCnnExperiments = list(range(101,133))
evaluateDinoModels = [{'exp':127, 'bestEpoch':14}, {'exp':128, 'bestEpoch':19}, {'exp':132, 'bestEpoch':9}] 

# DiMoGAN experiments
diMoGANExperiments = [10,11,12,20,21,30,31,35,40,41,42,60]

trainRfiGan = False
trainDinoSearch = False
trainDiMoGan = False


def getGanExperimentsFromNumbers(experimentNumbers):
    ganExperiments = []
    for experimentIdx in experimentNumbers:
        for experiment in ganExperimentNames:
            if experiment['exp'] == experimentIdx:
                ganExperiments.append(experiment)
    return ganExperiments

def background_2_8_2():
    ## 2.8.2 Simulators. 
    # - HIDE
    # - Hera_sim amplitude and phase
    import datasets.hide as hideSim
    import datasets.hera_sim as heraSim
    print("Start Section 2.8.2: Simulators")
    resultsLocation = utils.functions.getPlotLocation("2_8_2 Background (simulators)")
    os.makedirs(resultsLocation, exist_ok=True)
    hideSim.Simulate(resultsLocation)
    heraSim.Simulate(resultsLocation)

def methods_3_1():
    ## 3 methods. 
    # 3.1 Data
    import datasets.sample_datasets as sample_datasets
    import datasets.plot_H5 as plot_H5
    import datasets.data_properties as data_properties
    print("Start Section 3.1: Data")

    resultsLocation = utils.functions.getPlotLocation("3_1 Methods (h5, RFI statistics, AOFlagger examples, RFI phenomena, Normalization)")
    os.makedirs(resultsLocation, exist_ok=True)

    # 3.1.4: Generate sample files for train and validation data
    sample_datasets.SampleH5()

    # 3.1.2: Plot LOFAR observation for all subbands the amplitude and phase
    h5SetsLocation = utils.functions.getH5SetLocation("LOFAR_L2014581 (recording)")
    h5Sets = [filename for filename in os.listdir(h5SetsLocation) if filename.endswith('.h5')]
    if len(h5Sets) == 0:
        print("(Not all) h5 files found in the location: {}. Skipping plotting data visualization".format(h5SetsLocation))
    else:
        plot_H5.plotH5Set(antennasA=["CS002HBA1"], antennasB=["CS005HBA1"], plotType='rgb', plotLocation = resultsLocation)

    # 3.1.2: Plot AOFlagger reference truth labels
    data_properties.PlotAoflaggerExamples(resultsLocation) # Example 85 is used in the thesis

    # 3.1.2: Plot statistics of the datasets. Print avarage RFI and plot the percentage of RFI per subband
    data_properties.PlotRfiPercentagePerSubband(resultsLocation)

    # 3.1.3: Plot RFI phenomena based on the AOFlagger label
    data_properties.PlotRfiPhenomena(resultsLocation) # Example 15 is used in the thesis

    # 3.1.4: Plot for each normalization method the distribution of pixels with amplitudes
    data_properties.PlotAmplitudesHistogramNormalizations(resultsLocation) 

    # 3.1.4: Plot normalization examples for mean, median and none
    data_properties.PlotNormalizedExamples(resultsLocation) # Figure 16 are samples: 59,58,116

def results_4_1():
    # 4 Results
    ## 4.1 RFI-GAN baselines
    import gan.gan_train as gan_train
    import gan.gan_inference as gan_inference
    from gan.gan_compare import CompareModels
    print("Start Section 4.1: RFI-GAN baselines")
    resultsLocation = utils.functions.getPlotLocation("4_1 Results (RFI-GAN)")
    os.makedirs(resultsLocation, exist_ok=True)

    # 4.1: Train gan experiments 0-7 and test them on the test set
    for ganExperiment in ganExperiments:
        if trainRfiGan:
            gan_train.RepeatTrain(experiment=ganExperiment, nRepetitions=5) 
        for run in range(5): 
            # Calculate the tp,fp,tn,fn per sample, and stores it as pickle file
            gan_inference.CalculateMetrics(ganExperiment, dataset='test', run=run)
    
    # 4.1: Calculate statistics and Wilcoxon signed-rank sum test, print table.
    CompareModels(getGanExperimentsFromNumbers([1,2,0,7,6,5,4,3]), datasets = ['test'], logLocation=resultsLocation)
    
    # 4.1: plot figure with accuracy per training set size
    CompareModels(getGanExperimentsFromNumbers([3,4,5,6,7,1]), datasets = ['test'], plotLocation=resultsLocation)
 
def results_4_2():
    ## 4.2 DINO
    from dino.dino_train import train
    from dino.dino_inference import PlotTsne
    print("Start Section 4.2: DINO")

    resultsLocation = utils.functions.getPlotLocation("4_2 Results (DINO)")
    os.makedirs(resultsLocation, exist_ok=True)
    
    if trainDinoSearch:
        # 4.2.1: Train DINO experiments 1-16, the ViT backbone. While training, plot self-attention maps
        for dinoExperiment in dinoVitExperiments:
            train(dinoExperiment, saveEmbeddingEachEpoch=False, resultsLocation=resultsLocation)

        # 4.2.2: Train DINO experiments 101-132. While training, calculate each epoch the embedding on the validation set
        experimentsSaveEmbeddings = [exp['exp'] for exp in evaluateDinoModels]
        for dinoExperiment in dinoCnnExperiments: 
            saveEmbeddingsEachEpoch = dinoExperiment in experimentsSaveEmbeddings
            train(dinoExperiment, saveEmbeddingEachEpoch=saveEmbeddingsEachEpoch)

    for evaluateModel in evaluateDinoModels: 
        dinoExperiment = evaluateModel['exp']
        bestEpoch = evaluateModel['bestEpoch']
        modelDir = GetDinoModelDir(experiment=dinoExperiment)
        # 4.2.2: For experiment 27,28,32, perform for each epoch k-means on the validation set
        cluster_search.ClusterSearchEmbeddings(modelDir, 'kmeans', 'dlModel', valData=True)
        # 4.2.2: For experiment 27,28,32, plot k-means search. On the validation set
        cluster_search.PlotKmeansDirectory(modelDir, resultsLocation, 'dlModel',valData=True, limitK=50, prefixSavename = 'Experiment={} '.format(dinoExperiment))
        # 4.2.2: For experiment 27,28,32, plot k-means search. Highlight epochs with best k-values. On the validation set
        cluster_search.PlotKmeansDirectory(modelDir, resultsLocation, 'dlModel',highlightEpoch=bestEpoch, valData=True, limitK=50, prefixSavename = 'Experiment={} '.format(dinoExperiment))
        # 4.2.2: For experiment 27,28,32, perform t-SNE on the validation set
        PlotTsne(dinoExperiment, bestEpoch,resultsLocation, valData=True,prefixSavename = 'Experiment={} '.format(dinoExperiment)) 

def results_4_3():
    ## 4.3 Embedding and features evaluation
    import gan.gan_inference as gan_inference
    import dino.dino_inference as dino
    import unsupervised.cluster_pipeline as cluster_pipeline
    print("Start Section 4.3: Embedding and features analysis")
    
    resultsLocation = utils.functions.getPlotLocation("4_3 Results (Embedding and features analysis)")
    os.makedirs(resultsLocation, exist_ok=True)

    # Known at this point
    bestGanExperiment = 1
    bestDinoExperiment = 132
    bestDinoEpoch = 9
    bestGanEpoch = 120 # (last epoch), hardcoded to speed up loading. Else the checkpoint has to be loaded each time.

    # 4.3.1: For ICA, PCA, SVD, RFI-GAN, DINO, calculate embedding on training set SL
    dimensionReduction.CalcEmbeddings() # ICA, PCA, SVD
    gan_inference.calcEmbedding(bestGanExperiment, bestGanEpoch, dataset = 'train', run=0) 
    dino.CalcEmbedding(bestDinoExperiment, bestDinoEpoch, valData=False)

    # 4.3.1: For ICA, PCA, SVD, k-means search for m=2,4,...,62, k=5,...,50
    icaPcaSvd = [{'methodName': 'ICA' ,'modelDir':utils.functions.getModelLocation('ica_norm=12')[0], 'algorithm':'kmeans', 'modelType':'traditional'},
                 {'methodName': 'PCA' ,'modelDir':utils.functions.getModelLocation('pca_norm=12')[0], 'algorithm':'kmeans', 'modelType':'traditional'},
                 {'methodName': 'SVD' ,'modelDir':utils.functions.getModelLocation('svd_norm=12')[0], 'algorithm':'kmeans', 'modelType':'traditional'}]
    for method in icaPcaSvd:
        cluster_search.ClusterSearchEmbeddings(method['modelDir'], method['algorithm'], method['modelType'], valData = False)	
    
    # 4.3.1: For ICA, PCA, SVD, plot maximum of k-means search
    cluster_search.PlotMaxSilhouetteScores(icaPcaSvd, plotLocation = resultsLocation)

    # Based on the maxiumum values, and the 2d silhouette maps, the optimal k values and corresponding m features per method are determined
    mFeaturesIca=16
    mFeaturesPca=6
    mFeaturesSvd=6
    kClustersIca=20
    kClustersPca=18
    kClustersSvd=19

    # With this, the experiment data for ica pca and svd is updated
    icaPcaSvd = [{'methodName': 'ICA' ,'modelDir':utils.functions.getModelLocation('ica_norm=12')[0], 'algorithm':'kmeans', 'modelType':'traditional', 'highlight':mFeaturesIca},
                 {'methodName': 'PCA' ,'modelDir':utils.functions.getModelLocation('pca_norm=12')[0], 'algorithm':'kmeans', 'modelType':'traditional', 'highlight':mFeaturesPca},
                 {'methodName': 'SVD' ,'modelDir':utils.functions.getModelLocation('svd_norm=12')[0], 'algorithm':'kmeans', 'modelType':'traditional', 'highlight':mFeaturesSvd}]

    # 4.3.1: For RFI-GAN, DINO, k-means search as well
    ganDino  = [{'methodName': 'RFI-GAN' ,'modelDir':GetGanModelDir(experiment=bestGanExperiment, run=0), 'modelType':'dlModel', 'algorithm':'kmeans','highlight':bestGanEpoch},
                {'methodName': 'DINO' ,'modelDir':GetDinoModelDir(experiment=bestDinoExperiment), 'modelType':'dlModel','algorithm':'kmeans', 'highlight':bestDinoEpoch, 'kLimit':50}]
    for method in ganDino:
        cluster_search.ClusterSearchEmbeddings(method['modelDir'], method['algorithm'], method['modelType'], valData = False)

    # 4.3.1: For ICA, PCA, SVD, RFI-GAN, DINO, plot k-means silhouette scores
    # 4.3.1: For ICA, PCA, SVD, RFI-GAN, DINO, calculate statistics of k-means search by the runs test
    allModels = icaPcaSvd.copy()
    allModels.extend(ganDino)
    cluster_search.EvaluateClusteringSearch(allModels, resultsLocation=resultsLocation, valData=False) 

    # Based on the silhouette plots, the k-values are also determined for RFI-GAN and DiMoGAN
    kClustersGan = 10
    kClustersDino = 34

    icaPcaSvd = [{'method':'ica', 'modelDir':utils.functions.getModelLocation('ica_norm=12')[0], 'nFeatures':mFeaturesIca,  'kClusters':kClustersIca, 'perplexity':10},
                 {'method':'pca', 'modelDir':utils.functions.getModelLocation('pca_norm=12')[0], 'nFeatures':mFeaturesPca,  'kClusters':kClustersPca, 'perplexity':30},
                 {'method':'svd', 'modelDir':utils.functions.getModelLocation('svd_norm=12')[0], 'nFeatures':mFeaturesSvd,  'kClusters':kClustersSvd, 'perplexity':30}]
    ganModel =  [{'method':'gan', 'modelDir':GetGanModelDir(experiment=1, run=0),                'epoch':bestGanEpoch,      'kClusters':kClustersGan, 'perplexity':30}]
    dinoModel = [{'method':'dino','modelDir':GetDinoModelDir(experiment=132),                     'epoch':bestDinoEpoch,     'kClusters':kClustersDino, 'perplexity':30}]
    icaPcaSvdGan = icaPcaSvd.copy()
    icaPcaSvdGan.extend(ganModel)
    allModels = icaPcaSvd.copy()
    allModels.extend(dinoModel)
    allModels.extend(ganModel)

    # 4.3.1: For ICA, PCA, SVD, RFI-GAN, DINO, calculate odd-even clusters by k-means clustering
    cluster_pipeline.clusterEmbeddings(allModels)
    
    # 4.3.1: For ICA, PCA, SVD, RFI-GAN, DINO, calculate t-SNE transformations ?
    cluster_pipeline.tsneEmbeddings(allModels)

    # 4.3.2: For each method: plot t-SNE visualization
    # 4.3.2: For each method: plot silhouette plot with k-means clusters and RFI phenomena
    # 4.3.2: For each method: print clustering statistics
    # 4.3.2: For each method: print odd-even clustering statistic
    cluster_pipeline.IndividualModels(allModels, resultsLocation = resultsLocation) 
  
    # 4.3.6: Compare ICA, PCA, SVD
    # 4.3.6: Plot ICA cluster 7
    cluster_pipeline.CompareModels(icaPcaSvd, 'ica_pca_svd', resultsLocation = resultsLocation) 
    
    # 4.3.6: Compare ICA, PCA, SVD, RFI-GAN
    # 4.3.6: Plot t-SNE map of RFI-GAN embedding and clusters. Semi-transparant non-matching with ICA, PCA, SVD
    cluster_pipeline.CompareModels(icaPcaSvdGan, 'ica_pca_svd_rfiGan', resultsLocation = resultsLocation)
    
    # 4.3.6: Compare ICA, PCA, SVD, RFI-GAN, DINO
    # 4.3.6: Plot t-SNE map of DINO embedding. Coloured by the ICA clusters.
    cluster_pipeline.CompareModels(allModels, 'ica_pca_svd_rfiGan_dino', resultsLocation = resultsLocation)

def results_4_4():
    import gan.gan_train as gan_train
    import gan.gan_inference as gan_inference
    from gan.gan_compare import CompareModels, EvaluateTraining, VisualizePredictions
    print("Start Section 4.4: DiMoGAN")

    resultsLocation = utils.functions.getPlotLocation("4_4 Results (DiMoGAN)")
    os.makedirs(resultsLocation, exist_ok=True)
    
    ## 4.4 DiMoGAN
    # 4.4: Train 5 times each DiMoGAN experiments: 8-9(?),10-15,20-21,30-31,35,40-42,50,55-56(?),60-61
    for ganExperiment in diMoGANExperiments:
        # 4.4: Train 5 times each DiMoGAN experiments
        if trainDiMoGan:
            gan_train.RepeatTrain(experiment=ganExperiment, nRepetitions=5)

        for run in range(5):
            # 4.4: Calc both val and test embedding on all above experiments
            gan_inference.CalculateMetrics(ganExperiment, dataset='test', run=run)
            gan_inference.CalculateMetrics(ganExperiment, dataset='val', run=run)
    
    # 4.4: Evaluate and print comparison of DiMoGAN experiments on val set for table 8. Print table
    diMoGanSearchLocation = os.path.join(resultsLocation,'DiMoGAN search')
    os.makedirs(diMoGanSearchLocation, exist_ok=True)
    CompareModels(getGanExperimentsFromNumbers(diMoGANExperiments), datasets = ['val'], logLocation=diMoGanSearchLocation)

    # 4.4.1: Plot validation accuracy per epoch for RFI-GAN and DiMoGAN 500
    diMoGanBL500Location = os.path.join(resultsLocation,'DiMoGAN_BL_500')
    os.makedirs(diMoGanBL500Location, exist_ok=True)
    CompareModels(getGanExperimentsFromNumbers([3,10]), datasets = ['test'], logLocation=diMoGanBL500Location)
    EvaluateTraining(getGanExperimentsFromNumbers([3,10]), plotLocation=diMoGanBL500Location, plotZoom=False)

    # 4.4.1: Plot validation accuracy per epoch for RFI-GAN and DiMoGAN 6000
    diMoGanBL6000Location = os.path.join(resultsLocation,'DiMoGAN_BL_6000')
    os.makedirs(diMoGanBL6000Location, exist_ok=True)
    CompareModels(getGanExperimentsFromNumbers([1,60]), datasets = ['test'], logLocation=diMoGanBL6000Location)
    EvaluateTraining(getGanExperimentsFromNumbers([1,60]), plotLocation=diMoGanBL6000Location, plotZoom=False)

    # 4.4.2: Plot AOFlagger labels together with DiMoGAN 6000 labels with the TP/TF plots
    gan_inference.Predict(1,'val')
    gan_inference.Predict(60,'val')
    VisualizePredictions(getGanExperimentsFromNumbers([1,60]), resultsLocation=resultsLocation) 
    
    # 4.4.1: Calculate inference time for RFI-GAN, DiMoGAN 6000 for batch size 10,20,25
    inferenceTimes = {}
    for batchSize in tqdm([25,20,10],position=0,desc='Batch size',leave=False):
        inferenceTimes[batchSize] = {}
        for ganExperiment in tqdm([1,60], position=1, desc='Model', leave=False):
            experimentName = getGanExperimentsFromNumbers([ganExperiment])[0]['name']
            inferenceTimes[batchSize][experimentName] = {}
            experimentResults = gan_inference.measureInferenceTime(ganExperiment, batchSize, repeat = 5)
            inferenceTimes[batchSize][experimentName] = np.mean(experimentResults)
    print(inferenceTimes)


if __name__ == '__main__':
    background_2_8_2()
    methods_3_1() 
    results_4_1() 
    results_4_2() 
    results_4_3()
    results_4_4() 