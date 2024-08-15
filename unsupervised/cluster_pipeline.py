import os
import pickle
import time
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

import utils as utils
import unsupervised.tsne as tsne
import unsupervised.cluster_algorithms as clusterAlgorithms


datasetName = "LOFAR_L2014581 (recording)"
datafolderSubdir = 'dataset250k'
nTrainingSamples = 6000
experimentName = 'clustering_pipeline'

dataSettings = {'batchSize': 10, 'normalizationMethod': 12,'subbands':None,}

h5SetsLocation = utils.functions.getH5SetLocation(datasetName)
trainSamplesFilename = utils.functions.getDatasetLocation(datasetName, 'trainSamples',subdir=datafolderSubdir)

def clusterEmbeddings(compareModels):
    for model in compareModels:
        methodName = model['method']
        kClusters = model['kClusters']

        if 'epoch' in model.keys():
            embeddingFilename = os.path.join(os.path.join(model['modelDir'],'embedding'), 'embedding_epoch={}.pkl'.format(model['epoch']))
            clustersFilename = os.path.join(os.path.join(model['modelDir'],'kmeans_clusters'), 'clusters_epoch={}_k={}.pkl'.format(model['epoch'],kClusters))
            oddEvenClusteringFilename = os.path.join(os.path.join(model['modelDir'],'kmeans_clusters'), 'oddEven_clusters_epoch={}_k={}.pkl'.format(model['epoch'],kClusters))
        else:
            embeddingFilename = os.path.join(os.path.join(model['modelDir'],'embedding'), 'embedding_nFeatures={}.pkl'.format(model['nFeatures']))
            clustersFilename = os.path.join(os.path.join(model['modelDir'],'kmeans_clusters'), 'clusters_nFeatures={}_k={}.pkl'.format(model['nFeatures'],kClusters))
            oddEvenClusteringFilename = os.path.join(os.path.join(model['modelDir'],'kmeans_clusters'), 'oddEven_clusters_nFeatures={}_k={}.pkl'.format(model['nFeatures'],kClusters))
        os.makedirs(os.path.join(model['modelDir'],'kmeans_clusters'),exist_ok=True)
        if os.path.exists(clustersFilename)==False:
            print("Start k-means clustering for {}.".format(methodName))
            nFeaturesOrEpoch, embeddingX, savedHash = utils.functions.LoadEmbedding(embeddingFilename)
            clusters, centroids = clusterAlgorithms.kmeans(embeddingX, kClusters, returnCentroids = True)
            utils.functions.SaveClusters(clustersFilename, nFeaturesOrEpoch, clusters, centroids,savedHash)

        if os.path.exists(oddEvenClusteringFilename)==False:
            print("Start odd-even clustering for {}.".format(methodName))
            nFeaturesOrEpoch, embeddingX, savedHash = utils.functions.LoadEmbedding(embeddingFilename)

            oddEmbedding = embeddingX[::2]
            evenEmbedding = embeddingX[1::2]
            oddClusters, oddCentroids = clusterAlgorithms.kmeans(oddEmbedding, kClusters, returnCentroids = True)
            evenClusters, evenCentroids = clusterAlgorithms.kmeans(evenEmbedding, kClusters, returnCentroids = True)
            utils.functions.SaveOddEvenClusters(oddEvenClusteringFilename, nFeaturesOrEpoch, oddClusters, oddCentroids, evenClusters, evenCentroids, savedHash)
        print("(Odd-even) clusters exist for {}.".format(methodName))
            
def tsneEmbeddings(compareModels):
    for model in compareModels:
        methodName = model['method']
        tsnePerplexity = model['perplexity']
        experimentModelDir = os.path.join(model['modelDir'],"clusterPipelineData")
        os.makedirs(experimentModelDir,exist_ok=True)

        if 'epoch' in model.keys():
            embeddingFilename = os.path.join(os.path.join(model['modelDir'],'embedding'), 'embedding_epoch={}.pkl'.format(model['epoch']))
            tsneFilename = os.path.join(os.path.join(model['modelDir'],'tsne_features'), 'tsne_epoch={}_perplexity={}.pkl'.format(model['epoch'], tsnePerplexity))
        else:
            embeddingFilename = os.path.join(os.path.join(model['modelDir'],'embedding'), 'embedding_nFeatures={}.pkl'.format(model['nFeatures']))
            tsneFilename = os.path.join(os.path.join(model['modelDir'],'tsne_features'), 'tsne_nFeatures={}_perplexity={}.pkl'.format(model['nFeatures'],tsnePerplexity))
        
        if os.path.exists(tsneFilename):
            print("t-SNE features exist for {}.".format(methodName))
            continue
        
        print("Start tsne for {}.".format(methodName))
        nFeaturesOrEpoch, embedding, savedHash = utils.functions.LoadEmbedding(embeddingFilename)

        tsneFeatures = tsne.tsne(embedding, perplexity=tsnePerplexity)
        utils.functions.SaveTsneFeatures(tsneFilename, tsneFeatures, nFeaturesOrEpoch, savedHash)

class ClusteringPipeline():
    class Method:
        def __init__(self, method, modelName, modelSubdir, embeddingX, clusters,centroids, oddClusters, oddCentroids, evenClusters, evenCentroids, kClusters, tsneFeatures):
            self.method = method
            self.modelName = modelName
            self.modelSubdir = modelSubdir
            self.embeddingX = embeddingX
            self.clusters = clusters
            self.clusterCentroids = centroids
            self.oddClusters = oddClusters
            self.oddCentroids = oddCentroids
            self.evenClusters = evenClusters
            self.evenCentroids = evenCentroids
            self.kClusters = kClusters
            self.tsneFeatures = tsneFeatures

    def __init__(self, models,modelCollectionName=None, resultsLocation=None):
        self.pipelineModelDir = utils.functions.getModelLocation(experimentName)[0]

        if resultsLocation is None:
            self.plotsLocationIndividualModels = utils.functions.getPlotLocation(datasetName, os.path.join(experimentName))
        else:
            self.plotsLocationIndividualModels = resultsLocation

        if modelCollectionName is not None:
            if resultsLocation is None:
                self.plotsLocationCompareModels = utils.functions.getPlotLocation(datasetName, os.path.join(experimentName,modelCollectionName))
            else:
                self.plotsLocationCompareModels = os.path.join(resultsLocation,modelCollectionName)
            os.makedirs(self.plotsLocationCompareModels,exist_ok=True)

        start_time = time.time()
        self.loadDatasetData()
        self.methodNames = []
        self.method: Dict[str, ClusteringPipeline.Method] = {}
        for model in models:
            methodName, methodData = self.loadMethod(model)
            self.methodNames.append(methodName)
            self.method[methodName] = methodData
            modelPlotDir = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
            os.makedirs(modelPlotDir,exist_ok=True)

        self.finalClusters = -1*np.ones(len(self.method[methodName].clusters),dtype=np.uint32)
        self.lastCluster = -1
        # print("Loading dataset and models took: %s seconds" % (time.time() - start_time))

    def loadDatasetData(self):
        pipelineDatasetFilename = os.path.join(self.pipelineModelDir, 'pipelineDatasetData.pkl')

        if os.path.exists(pipelineDatasetFilename):
            
            with open(pipelineDatasetFilename, 'rb') as file:
                self.dataX, self.dataY, self.metadata, self.samplesHash = pickle.load(file)
            return

        print("No cache file available for the clustering pipeline. Load data from h5 files.")
        self.dataGenerator = utils.datasets.Generators.UniversalDataGenerator(h5SetsLocation, 'dimensionReduction', 'original', 8 ,trainSamplesFilename, dataSettings=dataSettings, cacheDataset=True, bufferAll=True, nSamples=6000)
        loadedMetadata = self.dataGenerator.getMetadata()
        self.metadata = list(zip(*loadedMetadata))
        self.samplesHash = self.dataGenerator.samplesHash
        
        dataX = []
        dataY = []
        for batchX, batchY in self.dataGenerator:
            dataX.extend(batchX)
            dataY.extend(batchY)
        self.dataX = np.asarray(dataX)
        self.dataY = np.asarray(dataY)
        
        with open(pipelineDatasetFilename, 'wb') as file:
            pickle.dump([self.dataX, self.dataY, self.metadata, self.samplesHash], file)

    # Functions for loading data
    def loadMethod(self, model):
        # Select reduction algorithm model
        methodName = model['method']

        if 'epoch' in model.keys():
            modelType = 'dlModel'
            nFeaturesOrEpoch = model['epoch']
            methodName += '_epoch={}'.format(nFeaturesOrEpoch)
        else:
            modelType = 'traditional'
            nFeaturesOrEpoch = model['nFeatures']
            methodName += '_nFeatures={}'.format(nFeaturesOrEpoch)
        methodName += '_k={}'.format(model['kClusters'])
        modelSubdir = methodName

        embeddingFilename = utils.functions.GetEmbeddingFilename(model['modelDir'],modelType,nFeaturesOrEpoch,valData=False)
        clustersFilename = utils.functions.GetClustersFilename(model['modelDir'],modelType,nFeaturesOrEpoch,model['kClusters'],'kmeans',valData=False)
        oddEvenClustersFilename = utils.functions.GetOddEvenClustersFilename(model['modelDir'],modelType,nFeaturesOrEpoch,model['kClusters'],'kmeans',valData=False)
        tsneFilename = utils.functions.GetTsneFilename(model['modelDir'],modelType,nFeaturesOrEpoch,model['perplexity'],valData=False)

        embeddingEpochOrFeatures, embedding, _ = utils.functions.LoadEmbedding(embeddingFilename, self.samplesHash)
        clustersEpochOrFeatures, clusters, centroids, _ = utils.functions.LoadClusters(clustersFilename, self.samplesHash)
        oddEvenClustersEpochOrFeatures, oddClusters, oddCentroids, evenClusters, evenCentroids, _ = utils.functions.LoadOddEvenClusters(oddEvenClustersFilename, self.samplesHash)
        tsneEpochOrFeatures, tsneFeatures, _ = utils.functions.LoadTsneFeatures(tsneFilename, self.samplesHash)

        if embeddingEpochOrFeatures != clustersEpochOrFeatures or embeddingEpochOrFeatures != tsneEpochOrFeatures or embeddingEpochOrFeatures != oddEvenClustersEpochOrFeatures :
            raise Exception("Epochs or features do not match")

        embedding = np.asarray(embedding)
        tsneFeatures = np.asarray(tsneFeatures)
        clusters = np.asarray(clusters)
        methodData = ClusteringPipeline.Method(model['method'],methodName, modelSubdir, embedding, clusters, centroids, oddClusters, oddCentroids, evenClusters, evenCentroids, model['kClusters'], tsneFeatures)
        return methodName, methodData

    # individual methods
    def compareClustersStatistics(self, kClusters, methodOneLabels, methodTwoLabels):
        nSamplesOne = []
        unions = []
        differences = []
        symmetricDifferences = []
        intersections = []
        jaccardSimilarities = []

        totalUnion = 0
        totalDifference = 0
        totalSymmetricDifference = 0
        totalIntersection = 0

        for cluster in range(kClusters):
            samplesMethodOne = set(np.where(methodOneLabels == cluster)[0])
            samplesMethodTwo = set(np.where(methodTwoLabels == cluster)[0])
            
            # number of samples in cluster of method one
            nSamplesOne.append(len(samplesMethodOne))

            # Calculate union, intersection and difference
            intersection = len(samplesMethodOne.intersection(samplesMethodTwo))
            union = len(samplesMethodOne.union(samplesMethodTwo))
            difference = len(samplesMethodOne.difference(samplesMethodTwo))
            symmetricDifference = len(samplesMethodOne.difference(samplesMethodTwo)) + len(samplesMethodTwo.difference(samplesMethodOne))
            unions.append(union)
            differences.append(difference)
            symmetricDifferences.append(symmetricDifference)
            intersections.append(intersection)
            totalUnion += union
            totalDifference += difference
            totalSymmetricDifference += symmetricDifference
            totalIntersection += intersection

            # Jacard similarity between the clusters of method one and remapped method two
            jaccardSimilarity = intersection / union if union != 0 else 0
            jaccardSimilarities.append(round(jaccardSimilarity,3))
        results = {}
        results['nSamples'] = nSamplesOne
        results['intersection'] = intersections
        results['intersection (%)'] = list(np.round(np.divide(intersections,nSamplesOne)*100).astype(int))
        results['union'] = unions
        results['difference'] = differences
        results['symmetric difference'] = symmetricDifferences
        results['Jaccard sim'] = jaccardSimilarities
        
        # Calculation over all samples instead of per cluster
        globalJaccardSimilarity = totalIntersection / totalUnion if totalUnion != 0 else 0

        results['global'] = {}
        results['global']['n samples'] = len(self.dataX)
        results['global']['intersection'] = totalIntersection
        results['global']['intersection (%)'] = int(round(100*totalIntersection/len(self.dataX),0))
        results['global']['union'] = totalUnion
        results['global']['difference'] = totalDifference
        results['global']['symmetric difference'] = totalSymmetricDifference
        results['global']['Jaccard sim'] = round(globalJaccardSimilarity,3)

        return results

    def oddEvenClustersStatistics(self, methodName, plot=False):
        kClusters = self.method[methodName].kClusters
        clusters = self.method[methodName].clusters

        # Statistics on original clustering
        sampleSilhouetteValue = silhouette_samples(self.method[methodName].embeddingX, self.method[methodName].clusters)
        silhouette_avg = silhouette_score(self.method[methodName].embeddingX, self.method[methodName].clusters)
        
        silhouetteScoresPerCluster = []
        for clusterIdx in range(kClusters):
            clusterSamples = np.where(clusters == clusterIdx)[0]
            clusterSilhouette = sampleSilhouetteValue[clusterSamples]
            clusterSilhouetteAvg = np.mean(clusterSilhouette)
            clusterSilhouetteStd = np.std(clusterSilhouette)
            clusterSilhouetteMin = np.min(clusterSilhouette)
            clusterSilhouetteMax = np.max(clusterSilhouette)
            silhouetteScoresPerCluster.append([len(clusterSamples), clusterSilhouetteAvg, clusterSilhouetteStd, clusterSilhouetteMin, clusterSilhouetteMax])

        # Odd-even clustering
        remappedOdd, oddContigency, oddContigencyRemapped = self.remapLabels(clusters[::2], self.method[methodName].oddClusters)
        remappedEven, evenContigency, evenContigenceRemapped = self.remapLabels(clusters[1::2], self.method[methodName].evenClusters)
        remappedClusters = np.zeros_like(clusters)
        remappedClusters[::2] = remappedOdd
        remappedClusters[1::2] = remappedEven
        comparisonResults = self.compareClustersStatistics(kClusters, clusters, remappedClusters)
        
        saveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        oddEvenStatisticsFileName = os.path.join(saveLocation, '# Statistics (with odd-even clustering).txt')
        with open(oddEvenStatisticsFileName, 'w') as file:
            file.write('------- Evaluation original clusters for {} -------\n'.format(methodName))
            silhouetteDf = pd.DataFrame(silhouetteScoresPerCluster, columns=['nSamples', 'Mean Silhouette', 'std', 'Min Silhouette', 'Max Silhouette'])
            file.write("Mean silhouette score: {}\n".format(silhouette_avg))
            file.write("Silhouette per cluster:\n")
            file.write(silhouetteDf.to_string())
            file.write('\n\n')

            silhouetteDf = silhouetteDf.sort_values(by='Mean Silhouette', ascending=True)#, inplace=True)
            file.write("Sorted by mean silhouett:\n")
            file.write(silhouetteDf.to_string())
            file.write('\n\n')


            file.write('------- Evaluation original clusters compared with odd-even clusters for {} -------\n'.format(methodName))
            columnNames = list(comparisonResults.keys())[:-1]
            localDict = dict(list(comparisonResults.items())[:-1])
            comparisonDf = pd.DataFrame(localDict, columns=columnNames)

            for metricName, metricValue in comparisonResults['global'].items():
                file.write("{}: {}\n".format(metricName, metricValue))
            file.write('\n')
            file.write("Comparison per clusters:\n")
            file.write(comparisonDf.to_string())
            file.write('\n\n')  

            file.write("Comparison per clusters, sorted by Jaccard sim:\n")
            comparisonDf = comparisonDf.sort_values(by='Jaccard sim', ascending=False)
            file.write(comparisonDf.to_string())
            file.write('\n\n')

        if plot:
            oddEvenRemappedContigency = contingency_matrix(clusters, remappedClusters)
            oddEvenRemappedContigency = oddEvenRemappedContigency / np.sum(oddEvenRemappedContigency, axis=1, keepdims=True)
            self.plotPurityMatrix(oddContigency, 'Contigency matrix before remapping odd clusters', 'all', 'odd clusters',saveLocation,True)
            self.plotPurityMatrix(oddContigencyRemapped, 'Contigency matrix after remapping odd clusters','all', 'odd clusters',saveLocation,True)
            self.plotPurityMatrix(evenContigency, 'Contigency matrix before remapping even clusters', 'all', 'even clusters',saveLocation,True)
            self.plotPurityMatrix(evenContigenceRemapped, 'Contigency matrix after remapping even clusters','all', 'even clusters',saveLocation,True)
            self.plotPurityMatrix(oddEvenRemappedContigency, 'Contigency matrix after remapping odd-even clusters','all', 'odd-even clusters',saveLocation,True)

    def plotSilhouetteTsneAndRfiTypes(self,methodName):
        # Calc silhouette scores for silhouette plot
        sampleSilhouetteValue = silhouette_samples(self.method[methodName].embeddingX, self.method[methodName].clusters)
        silhouette_avg = silhouette_score(self.method[methodName].embeddingX, self.method[methodName].clusters)

        # Calc dominant RFI types per sample for RFI phenomena plot. Map them to three colors
        rfiResults, rfiRatioResults, rfiCategories = utils.datasets.calcRfiTypesPerSample(self.dataY)
        rfiTypesPerSample = rfiRatioResults[:,1:]
        dominantCategoryPerSample = np.argmax(rfiTypesPerSample, axis=1)

        # Retrieve other data
        tsneFeatures = self.method[methodName].tsneFeatures
        kClusters = self.method[methodName].kClusters
        clusters = self.method[methodName].clusters

        plotLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        saveName = '{} silhouette plot with rfi phenomena'.format(self.method[methodName].method)

        # Create a subplot with 1 row and 3 columns        
        # Define the grid
        fig = plt.figure(figsize=(18, 6)) # l = 3*r
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 2])#, height_ratios=[2, 2, 2])

        # Create subplots
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

        ax1.set_ylim([0, len(tsneFeatures) + (kClusters + 1) * 10])
        y_lower = 10
        for i in range(kClusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sampleSilhouetteValue[clusters == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / kClusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.7,)
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i),fontsize='x-small')
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples 
        ax1.set_title("Silhouette plot.")
        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster label")
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks

        
        # 2nd Plot showing the actual clusters formed 
        colors = cm.nipy_spectral(clusters.astype(float) / len(np.unique(clusters)))
        colors=np.asarray(colors)
        ax2.scatter(tsneFeatures[:, 0], tsneFeatures[:, 1], marker=".", s=50, alpha=1, lw=0, c=colors, edgecolor="k")
        centers = []
        for i in range(len(np.unique(clusters))):
            center = np.mean(tsneFeatures[clusters == i], axis=0)
            centers.append(center)
        centers = np.asarray(centers)
        ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=0.7,s=200,edgecolor="k",)
        for i, c in enumerate(centers):
            ax2.annotate(str(i), (c[0], c[1]-0.5), fontsize=10, color='black', ha='center', va='center')
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.set_title("t-SNE visualization of the clustered data.")


        # 3th plot
        colorMap=['m','c','k']
        legend=rfiCategories[1:]
        clusters = dominantCategoryPerSample
        
        colors = []
        for cluster in clusters:
            colors.append(colorMap[cluster])
        colors=np.asarray(colors)
  
        for clusterIdx in np.unique(clusters):
            clusterSamples = np.where(clusters == clusterIdx)
            ax3.scatter(tsneFeatures[clusterSamples, 0], tsneFeatures[clusterSamples, 1], marker=".", s=50, alpha=1, lw=0, c=colors[clusterSamples], edgecolor="k", label=legend[clusterIdx])
        ax3.legend()

        ax3.set_xlabel("t-SNE 1")
        ax3.set_ylabel("t-SNE 2")
        ax3.set_title("t-SNE visualization of the RFI phenomena.")

        plt.savefig(os.path.join(plotLocation, saveName), dpi=300, bbox_inches='tight')
        plt.close()

    # Plotting individual methods
    def plotRfiTypesPerCluster(self, methodName):
        plotLocation = os.path.join(self.plotsLocationIndividualModels,methodName)
        os.makedirs(plotLocation,exist_ok=True)
        rfiResults, rfiRatioResults, rfiCategories = utils.datasets.calcRfiTypesPerSample(self.dataY)

        # Map each sample to one of three colors: RFI local in time, RFI local in frequency, or weak RFI
        rfiTypesPerSample = rfiRatioResults[:,1:]
        dominantCategoryPerSample = np.argmax(rfiTypesPerSample, axis=1)

        utils.plotter.tsneClusterScatter(self.method[methodName].tsneFeatures, plotLocation, plotTitle='{} dominant rfi phenomena per sample'.format(self.method[methodName].method), clusters=dominantCategoryPerSample, plotCenters=False, legend=rfiCategories[1:],colorMap=['m','c','k'])

        # make the line plot for the different categories
        rfiDistributionResults = []
        for clusterIdx in range(self.method[methodName].kClusters):
            clusterSamples = np.where(self.method[methodName].clusters == clusterIdx)[0]
            clusterResults = rfiRatioResults[clusterSamples]
            clusterResults = np.mean(clusterResults,axis=0)
            rfiDistribution = clusterResults
            rfiDistributionResults.append(rfiDistribution)            
        rfiDistributionResults = np.asarray(rfiDistributionResults)
        rfiDistributionResults = np.nan_to_num(rfiDistributionResults)

        nCategories = len(rfiCategories)
        # make subplots
        fig, ax = plt.subplots(1, nCategories-1, sharey=False,figsize=(25, 15))

        x = np.arange(0,nCategories)
        xLabels = rfiCategories

        # plot subplots and set xlimit
        for i in range(nCategories-1):
            for j in range(len(rfiDistributionResults)):
                ax[i].plot(x,rfiDistributionResults[j],label='Cluster {}'.format(j))
            ax[i].set_xlim([x[i],x[i+1]])
            ax[i].set_xticks(x[i:i+2],xLabels[i:i+2])
            ax[i].set_ylim([-0.02,1.02])
            ax[i].set_yticks([0,1],[0,1])

        clustersYstart = rfiDistributionResults[:,0]
        argSorted = np.argsort(clustersYstart)
        for j in range(len(rfiDistributionResults)):
            # if argsorted[j] is odd, plot on the left, else plot on the right
            # Get the index of argsort with value j
            argSortIndex = np.where(argSorted == j)[0][0]
            if argSortIndex % 2 == 0:
                xPos = -0.1
            else:
                xPos = -0.05
            ax[0].text(xPos,clustersYstart[j],'{}'.format(j))
        
        # set width space to zero
        plt.subplots_adjust(wspace=0)   
        dpi=300
        plt.savefig(os.path.join(plotLocation, "Rfi distribution model {}".format(methodName)), dpi=dpi, bbox_inches='tight')
        plt.close()
        
    def plotPurityMatrix(self, matrix, plotName,methodNameOne, methodNameTwo, plotLocation = None, plotTight = False):
        if plotLocation is None:
            plotLocation = self.plotsLocationCompareModels
        #plt.figure(figsize=(12, 10)).gca()
        plt.figure(figsize=(10, 8)).gca()
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

        plt.colorbar()
        plt.yticks(np.arange(matrix.shape[0]), range(matrix.shape[0]))
        plt.xticks(np.arange(matrix.shape[1]), range(matrix.shape[1]))
        
        plt.ylabel(methodNameOne + ' cluster')
        plt.xlabel(methodNameTwo + ' cluster')

        plt.tight_layout()

        width, height = matrix.shape

        for x in range(width):
            for y in range(height):
                if plotTight:
                    roundedValue = round(matrix[x][y],2)
                else:
                    roundedValue = round(matrix[x][y],3)
                if roundedValue == 0:
                    roundedValue = '0'
                elif roundedValue == 1:
                    roundedValue = '1'
                else:
                    roundedValue = str(roundedValue)

                # if roundedValue start with 0., remove the 0.
                if plotTight:
                    if roundedValue.startswith('0.'):
                        roundedValue = roundedValue[1:]
                plt.annotate(str(roundedValue), xy=(y, x), horizontalalignment='center',verticalalignment='center',fontsize='small')
        dpi=300
        plt.savefig(os.path.join(plotLocation, plotName), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plotComembershipScatter(self,methodOne, plotName, remappedLabels):
        centers = []
        for i in range(self.method[methodOne].kClusters):
            center = np.mean(self.method[methodOne].tsneFeatures[self.method[methodOne].clusters == i], axis=0)
            centers.append(center)
        centers = np.asarray(centers)

        plt.figure(figsize=(10, 10))

        labelIntersection = self.method[methodOne].clusters == remappedLabels[0]
        if len(remappedLabels)>0:
            for newLabelsIndex in range(1,len(remappedLabels)):
                tempIntersection = self.method[methodOne].clusters == remappedLabels[newLabelsIndex]
                labelIntersection = np.logical_and(labelIntersection, tempIntersection)

        labelDifference = labelIntersection == False

        colors = cm.nipy_spectral(self.method[methodOne].clusters.astype(float) / self.method[methodOne].kClusters)
        
        plt.scatter(self.method[methodOne].tsneFeatures[labelIntersection, 0], self.method[methodOne].tsneFeatures[labelIntersection, 1], marker=".", s=50, lw=0, alpha=1.0, c=colors[labelIntersection], edgecolor="k")
        plt.scatter(self.method[methodOne].tsneFeatures[labelDifference, 0], self.method[methodOne].tsneFeatures[labelDifference, 1], marker=".", s=50, lw=0, alpha=0.05, c=colors[labelDifference], edgecolor="k")
        
        plt.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)
        for i, c in enumerate(centers):
            plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        #plt.scatter(tsneFeatures[:, 0], tsneFeatures[:, 1],s=10, c=clusters, cmap='hsv', vmin=0, vmax=np.max(clusters))
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        dpi=300
        plt.savefig(os.path.join(self.plotsLocationCompareModels, plotName), dpi=dpi, bbox_inches='tight')
        plt.close()

    def plotTsneClusters(self, methodName, overrideClusters = None, saveLocation = None, saveName = None):
        if overrideClusters is None:
            clusters = self.method[methodName].clusters
        else:
            clusters = overrideClusters

        if saveLocation is None:
            saveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        
        if saveName is None:
            saveName = '{} kmeans clustering tsne'.format(self.method[methodName].method)

        utils.plotter.tsneClusterScatter(self.method[methodName].tsneFeatures, saveLocation, plotTitle=saveName, clusters=clusters, plotCenters=True)

    def plotTsneOddEvenClusters(self, methodName):
        oddClusters = self.method[methodName].oddClusters
        evenClusters = self.method[methodName].evenClusters
        tsneFeatures = self.method[methodName].tsneFeatures
        oddTsneFeatures = tsneFeatures[::2]
        evenTsneFeatures = tsneFeatures[1::2]

        # Remap the odd and even clusters to the original clusters
        clusters = self.method[methodName].clusters
        remappedOdd, _, _ = self.remapLabels(clusters[::2], oddClusters)
        remappedEven, _, _ = self.remapLabels(clusters[1::2], evenClusters)
        remappedClusters = np.zeros_like(clusters)
        remappedClusters[::2] = remappedOdd
        remappedClusters[1::2] = remappedEven

        saveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        oddSaveName = '{} kmeans clustering tsne odd clusters'.format(self.method[methodName].method)
        evenSaveName = '{} kmeans clustering tsne even clusters'.format(self.method[methodName].method)
        oddEvenSaveName = '{} kmeans clustering tsne remapped odd-even clusters'.format(self.method[methodName].method)

        utils.plotter.tsneClusterScatter(oddTsneFeatures, saveLocation, plotTitle=oddSaveName, clusters=remappedOdd, plotCenters=True)
        utils.plotter.tsneClusterScatter(evenTsneFeatures, saveLocation, plotTitle=evenSaveName, clusters=remappedEven, plotCenters=True)
        utils.plotter.tsneClusterScatter(tsneFeatures, saveLocation, plotTitle=oddEvenSaveName, clusters=remappedClusters, plotCenters=True)

    def plotTopCentroidsScatter(self, methodName, topN = 1):
        centroids = self.method[methodName].clusterCentroids
        clusters = self.method[methodName].clusters
        embedding = self.method[methodName].embeddingX
        tsneFeatures = self.method[methodName].tsneFeatures

        # For each cluster, calculate and sort the distance of the samples and the centroids
        distances = np.zeros((len(centroids), len(embedding)))
        sortedDistances = np.zeros((len(centroids), len(embedding)),dtype=int)
        for clusterIdx, centroid in enumerate(centroids):
            distances[clusterIdx] = np.linalg.norm(embedding - centroid, axis=1)
            sortedDistances[clusterIdx] = np.argsort(distances[clusterIdx])

        closestSamplesPerCluster = sortedDistances[:,:topN]
        
        # For the top n samples, convert them to images
        tsnePerCluster = []
        images = {}
        for clusterIdx, closestSamples in enumerate(closestSamplesPerCluster):
            samplesTsne = []
            for sampleIdx in closestSamples:
                images[sampleIdx] = utils.datasets.Generators.UniversalDataGenerator.ConvertToImage(self=None, dataX = self.dataX[sampleIdx])
                samplesTsne.append(tsneFeatures[sampleIdx]) 
            tsnePerCluster.append(samplesTsne)
        tsnePerCluster = np.asarray(tsnePerCluster)

        # Plotting parameters
        figureHeight = 12
        imageZoom = 0.3
        scatterInsetClearance = 20
        insetBorderClearance = 10

        imageWidth = 256*imageZoom
        imageHeight = 64*imageZoom

        xMin = np.min(tsneFeatures[:, 0]) - imageWidth - (scatterInsetClearance+insetBorderClearance)
        xMax = np.max(tsneFeatures[:, 0]) + imageWidth + (scatterInsetClearance+insetBorderClearance)
        yMin = np.min(tsneFeatures[:, 1]) - imageHeight - (scatterInsetClearance+insetBorderClearance)
        yMax = np.max(tsneFeatures[:, 1]) + imageHeight + (scatterInsetClearance+insetBorderClearance)

        # For each top-n samples, highlight the samples
        clusterTsneCenters = []
        clusterAngles = []
        for clusterIdx, closestSamples in enumerate(closestSamplesPerCluster):
            clusterSampleAngles = []
            for tsneIdx, sampleIdx in enumerate(closestSamples):
                sampleAngle = np.arctan2(tsnePerCluster[clusterIdx,tsneIdx, 1], tsnePerCluster[clusterIdx, tsneIdx, 0])
                clusterSampleAngles.append(sampleAngle)
            clusterAngles.append(np.mean(clusterSampleAngles))
            clusterTsneCenters.append(tsnePerCluster[clusterIdx,0])
        clusterTsneCenters = np.asarray(clusterTsneCenters)

        xLeft = xMin+imageWidth+insetBorderClearance
        xRight = xMax-imageWidth-insetBorderClearance
        yTop = yMax-imageHeight-insetBorderClearance
        yDown = yMin+imageHeight+insetBorderClearance

        nTop = int((xRight-xLeft)//imageWidth)
        nDown = nTop

        nRight = int((len(clusterAngles)-nTop-nDown)/2)
        nLeft = len(clusterAngles)-nTop-nDown - nRight

        yLeftSteps = np.linspace(yTop+imageHeight/2, yDown-imageHeight/2, nLeft+2)[1:-1]
        yRightSteps = np.linspace(yTop+imageHeight/2, yDown-imageHeight/2, nRight+2)[1:-1]
        xStepsTop = np.linspace(xLeft-imageWidth/2, xRight+imageWidth/2, nTop+2)[1:-1]
        xStepsDown = xStepsTop

        clusterPlotPos = []
        for leftIdx in range(nLeft): clusterPlotPos.append([xLeft, yLeftSteps[leftIdx],0])
        for topIdx in range(nTop): clusterPlotPos.append([xStepsTop[topIdx],yTop,1])
        for rightIdx in range(nRight): clusterPlotPos.append([xRight, yRightSteps[rightIdx],2])
        for downIdx in range(nDown): clusterPlotPos.append([xStepsDown[downIdx],yDown,3])
        clusterPlotPos = np.asarray(clusterPlotPos)

        distanceMatrix = np.zeros((len(clusterPlotPos), len(clusterTsneCenters)))
        for i, clusterPos in enumerate(clusterPlotPos):
            for j, clusterCenter in enumerate(clusterTsneCenters):
                distanceMatrix[j,i] = np.linalg.norm(clusterPos[0:2] - clusterCenter)
        row_ind, col_ind = linear_sum_assignment(distanceMatrix)
        clusterPlotPos = clusterPlotPos[col_ind]

        # Plot standard scatter plot
        figureWidth = figureHeight*(xMax-xMin)/(yMax-yMin)
        plt.figure(figsize=(figureWidth, figureHeight)) #plt.figure(figsize=(10, 8))
        ax = plt.axes()
        colors = cm.nipy_spectral(clusters.astype(float) / len(np.unique(clusters)))
        ax.scatter(tsneFeatures[:, 0], tsneFeatures[:, 1], marker=".", s=50, lw=0, c=colors, edgecolor="k")
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)

        for clusterIdx, closestSamples in enumerate(closestSamplesPerCluster):
            plotSide = clusterPlotPos[clusterIdx,2]
            if plotSide == 0:
                loc = 'center right'
            elif plotSide == 1:
                loc = 'lower center'
            elif plotSide == 2:
                loc = 'center left'
            elif plotSide == 3:
                loc = 'upper center'

            bboxX = (clusterPlotPos[clusterIdx,0]-xMin)/(xMax-xMin)
            bboxY = (clusterPlotPos[clusterIdx,1]-yMin)/(yMax-yMin)

            axins = zoomed_inset_axes(ax, zoom=imageZoom, loc=loc, bbox_to_anchor=(bboxX, bboxY), bbox_transform=ax.transAxes, borderpad=0)
            axins.imshow(images[closestSamples[0]])
            axins.set_xticks([])
            axins.set_yticks([])
            axins.grid(False)

            # Draw line from point to image
            ax.plot([clusterTsneCenters[clusterIdx, 0], clusterPlotPos[clusterIdx,0]], [clusterTsneCenters[clusterIdx, 1], clusterPlotPos[clusterIdx,1]], color='k', linewidth=1.0)

        ax.scatter(clusterTsneCenters[:, 0],clusterTsneCenters[:, 1],marker="o",c="white",alpha=0.7,s=200,edgecolor="k",)
        for i, c in enumerate(clusterTsneCenters):
            ax.annotate(str(i), (c[0], c[1]-0.5), fontsize=10, color='black', ha='center', va='center')

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        os.makedirs(os.path.join(self.plotsLocationIndividualModels, methodName), exist_ok=True)
        plt.savefig(os.path.join(self.plotsLocationIndividualModels, methodName, '{} closestClusters'.format(self.method[methodName].method)), dpi=300, bbox_inches='tight')
        plt.close()

        # Save the samples as images
        clusterSamplesPlotsDir = os.path.join(self.plotsLocationIndividualModels, methodName ,'topCentroids')
        os.makedirs(clusterSamplesPlotsDir,exist_ok=True)
        for clusterIdx, closestSamples in enumerate(closestSamplesPerCluster):
            for topIdx, sampleIdx in enumerate(closestSamples):
                uintImage = (images[sampleIdx]*255).astype(np.uint8)
                pilImage = Image.fromarray(uintImage)
                saveName = 'cluster={}_top={}.png'.format(clusterIdx, topIdx)
                pilImage.save(os.path.join(clusterSamplesPlotsDir, saveName))

    def plotTsneVisualization(self, methodName):
        saveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        magnitudeImage, phaseImage = utils.datasets.Generators.UniversalDataGenerator.ConvertToImage(self=None, dataX = self.dataX, calcPhase = True)
        utils.plotter.tsneVisualization(self.method[methodName].tsneFeatures,magnitudeImage,saveLocation,'{} tsne visualization'.format(self.method[methodName].method))
        utils.plotter.tsneVisualization(self.method[methodName].tsneFeatures,phaseImage,saveLocation,'{} tsne visualization phase'.format(self.method[methodName].method))       

    def plotSilhouettePlot(self, methodName):
        sampleSilhouetteValue = silhouette_samples(self.method[methodName].embeddingX, self.method[methodName].clusters)
        silhouette_avg = silhouette_score(self.method[methodName].embeddingX, self.method[methodName].clusters)

        saveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        saveName = '{} silhouette plot'.format(self.method[methodName].method)

        utils.plotter.tsneClustersWithSilhouette(self.method[methodName].tsneFeatures, saveLocation, saveName, self.method[methodName].clusters, self.method[methodName].kClusters, sampleSilhouetteValue, silhouette_avg, plotCenters=True)

    def plotClusterTimeline(self, methodName):
        # This function contains code from AOFlagger 3.3 by Offringa, A.R. found at
        # https://gitlab.com/aroffringa/aoflagger/
        # License: GPL-3.0

        nClusters = len(np.unique(self.method[methodName].clusters))

        timeStamps = []
        for sampleIndex, sample in enumerate(self.metadata):
            aipsMJD = sample[3]

            mjd = aipsMJD / (60.0 * 60.0 * 24.0)
            jd = mjd + 2400000.5
            time = np.fmod(jd + 0.5, 1.0) * 24.0
            secs = (time * 3600).astype(np.int32)
            timeStamps.append(secs)

        timeStamps = np.asarray(timeStamps)
        timeStamps -= np.min(timeStamps)
        nTimeSteps = np.max(timeStamps)+1

        clusterTimeCounts = np.zeros((nClusters,nTimeSteps),dtype=np.int32)
        for sampleIndex, cluster in enumerate(self.method[methodName].clusters):
            times = timeStamps[sampleIndex]
            clusterTimeCounts[cluster,times] += 1
        
        # plot the figure
        ax = plt.figure(figsize=(10, 10)).gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for clusterIndex in range(nClusters):
            clusterTimes = clusterTimeCounts[clusterIndex]
            plt.plot(np.arange(nTimeSteps), clusterTimes, label='class {}'.format(clusterIndex))

        plt.xlabel('time')
        plt.ylabel('counts')
        plt.savefig(os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir, 'clusters over time'), dpi=300, bbox_inches='tight')
        plt.close()
 
    def plotClusterSamplesDistribution(self, methodName):
        ax = plt.figure(figsize=(10, 10)).gca()
        clusters = self.method[methodName].clusters
        metadata = self.metadata
        subbands = self.getSubbandFromMetadata(metadata)
        uniqueClusters, samplesPerCluster = np.unique(clusters, return_counts=True)

        plt.scatter(clusters,subbands)

        # Calculate the probabilities for each cluster
        probabilities = samplesPerCluster / np.sum(samplesPerCluster)

        # Calculate the entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        print("Entropy for {} is {}".format(methodName, entropy))

        #plt.legend()
        plt.ylabel("subband")
        plt.xlabel("cluster")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        

        dpi=300
        plt.savefig(os.path.join(self.plotsLocationIndividualModels,methodName, "subband vs clusters method"), dpi=dpi, bbox_inches='tight')
        plt.close()

    # Comparing methods
    def plotCompareRfiTypeDistributions(self):
        rfiResults, rfiRatioResults, valueNames = utils.datasets.calcRfiTypesPerSample(self.dataY)
        
        methodsSamplesPerValue = {}
        for experimentIndex, experimentName in  enumerate(self.methodNames):
            methodName = self.method[experimentName].method
            _, samplesPerCluster = np.unique(self.method[experimentName].clusters, return_counts=True)

            # Calculate the rfi distributions for each cluster for this method
            rfiDistributionResults = []
            for clusterIdx in range(self.method[experimentName].kClusters):
                clusterSamples = np.where(self.method[experimentName].clusters == clusterIdx)[0]
                clusterResults = rfiRatioResults[clusterSamples]
                clusterResults = np.mean(clusterResults,axis=0)
                rfiDistributionResults.append(clusterResults)
            rfiDistributionResults = np.asarray(rfiDistributionResults)
            rfiDistributionResults = np.nan_to_num(rfiDistributionResults)

            samplesPerValue = []
            for valueIdx in range(rfiDistributionResults.shape[1]):
                samplesPerValue.append([value for value, count in zip(rfiDistributionResults[:,valueIdx], samplesPerCluster) for _ in range(count)])
            methodsSamplesPerValue[experimentName] = samplesPerValue

        for valueIdx, valueName in enumerate(valueNames):
            ax = plt.figure(figsize=(10, 6)).gca()
            for experimentName in self.methodNames:
                samplesPerValue = methodsSamplesPerValue[experimentName][valueIdx]
                plt.hist(samplesPerValue,bins=10, range=(0,1),label=experimentName, alpha=0.5, linewidth=3, edgecolor = 'black',histtype='stepfilled')#, density=False, histtype='step')

            plt.xlim(0,1)
            plt.ylabel('Number of samples')
            plt.xlabel('Ratio {}'.format(valueName))
            plt.legend()
            dpi=300
            plt.savefig(os.path.join(self.plotsLocationCompareModels, "Rfi distribution {}".format(valueName)), dpi=dpi, bbox_inches='tight')
            plt.close()

    def plotCoMemberClusters(self, thesisOnly = False):
        for indexOne, experimentNameOne in  enumerate(self.methodNames):
            methodNameOne = self.method[experimentNameOne].method # methodName is for example ica, or dino. Experimentname is with featers/epoch and k
            remappedLabels = []

            for indexTwo, experimentNameTwo in enumerate(self.methodNames):
                if indexOne == indexTwo: continue
                methodNameTwo = self.method[experimentNameTwo].method
                methodTwoLabels, normalizedContigency, newNormalizedContigency = self.remapLabels(self.method[experimentNameOne].clusters, self.method[experimentNameTwo].clusters)
                remappedLabels.append(methodTwoLabels)

                self.plotTsneClusters(experimentNameOne, self.method[experimentNameTwo].clusters, self.plotsLocationCompareModels, 't-SNE of {} with clusters of {}'.format(methodNameOne, methodNameTwo))
                if thesisOnly:
                    continue
                self.plotPurityMatrix(normalizedContigency, 'Contigency matrix before remapping for {} and {}'.format(methodNameOne, methodNameTwo), methodNameOne, methodNameTwo, plotTight=True)
                self.plotPurityMatrix(newNormalizedContigency, 'Contigency matrix after remapping for {} and {}'.format(methodNameOne, methodNameTwo),methodNameOne, methodNameTwo, plotTight=True)
                self.plotTsneClusters(experimentNameOne, methodTwoLabels, self.plotsLocationCompareModels, 't-SNE of {} by remapped clusters of {}'.format(methodNameOne, methodNameTwo))
            
            utils.plotter.tsneClusterScatter(self.method[experimentNameOne].tsneFeatures, self.plotsLocationCompareModels, plotTitle='t-SNE intersection {} and all other methods'.format(methodNameOne), clusters=self.method[experimentNameOne].clusters, plotCenters=True,remappedLabels=remappedLabels)
           
    def remapLabels(self, clustersOne, clustersTwo):
        # Create contingency matrix
        contingencyMatrix = contingency_matrix(clustersOne, clustersTwo)
        normalizedContigency = contingencyMatrix / np.sum(contingencyMatrix, axis=1, keepdims=True)

        # Use the Hungarian algorithm to find the optimal mapping
        row_ind, col_ind = linear_sum_assignment(-contingencyMatrix)

        # Map cluster indices
        remappedLabels = np.zeros_like(clustersTwo)
        for i, j in zip(row_ind, col_ind):
            remappedLabels[clustersTwo == j] = i

        # Create new contingency matrix and calculate co-membership
        newContingencyMatrix = contingency_matrix(clustersOne, remappedLabels)
        newNormalizedContigency = newContingencyMatrix / np.sum(newContingencyMatrix, axis=1, keepdims=True)

        return remappedLabels, normalizedContigency, newNormalizedContigency

    def compareMethodStatistics(self):
        results = {}
        for indexOne, experimentNameOne in  enumerate(self.methodNames):
            methodNameOne = self.method[experimentNameOne].method # methodName is for example ica, or dino. Experimentname is with featers/epoch and k
            results[experimentNameOne] = {}

            # Calculate the entropy
            _, samplesPerCluster = np.unique(self.method[experimentNameOne].clusters, return_counts=True)
            probabilities = samplesPerCluster / np.sum(samplesPerCluster)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            results[experimentNameOne]['entropy'] = entropy
           
            for indexTwo, experimentNameTwo in enumerate(self.methodNames):
                if indexOne == indexTwo: continue
                methodNameTwo = self.method[experimentNameTwo].method
                results[experimentNameOne][experimentNameTwo] = {}
                
                # Remap labels
                methodOneLabels = self.method[experimentNameOne].clusters
                methodTwoLabels, _, _ = self.remapLabels(methodOneLabels, self.method[experimentNameTwo].clusters)

                results[experimentNameOne][experimentNameTwo] = self.compareClustersStatistics(self.method[experimentNameTwo].kClusters,methodOneLabels, methodTwoLabels)   

        comembersResultFile = os.path.join(self.plotsLocationCompareModels, 'cluster similarities.txt')
        with open(comembersResultFile, 'w') as file:
            for indexOne, experimentNameOne in  enumerate(self.methodNames):
                methodNameOne = self.method[experimentNameOne].method # methodName is for example ica, or dino. Experimentname is with featers/epoch and k

                file.write("########################## {} ##########################\n".format(methodNameOne))
                file.write('Entropy: {}\n\n'.format(results[experimentNameOne]['entropy']))
       
                for indexTwo, experimentNameTwo in enumerate(self.methodNames):
                    if indexOne == indexTwo: continue
                    methodNameTwo = self.method[experimentNameTwo].method
                    columnNames = list(results[experimentNameOne][experimentNameTwo].keys())[:-1]
                    localDict = list(results[experimentNameOne][experimentNameTwo].items())[:-1]
                    localDict = dict(localDict)
                    comparisonDf = pd.DataFrame(localDict, columns=columnNames)

                    file.write("############# {}-{} #############\n".format(methodNameOne, methodNameTwo))
                    for metricName, metricValue in results[experimentNameOne][experimentNameTwo]['global'].items():
                        file.write("{}: {}\n".format(metricName, metricValue))
                    file.write('\n')
                    file.write("Comparison per clusters:\n")
                    file.write(comparisonDf.to_string())
                    file.write('\n\n')  
    
    # Plotting combined methods
    def plotAllMethodsSimilarities(self, plot):
        labelReferences = {}
        for indexOne, experimentNameOne in  enumerate(self.methodNames):
            methodNameOne = self.method[experimentNameOne].method # methodName is for example ica, or dino. Experimentname is with featers/epoch and k
            labelReferences[experimentNameOne] = {}
            for indexTwo, experimentNameTwo in enumerate(self.methodNames):
                if indexOne == indexTwo: continue
                
                # Remap labels
                methodOneLabels = self.method[experimentNameOne].clusters
                methodTwoLabels, _, _ = self.remapLabels(methodOneLabels, self.method[experimentNameTwo].clusters)
                labelReferences[experimentNameOne][experimentNameTwo] = methodTwoLabels

        similaritiesPerCluster = {}
        for indexOne, experimentNameOne in  enumerate(self.methodNames):
            methodNameOne = self.method[experimentNameOne].method # methodName is for example ica, or dino. Experimentname is with featers/epoch and k
            similaritiesPerCluster[methodNameOne] = []

            for clusterIdx in range(self.method[experimentNameOne].kClusters):
                clusterSamplesOne = np.where(self.method[experimentNameOne].clusters == clusterIdx)[0]
                clustersIntersection = clusterSamplesOne
                clustersUnion = clusterSamplesOne
                
                for indexTwo, experimentNameTwo in enumerate(self.methodNames):
                    if indexOne == indexTwo: continue
                    clusterSamplesTwo = np.where(labelReferences[experimentNameOne][experimentNameTwo] == clusterIdx)[0]
                    clustersIntersection = np.intersect1d(clustersIntersection, clusterSamplesTwo)
                    clustersUnion = np.union1d(clustersUnion, clusterSamplesTwo)
                
                similaritiesPerCluster[methodNameOne].append([clusterSamplesOne,clustersIntersection,clustersUnion])
        if plot:
            for indexOne, experimentNameOne in  enumerate(self.methodNames):
                methodNameOne = self.method[experimentNameOne].method
                for clusterIdx in range(self.method[experimentNameOne].kClusters):
                    clusterSamplesOne, clustersIntersection, clustersUnion = similaritiesPerCluster[methodNameOne][clusterIdx]
                    if len(clustersIntersection)<=1: continue

                    clusterDir = os.path.join(self.plotsLocationCompareModels, 'same clusters {}'.format(experimentNameOne), '{}'.format(clusterIdx))
                    os.makedirs(clusterDir, exist_ok=True)
                    images = []
                    for sampleIndex in clustersIntersection:
                        image = utils.datasets.Generators.UniversalDataGenerator.ConvertToImage(self=None, dataX = self.dataX[sampleIndex])
                        images.append(image)
                        pilImage = Image.fromarray((image*255).astype(np.uint8))
                        pilImage.save(os.path.join(clusterDir, '{}.png'.format(sampleIndex)))
   
                    sampleHeight = images[0].shape[0]
                    sampleWidth = images[0].shape[1]
                    summaryRows = 5
                    summaryCols = 3
                    rowSpacing = 10
                    columnSpacing = 10
                    
                    if len(clustersIntersection)>=summaryRows*summaryCols:
                        summaryImage = np.ones((summaryRows*sampleHeight+(summaryRows-1)*rowSpacing, summaryCols*sampleWidth+(summaryCols-1)*columnSpacing, 3))
                        for row in range(summaryRows):
                            startY = row*(sampleHeight+rowSpacing)
                            for col in range(summaryCols):
                                startX = col*(sampleWidth+columnSpacing)
                                image = images[row*summaryCols+col]
                                # flip y-axis with numpy
                                image = np.flipud(image)
                                summaryImage[startY:startY+sampleHeight, startX:startX+sampleWidth] = image
                        summaryImage = (summaryImage*255).astype(np.uint8)
                        pilImage = Image.fromarray(summaryImage)
                        pilImage.save(os.path.join(self.plotsLocationCompareModels, 'same clusters {}'.format(experimentNameOne), 'summary_{}.png'.format(clusterIdx)))

        # Write summary to file
        comembersResultFile = os.path.join(self.plotsLocationCompareModels, 'all methods similarities.txt')
        with open(comembersResultFile, 'w') as file:
            for indexOne, experimentNameOne in  enumerate(self.methodNames):
                methodNameOne = self.method[experimentNameOne].method # methodName is for example ica, or dino. Experimentname is with featers/epoch and k
                file.write("########################## {} ##########################\n".format(methodNameOne))
                similaritiesSummary = []
                for clusterSamplesOne, clustersIntersection, clustersUnion in similaritiesPerCluster[methodNameOne]:
                    similaritiesSummary.append([len(clusterSamplesOne),len(clustersIntersection),len(clustersUnion), len(clustersIntersection)/len(clustersUnion)])
                
                allComparisonDf = pd.DataFrame(similaritiesSummary, columns=['nSamples', 'nIntersect', 'nUnion', 'IoU'])
                file.write(allComparisonDf.to_string())
                file.write('\n\n') 

    def plotCompareSamplesPerCluster(self):
        for methodName in  self.methodNames:
            ax = plt.figure(figsize=(10, 10)).gca()
            clusters = self.method[methodName].clusters
            metadata = self.metadata
            subbands = self.getSubbandFromMetadata(metadata)

            plt.scatter(clusters,subbands)

            # Calculate the entropy
            probabilities = samplesPerCluster / np.sum(samplesPerCluster)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            print("Entropy for {} is {}".format(methodName, entropy))

            #plt.legend()
            plt.ylabel("subband")
            plt.xlabel("cluster")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            dpi=300
            plt.savefig(os.path.join(self.plotsLocationIndividualModels,methodName, "subband vs clusters method"), dpi=dpi, bbox_inches='tight')
            plt.close()

        plt.figure(figsize=(10, 10))
        for methodName in  self.methodNames:
            clusters = self.method[methodName].clusters
            metadata = self.metadata
            uniqueClusters, samplesPerCluster = np.unique(clusters, return_counts=True)
            plt.plot(np.sort(samplesPerCluster)[::-1], label=methodName)
    
        plt.legend()
        plt.xlabel("cluster (sorted)")
        plt.ylabel("number of samples")
        dpi=300
        plt.savefig(os.path.join(self.plotsLocationCompareModels, "Number of samples per cluster per method"), dpi=dpi, bbox_inches='tight')
        plt.close()

    # Functions for saving/writing data
    def saveSamples(self, methodName):
        saveClusters = self.method[methodName].clusters
        samplesSaveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir, 'clusters')

        if os.path.exists(os.path.join(samplesSaveLocation, '0')):
            print("Samples for {} already saved. Skip it.".format(methodName))
            return
        
        if -1 in saveClusters:
            print('Cluster -1 found.')

        for cluster in saveClusters:
            newDir = os.path.join(samplesSaveLocation, str(cluster))
            os.makedirs(newDir,exist_ok=True)
        
        imagesX = utils.datasets.Generators.UniversalDataGenerator.ConvertToImage(self=None, dataX = self.dataX)
        for sampleIndex, image in enumerate(imagesX):
            cluster = saveClusters[sampleIndex]
            datapointInfo = self.metadata[sampleIndex]

            correlationName = '{}-{}'.format(datapointInfo[0][0], datapointInfo[0][1])
            saveDir = os.path.join(samplesSaveLocation, str(cluster))
            savePath = os.path.join(saveDir,"{}_{}.png".format(sampleIndex, correlationName))
            if os.path.isfile(savePath):
                raise Exception("file already exists")

            colorImage = (image*255).astype(np.uint8)
            im = Image.fromarray(colorImage)
            im.save(savePath)

    def makeClusterTemplates(self, methodName, overrideClusters = None):
        if overrideClusters is None:
            saveClusters = self.method[methodName].clusters
            samplesSaveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        else:
            saveClusters = overrideClusters
            samplesSaveLocation = os.path.join(self.plotsLocationIndividualModels, 'combined')
        
        if -1 in saveClusters:
            print('Cluster -1 found.')

        meanImages = []
        meanEmbeddings = []
        for cluster in np.unique(saveClusters):
            sampleIndices = np.where(saveClusters==cluster)
            clusterSamples = self.dataX[sampleIndices]
            clusterEmbeddings = self.method[methodName].embeddingX[sampleIndices]

            # Calculate the image per cluster
            meanClusterImage = np.mean(clusterSamples,axis=0)
            meanClusterImage = (meanClusterImage-np.min(meanClusterImage))/(np.max(meanClusterImage)-np.min(meanClusterImage))
            colorImage = np.zeros((meanClusterImage.shape[0],meanClusterImage.shape[1],3))
            colorImage[:,:,0] = meanClusterImage[:,:,0]
            colorImage[:,:,1] = 0.5*meanClusterImage[:,:,1] + 0.5*meanClusterImage[:,:,2]
            colorImage[:,:,2] = meanClusterImage[:,:,3]
            meanImages.append(colorImage)

            # Calculate the histogram for the cluster's features
            meanEmbedding = np.mean(clusterEmbeddings,axis=0)
            meanEmbeddings.append(meanEmbedding)

        # From the sortedImages, create a 2d grid of the images as a numpy array, seperated by a small margin
        nCols = 4
        nRows = int(np.ceil(len(meanImages)/nCols))
        subHeight = meanImages[0].shape[0]
        subWidth = meanImages[0].shape[1]
        
        # make plt.Figure() with the figure height and width proportional to the grid height and width
        scaleFactor = 0.03
        fig, axs = plt.subplots(nCols, nRows, figsize=(subWidth*nCols*scaleFactor, scaleFactor*subHeight*nRows))
        for clusterIdx, ax in enumerate(axs.flat):
            if clusterIdx < len(meanImages):
                ax.set_title('cluster {}'.format(clusterIdx))
                flippedImage = np.flip(meanImages[clusterIdx],axis=0)
                ax.imshow(flippedImage)
            ax.axis('off')
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(os.path.join(samplesSaveLocation,'Cluster templates'),dpi=300)#,bbox_inches='tight')
        
        plt.figure()
        for clusterIdx, embedding in enumerate(meanEmbeddings):
            plt.plot(embedding, label='Cluster {}'.format(clusterIdx))
        plt.savefig(os.path.join(samplesSaveLocation,'Cluster mean embedding'),dpi=300)#,bbox_inches='tight')

    def getSubbandFromMetadata(self, metadata):
        defaultDataSettings = utils.models.defaultDataSettings

        baselines = []
        for sampleMetadata in metadata:
            fStart = sampleMetadata[2][0]
            fRel = fStart - defaultDataSettings['startFreq']
            baseline = int(fRel/defaultDataSettings['freqStepSize'])
            baselines.append(baseline)
        baselines = np.asarray(baselines)

        for notAvailable in [29,28,24,7]:# [7,24,28,29]
            decreaseSamples = np.where(baselines >= notAvailable)
            baselines[decreaseSamples] -= 1
        return baselines
    
    def getRelativeStartTimeFromMetadata(self, metadata):  
        startTimes = []
        for sampleMetadata in metadata:
            tStart = sampleMetadata[3][0]
            startTimes.append(tStart)
        startTimes = np.asarray(startTimes)
        startTimes -= np.min(startTimes)
        return startTimes.astype(np.int32)
    
    def plotClusterStatistics(self, methodName, overrideClusters = None):
        if overrideClusters is None:
            saveClusters = self.method[methodName].clusters
            samplesSaveLocation = os.path.join(self.plotsLocationIndividualModels, self.method[methodName].modelSubdir)
        else:
            saveClusters = overrideClusters
            samplesSaveLocation = os.path.join(self.plotsLocationIndividualModels, 'combined')
        
        if -1 in saveClusters:
            print('Cluster -1 found.')

        baselines = self.getSubbandFromMetadata(self.metadata)

        nClusters = len(np.unique(saveClusters))
        nBaselines = len(np.unique(baselines))
        
        clusterCounts = np.zeros((nClusters,nBaselines)) # there are also non-existing baselines
        for cluster in np.unique(saveClusters):
            sampleIndices = np.where(saveClusters==cluster)
            clusterBaselines = baselines[sampleIndices]
            clusters, counts =  np.unique(clusterBaselines,return_counts=True)
            clusterCounts[cluster,clusters] = counts

        ax = plt.figure().gca()
        xPos = np.arange(clusterCounts.shape[0])
        barWidth = 1
        colors = cm.nipy_spectral(np.unique(baselines).astype(float) / len(np.unique(baselines)))

        plt.bar(xPos, clusterCounts[:,0], color=colors[0], edgecolor='white', width=barWidth, log=True)
        for i in range(1,clusterCounts.shape[1]):
            bottom = np.sum(clusterCounts[:,:i],axis=1)
            plt.bar(xPos, clusterCounts[:,i], bottom = bottom, color=colors[i], edgecolor='white', width=barWidth, log=True)
        
        plt.xlabel('cluster')
        plt.ylabel('counts')
        plt.xticks(xPos)
        plt.tight_layout()
        plt.savefig(os.path.join(samplesSaveLocation,'baselines per cluster'),dpi=300)
        plt.close()

def IndividualModels(compareModels,plot=True, resultsLocation = None, thesisOnly = True):
    clustering = ClusteringPipeline(compareModels, resultsLocation=resultsLocation)

    # Analyze individual methods
    for model in compareModels:
        if 'nFeatures' in model:
            methodName = '{}_nFeatures={}_k={}'.format(model['method'], model['nFeatures'],model['kClusters'])
        else:
            methodName = '{}_epoch={}_k={}'.format(model['method'], model['epoch'],model['kClusters'])
        
        print("Analyzing {}".format(model['method']))

        clustering.oddEvenClustersStatistics(methodName, (plot and thesisOnly==False))
        
        if plot==False:
            continue

        clustering.plotTsneClusters(methodName)
        clustering.plotSilhouetteTsneAndRfiTypes(methodName)
        clustering.plotTsneVisualization(methodName)
        
        if thesisOnly:
            continue
        clustering.plotRfiTypesPerCluster(methodName)
        clustering.plotSilhouettePlot(methodName)
        clustering.plotTsneOddEvenClusters(methodName)
        clustering.saveSamples(methodName)
        clustering.plotClusterSamplesDistribution(methodName)
        clustering.plotTopCentroidsScatter(methodName)
        clustering.plotClusterStatistics(methodName)
        clustering.makeClusterTemplates(methodName)
        clustering.plotClusterTimeline(methodName)
        
def CompareModels(compareModels, modelCollectionName,plot=True, resultsLocation = None, thesisOnly = True):
    clustering = ClusteringPipeline(compareModels,modelCollectionName, resultsLocation)

    # Compare methods
    print("Comparing {}".format(modelCollectionName))
    clustering.compareMethodStatistics()
    clustering.plotAllMethodsSimilarities(plot)
    if plot == False:
        return
    clustering.plotCoMemberClusters(thesisOnly)
    if thesisOnly:
        return
    clustering.plotCompareRfiTypeDistributions()
    clustering.plotCompareSamplesPerCluster()