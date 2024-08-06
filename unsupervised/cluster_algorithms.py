import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering

def kmeansSearch(embeddingX, kValues, verbose=False):
    """
    Perform k-means clustering on the given data for multiple values of k.

    Parameters:
    embeddingX (array-like): The input data to be clustered.
    kValues (array-like): The list of values of k to be tested.
    verbose (bool, optional): Whether to display progress bar. Defaults to False.

    Returns:
    tuple: A tuple containing two arrays - inertias and silhouettes.
        - inertias (ndarray): The inertia values for each value of k.
        - silhouettes (ndarray): The silhouette scores for each value of k.
    """
    inertias = []
    silhouettes = []
    for k in tqdm(kValues, 'k-means search', disable=not verbose):
        kmeanModel = KMeans(n_clusters=k)
        clusters = kmeanModel.fit_predict(embeddingX)
        intertia = kmeanModel.inertia_
        silhouette = metrics.silhouette_score(embeddingX, clusters)
        inertias.append(intertia)
        silhouettes.append(silhouette)
        
    # Values to array
    inertias = np.asarray(inertias)
    silhouettes = np.asarray(silhouettes)

    return inertias, silhouettes

def kmeans(embeddingX, kClusters, returnCentroids=False):
    kmeanModel = KMeans(n_clusters=kClusters)
    clusters = kmeanModel.fit_predict(embeddingX)

    if returnCentroids:
        centroids = kmeanModel.cluster_centers_
        return clusters, centroids
    return clusters

def gmmSearch(embeddingX, nComponentsList):
    aics = []
    bics = []
    for nComponents in nComponentsList:
        gmm = GaussianMixture(n_components=nComponents)
        gmm.fit(embeddingX)
        probs = gmm.predict_proba(embeddingX)
        aics.append(gmm.aic(embeddingX))
        bics.append(gmm.bic(embeddingX))
        
    aics = np.asarray(aics)
    bics = np.asarray(bics)
    return aics, bics

def gmm(embeddingX, nComponents):
    gmm = GaussianMixture(n_components=nComponents)
    gmm.fit(embeddingX)
    clusters = gmm.predict(embeddingX)
    return clusters

def dbscanSearch(embeddingX):
    results = []
    for epsilonInt in range(10,60,2):#200, 10):
        epsilon = epsilonInt/100
        for min_samples in range(2,11,1):
            clusters = dbscan(embeddingX, eps=epsilon, min_samples=min_samples)
            clusters=np.asarray(clusters)
            uniqueClusters, countsPerCluster = np.unique(clusters,return_counts=True)
            if -1 in uniqueClusters:
                nNoise = countsPerCluster[uniqueClusters==-1][0]
            else:
                nNoise = 0
            if len(uniqueClusters) <= 1:
                silhouette = 1
            else:
                silhouette = metrics.silhouette_score(embeddingX, clusters)
                
            num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            print("espilon={} \t nSamples={} \tnClusters={}\t nNoise={} \t silhouette={}". format(epsilon, min_samples, num_clusters, nNoise,silhouette))
            results.append([epsilon, min_samples, num_clusters])
    return results

def aggloSearch(embeddingX):
    results = []
    for nClusters in range(5,46,5):
        for linkage in ['ward', 'complete', 'average', 'single']:
            for metric in ['euclidean', 'manhattan', 'cosine', 'l1', 'l2']:
                if linkage == 'ward' and metric != 'euclidean': continue
                clusters = agglomerative(embeddingX, nClusters, linkage, metric)
                silhouette = metrics.silhouette_score(embeddingX, clusters)
                results.append([nClusters, linkage, metric, silhouette])
                print("nClusters={} \t linkage={}     \tmetric={}        \t silhouette={:.4f}".format(nClusters, linkage, metric, silhouette))
    sortedResults = sorted(results, key=lambda x: x[-1], reverse=True)

def agglomerative(embeddingX, nClusters, linkage, metric):
    ward = AgglomerativeClustering(n_clusters=nClusters, affinity=metric, linkage=linkage)
    clusters = ward.fit_predict(embeddingX)
    return clusters

def dbscan(embeddingX, eps=0.23, min_samples=3):    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(embeddingX)
    clusters = dbscan.labels_
    return clusters