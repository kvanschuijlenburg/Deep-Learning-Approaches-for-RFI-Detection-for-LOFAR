import os
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import pandas as pd
from scipy.stats import shapiro, ranksums, sem
from scipy.stats import t as sciT
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import utils as utils
import gan.ganExperiments as ganExperiments

trainingMetrics = [['epoch_accuracy', 'accuracy']]
trainingLosses = [['epoch_gen_loss', 'generator loss'],
                  ['epoch_disc_loss', 'discriminator loss']]

dataSources = ['train', 'validation']
uncertainty = 'std'

datasetName = "LOFAR_L2014581 (recording)"
datasetSubDir = 'dataset250k'

dpi=300

# from utils.model_utils import defaultDataSettings
#datasetSubDir = defaultDataSettings['datasetSubDir']

## Evaluating training results
# Reading log files from training
def readSingleLog(log_dir):
    size_guidance = {
        'scalars': 0,
        'images': 0,
        'audio': 0,
        'histograms': 0,
        'compressedHistograms': 0,
        'tensors': 1050,
    }
    event_acc = event_accumulator.EventAccumulator(log_dir,size_guidance)
    event_acc.Reload()

    # Get scalar data
    loggedData = {}
    for tag in event_acc.Tags()['tensors']:
        if not 'epoch' in tag:
            continue
        loggedData[tag] = {}
        events = event_acc.Tensors(tag)
        for event in events:
            tensorValue = float(tf.make_ndarray(event.tensor_proto))
            if event.step not in loggedData[tag]:
                loggedData[tag][event.step]=tensorValue
            else:
                print("error: multiple epochs for the same variable")
    return loggedData

def readMultiLogs(experimentDir, rejectPartial=True):
    minEpochs = 10000
    maxEpochs = 0

    logs = {}
    for type in ['train', 'validation']:
        multiResults = []
        for runDir in os.scandir(experimentDir):
            if not runDir.name.startswith('run_'): continue
            for logDir in os.scandir(runDir.path):
                if not os.path.isdir(logDir.path): continue
                if not logDir.name.startswith('run_'): continue

                exerimentLog = os.path.join(logDir.path, type)
                # A run can consist of multiple log files. Merge them
                runResults = {}
                nEpochs = 0
                for logFile in os.scandir(exerimentLog):
                    fileResults = readSingleLog(logFile.path)
                    for tagName, tagSteps in fileResults.items():
                        if not tagName in runResults:
                            runResults[tagName] = {}
                        for step, stepData in tagSteps.items():
                            if step in runResults[tagName]:
                                print("error: multiple steps in tag")
                            runResults[tagName][step] = stepData
                            if (step+1) > nEpochs: nEpochs = (step+1)
                if nEpochs > maxEpochs:
                    maxEpochs = nEpochs
                if nEpochs < minEpochs: 
                    minEpochs = nEpochs
                multiResults.append(runResults)

        if rejectPartial:
            completeRuns = []
            for runIdx, runResults in enumerate(multiResults):
                for tagName, metrics in runResults.items():
                    if len(metrics) == maxEpochs:
                        completeRuns.append(runResults)
                        break
            multiResults = completeRuns
            minEpochs = maxEpochs

        # Convert the results for tag/step/repetitions
        typeResults = {}
        for tagName, _ in runResults.items():
            typeResults[tagName] = []
            for step in range(minEpochs):
                multiValues = []
                for singleResults in multiResults:
                    multiValues.append(singleResults[tagName][step])
                typeResults[tagName].append(multiValues)
            typeResults[tagName] = np.asarray(typeResults[tagName])
        
        logs[type] = typeResults
    return logs

def readAllLogs(experimentNames, logRoot, plotSteps=False):
    # Read all log files
    experimentLogs = {}
    stepsPerEpoch = {}
    for experiment in experimentNames:
        if 'exp' in experiment.keys():
            ganDataSettings, ganModelSettings, dinoModelSettings = ganExperiments.GetExperiment(experiment['exp'],False)
            ganName = utils.models.getGanModelName(ganModelSettings, ganDataSettings, dinoModelSettings)
            experimentDir = os.path.join(logRoot,ganName)
            if plotSteps:
                if ganDataSettings['nTrainingSamples'] is None:
                    raise Exception("Number of steps cannot be retrieved without training samples")
                stepsPerEpoch[experiment['name']] = int(np.floor(ganDataSettings['nTrainingSamples']/ganDataSettings['batchSize']))
        else:
            if plotSteps:
                raise Exception("Number of steps cannot be retrieved without experiment number")
            experimentDir = os.path.join(logRoot,experiment['dir'])
        experimentLogs[experiment['name']] = readMultiLogs(experimentDir)
    
    return experimentLogs, stepsPerEpoch

## Evaluating models on last epoch
# Reading log files from evaluating last epoch
def readEvaluationLogs(experimentNames,logRoot,datasets):
    experimentLogs = {}
    for experiment in experimentNames:
        if 'exp' in experiment.keys():
            ganDataSettings, ganModelSettings, dinoModelSettings = ganExperiments.GetExperiment(experiment['exp'],False)
            ganName = utils.models.getGanModelName(ganModelSettings, ganDataSettings, dinoModelSettings)
            experimentDir = os.path.join(logRoot,ganName)
        else:
            experimentDir = os.path.join(logRoot,experiment['dir'])
        
        experimentLogs[experiment['name']] = {}
        for dataset in datasets:
            samplesHash = ""
            experimentLogs[experiment['name']][dataset] = {}
            for run in range(5):
                runDir = os.path.join(experimentDir,'run_{}'.format(run))
                if dataset == 'train':
                    evaluationFilename = os.path.join(runDir ,'evaluation.pkl')
                elif dataset == 'val':
                    evaluationFilename = os.path.join(runDir ,'evaluation_val.pkl')
                elif dataset == 'valTest':
                    evaluationFilename = os.path.join(runDir ,'evaluation_valTest.pkl')
                elif dataset == 'test':
                    evaluationFilename = os.path.join(runDir ,'evaluation_test.pkl')

                if not os.path.exists(evaluationFilename):
                    raise Exception("No metrics found on {} dataset for {}. Predict model first".format(dataset, experiment['name']))
                
                with open(evaluationFilename, 'rb') as file:
                    [metricsResults, loadedSamplesHash] = pickle.load(file)
                if samplesHash == "":
                    samplesHash = loadedSamplesHash
                elif samplesHash != loadedSamplesHash:
                    raise Exception("Hashes of samples do not match")
                experimentLogs[experiment['name']][dataset][run] = metricsResults
    return experimentLogs

def calcEvaluationMetrics(experimentNames,datasets,experimentLogs):
    experimentResults = {}
    for experiment in experimentNames:       
        experimentResults[experiment['name']] = {} 
        for dataset in datasets:
            runResults = []
            for run in range(5):
                metricsResults = experimentLogs[experiment['name']][dataset][run]
                jaccardSim = []
                diceSim = []
                accuracy = []
                precision = []
                recall = []
                f1Score = []

                tp=0
                fp=0
                tn=0
                fn=0
                for [sampleTp,sampleFp,sampleTn,sampltFn] in metricsResults:
                    tp += sampleTp
                    fp += sampleFp
                    tn += sampleTn
                    fn += sampltFn

                jaccardSim.append(tp/(tp+fp+fn))
                diceSim.append(2*tp/(2*tp+fp+fn))
                accuracy.append((tp+tn)/(tp+fp+tn+fn))
                precision.append(tp/(tp+fp+1e-7))
                recall.append(tp/(tp+fn+1e-7))
                f1Score.append(2*tp/(2*tp+fp+fn+1e-7))

                jaccardSim = np.mean(jaccardSim)
                diceSim = np.mean(diceSim)
                accuracy = np.mean(accuracy)
                precision = np.mean(precision)
                recall = np.mean(recall)
                f1Score = np.mean(f1Score)
                runResults.append([accuracy,precision,recall,f1Score,jaccardSim,diceSim])
            
            runResults = np.asarray(runResults).T
            experimentResults[experiment['name']][dataset] = runResults #np.stack([meanRunResults,stdRunResults],axis=1)
    metricNames = ['Accuracy', 'Precision', 'Recall', 'F1 score','Jaccard', 'Dice']
    return experimentResults,metricNames

def calcEvaluationStatistics(experimentNames, datasets, experimentResults,metricNames, alternative):
    # Compare statistics between experiments
    statisticsResults = {}
    for experimentOne in experimentNames:
        statisticsResults[experimentOne['name']] = {} 
        for dataset in datasets:
            statisticsResults[experimentOne['name']][dataset] = {}
            for experimentTwo in experimentNames:
                if experimentOne['name'] == experimentTwo['name']: continue
                metricsPvalues = []
                for metricIdx, metricName in enumerate(metricNames):
                    experimentOneMetricPerRun = experimentResults[experimentOne['name']][dataset][metricIdx]
                    experimentTwoMetricPerRun = experimentResults[experimentTwo['name']][dataset][metricIdx]

                    ranksumTest = ranksums(experimentOneMetricPerRun, experimentTwoMetricPerRun, alternative=alternative)#='greater')
                    ranksumValue = round(ranksumTest[0],4)
                    ranksumP = round(ranksumTest[1],6)
                    tableText = 'z={}, p={}'.format(ranksumValue, ranksumP)
                    metricsPvalues.append(tableText)


                statisticsResults[experimentOne['name']][dataset][experimentTwo['name']] = metricsPvalues
    return statisticsResults

def CompareModels(experimentNames, datasets=['valTest', 'val','test'], logLocation = None, plotLocation = None):#plotAccuracyGraph=False):
    logRoot = os.path.join(utils.functions.modelsLocation,'gan')
    if logLocation is not None:
        os.makedirs(logLocation, exist_ok=True)
    if plotLocation is not None:
        os.makedirs(plotLocation, exist_ok=True)
    # Read all log files
    experimentLogs = readEvaluationLogs(experimentNames,logRoot,datasets)
    experimentResults,metricNames = calcEvaluationMetrics(experimentNames,datasets,experimentLogs)

    if plotLocation is not None:
        for dataset in datasets:
            plt.figure(figsize=(10,6))
            ax = plt.axes()

            accuracies = []
            standardDeviations = []
            for experiment in experimentNames:  
                tableRow = [experiment['name']]
                assert metricNames[0]=='Accuracy'
                experimentAccuracies = experimentResults[experiment['name']][dataset][0]
                accuracyMean = np.mean(experimentAccuracies)
                accuracyStd = np.std(experimentAccuracies)
                accuracies.append(accuracyMean)
                standardDeviations.append(accuracyStd)
            accuracies = np.asarray(accuracies)
            standardDeviations = np.asarray(standardDeviations)

            xValues = [500,1000,2000,3000,5000,6000]
            plt.plot(xValues, accuracies)#, label = name)
            plt.fill_between(xValues, accuracies - standardDeviations, accuracies + standardDeviations, alpha=0.3)

            plt.xticks(xValues)
            plt.xlabel('Training set size')
            plt.ylabel('Accuracy')
            
            plt.savefig(os.path.join(plotLocation,'accuracy graph {} set'.format(dataset)), dpi=dpi, bbox_inches='tight')
            plt.close()

    if logLocation is not None:
        experimentStatisticsTwoSided = calcEvaluationStatistics(experimentNames, datasets, experimentResults,metricNames, alternative='two-sided')
        experimentStatisticsGreater = calcEvaluationStatistics(experimentNames, datasets, experimentResults,metricNames, alternative='greater')

        statisticsFilepath = os.path.join(logLocation, 'statistics_model.txt')
        with open(statisticsFilepath, 'w') as statisticsFile:
            for dataset in datasets:
                resultsTable = []
                for experiment in experimentNames:  
                    tableRow = [experiment['name']]
                    for metric in experimentResults[experiment['name']][dataset]:
                        metricMean = round(np.mean(metric),4)
                        metricStd = round(np.std(metric),7)

                        tableRow.append(metricMean)
                        tableRow.append(metricStd)
                    resultsTable.append(tableRow)
                
                columnNames = ['Experiment']
                for metricName in metricNames:
                    columnNames.append(metricName)
                    columnNames.append('std')

                df = pd.DataFrame(resultsTable, columns =columnNames)
                
                statisticsFile.write('------- Evaluation on {} dataset -------\n'.format(dataset))
                statisticsFile.write(df.to_string(header=True, index=False))
                statisticsFile.write('\n\n')

                for experimentOne in experimentNames:  
                    pValueColumnNames = ['{} two-sided'.format(experimentOne['name'])]
                    for metricName in metricNames:
                        pValueColumnNames.append(metricName)
                    
                    pValueTable = []
                    for experimentTwo in experimentNames: 
                        if experimentOne['name'] == experimentTwo['name']: continue
                        pValues = experimentStatisticsTwoSided[experimentOne['name']][dataset][experimentTwo['name']]
                        pValueTable.append([experimentTwo['name']] + pValues)
                    pValueDf = pd.DataFrame(pValueTable, columns =pValueColumnNames)
                    statisticsFile.write(pValueDf.to_string(header=True, index=False))
                    statisticsFile.write('\n\n')

                for experimentOne in experimentNames:  
                    pValueColumnNames = ['{} greater than'.format(experimentOne['name'])]
                    for metricName in metricNames:
                        pValueColumnNames.append(metricName)
                    
                    pValueTable = []
                    for experimentTwo in experimentNames: 
                        if experimentOne['name'] == experimentTwo['name']: continue
                        pValues = experimentStatisticsGreater[experimentOne['name']][dataset][experimentTwo['name']]
                        pValueTable.append([experimentTwo['name']] + pValues)
                    pValueDf = pd.DataFrame(pValueTable, columns =pValueColumnNames)
                    statisticsFile.write(pValueDf.to_string(header=True, index=False))
                    statisticsFile.write('\n\n')

def plotTrainingGraphs(experimentLogs,stepsPerEpoch, plotLocation,plotSteps = False, plotMaxEpoch=False, plotZoom = True):
    for dataSource in dataSources:
        for tagName, metricsName in trainingMetrics:
            nStepsMax=0
            nStepsMin = 100000000
            plt.figure(figsize=(10,6))
            ax = plt.axes()
            if plotZoom:
                axins = zoomed_inset_axes(ax, zoom=5, loc='center right')

            for name, logs in experimentLogs.items():
                tagLogs = logs[dataSource][tagName]

            allLastMeans = []
            xAxisValues = []
            for name, logs in experimentLogs.items():
                tagLogs = logs[dataSource][tagName]

                tagMeans = np.mean(tagLogs,axis=1)
                tagStds = np.std(tagLogs,axis=1)
                allLastMeans.append(tagMeans[-1])

                tagCi = [sciT.interval(alpha=0.95, df=len(tempData)-1, loc=np.mean(tempData), scale=sem(tempData)) for tempData in tagLogs]
                tagCi = np.asarray(tagCi)

                if plotSteps:
                    nStepsPerEpoch = stepsPerEpoch[name]
                    nStepsOrEpochs = len(tagLogs)*stepsPerEpoch[name]
                    xValues = range(1,nStepsOrEpochs+1,nStepsPerEpoch)
                else:
                    nStepsOrEpochs = len(tagLogs)
                    xValues = range(1,nStepsOrEpochs+1)

                # Plot the mean and the std around the mean
                if nStepsOrEpochs>nStepsMax: 
                    nStepsMax = nStepsOrEpochs
                if nStepsOrEpochs<nStepsMin:
                    nStepsMin = nStepsOrEpochs
                xAxisValues.extend(list(xValues))
                if plotZoom:
                    axins.plot(xValues, tagMeans)
                ax.plot(xValues, tagMeans, label = name)
                
                if uncertainty == 'std': 
                    if plotZoom:
                        axins.fill_between(xValues, tagMeans - tagStds, tagMeans + tagStds, alpha=0.3)
                    ax.fill_between(xValues, tagMeans - tagStds, tagMeans + tagStds, alpha=0.3)
                    
                elif uncertainty == 'ci':
                    if plotZoom:
                        axins.fill_between(xValues, tagCi[:,0], tagCi[:,1], alpha=0.3)
                    ax.fill_between(xValues, tagCi[:,0], tagCi[:,1], alpha=0.3)
                    
            xAxisValues = np.unique(np.asarray(xAxisValues))

            if uncertainty == 'ci':
                ax.set_title("{} {} with {}".format(dataSource,metricsName, uncertainty))
            else:
                ax.set_title("{} {}".format(dataSource,metricsName))
            
            if plotMaxEpoch==False:
                xAxisValues = [value for value in xAxisValues if value<=nStepsMin]
            xAxisValues = np.asarray(sorted(xAxisValues))

            nXticks = 8
            xTicks = np.linspace(0,xAxisValues[-1],nXticks).astype(np.int32)
            xTicks[0]=1
            if xTicks[-1] <= 1000:
                roundBase = 5
            else:
                digits = int(math.log10(xTicks[-1]))+1
                roundBase = 10**(digits-2)
            xTicks = (roundBase * np.round(xTicks[:-1]/roundBase)).astype(np.int32)
            lastTick = 5 * round(xAxisValues[-1]/5)
            xTicks = np.append(xTicks, lastTick)

            ax.set_xlim((1,xAxisValues[-1]))
            ax.set_xticks(xTicks)
            if plotSteps:
                ax.set_xlabel('Steps')
            else:
                ax.set_xlabel('Epoch')
            ax.legend(loc='lower right')
            ax.set_ylim((0.85,1.0))

            if plotZoom:
                # Zoom in on the last proportion of the plot
                rightProportion = 0.0
                widthProportion = 0.12
                x2 = xAxisValues[-1]-xAxisValues[-1]*rightProportion
                x1 = x2-xAxisValues[-1]*widthProportion      

                # select y-range for zoomed region
                meanValue = np.mean(allLastMeans)
                y1 = np.mean(meanValue)-0.005
                y2 = np.mean(meanValue)+0.005

                # Make the zoom-in plot:
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                plt.xticks(visible=False)
                plt.yticks(visible=False)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

            if plotSteps:
                saveFileName = os.path.join(plotLocation, "steps {} {} {}".format(dataSource,metricsName,uncertainty))
            else:
                saveFileName = os.path.join(plotLocation, "epochs {} {} {}".format(dataSource,metricsName,uncertainty))
            plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')
            plt.close()

def plotTrainingLossGraphs(experimentLogs,stepsPerEpoch, plotLocation, plotMaxEpoch=False):
    for dataSource in dataSources:
        
        for tagName, lossName in trainingLosses:
            plt.figure(figsize=(10,6))
            ax = plt.axes()

            for name, logs in experimentLogs.items():
                tagLogs = logs[dataSource][tagName]

            allLastMeans = []
            xAxisValues = []
            modelIdx = 0
            for name, logs in experimentLogs.items():
                tagLogs = logs[dataSource][tagName]

                tagMeans = np.mean(tagLogs,axis=1)
                tagStds = np.std(tagLogs,axis=1)
                allLastMeans.append(tagMeans[-1])
    
                nStepsOrEpochs = len(tagLogs)
                xValues = range(1,nStepsOrEpochs+1)

                # Plot the mean and the std around the mean
                xAxisValues.extend(list(xValues))
                ax.plot(xValues, tagMeans, label = name)
                ax.fill_between(xValues, tagMeans - tagStds, tagMeans + tagStds, alpha=0.3)

                window_size = 15
                moving_avg = []
                for epoch in xValues:
                    if epoch < window_size:
                        moving_avg.append(tagMeans[epoch])
                    else:
                        moving_avg.append(sum(tagMeans[epoch-window_size:epoch]))
                threshold = 0.005
                thresholded = (np.abs(np.subtract(moving_avg[1:],moving_avg[:-1]))<threshold).astype(np.int32)
                thresholded = list(thresholded)
                thresholded.insert(0,0)
                thresholded = np.asarray(thresholded)

                if modelIdx % 2 == 0:
                    plotBaseHeight = tagMeans[-1] + 0.1
                else:
                    plotBaseHeight = tagMeans[-1] - 0.1

                ax.plot(xValues, (plotBaseHeight-0.05)+thresholded*0.05, label = name + ' moving avg')

                # Get postive flank
                difference = thresholded[1:]-thresholded[:-1]
                positiveFlanks = np.where(difference==1)[0]

                plotHeight =  plotBaseHeight
                for flankIdx, flank in enumerate(positiveFlanks):
                    if flankIdx >0:
                        if flank-positiveFlanks[flankIdx-1] < 10:
                            plotHeight += 0.01
                        else:
                            plotHeight = plotBaseHeight
                    ax.text(flank, plotHeight, str(flank), fontsize=8)
                modelIdx += 1

            xAxisValues = np.unique(np.asarray(xAxisValues))
            ax.set_title("{} {}".format(dataSource,lossName))
            
            xAxisValues = np.asarray(sorted(xAxisValues))

            nXticks = 8
            xTicks = np.linspace(0,xAxisValues[-1],nXticks).astype(np.int32)
            xTicks[0]=1
            if xTicks[-1] <= 1000:
                roundBase = 5
            else:
                digits = int(math.log10(xTicks[-1]))+1
                roundBase = 10**(digits-2)
            xTicks = (roundBase * np.round(xTicks[:-1]/roundBase)).astype(np.int32)
            lastTick = 5 * round(xAxisValues[-1]/5)
            xTicks = np.append(xTicks, lastTick)

            ax.set_xlim((1,xAxisValues[-1]))
            ax.set_xticks(xTicks)
            ax.set_xlabel('Epoch')
            ax.legend(loc='lower right')

            meanValue = np.mean(allLastMeans)
            ax.set_ylim((meanValue-1.0,meanValue+1.0))

            saveFileName = os.path.join(plotLocation, "Loss {} {}".format(dataSource,lossName))
            plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')
            plt.close()
       
def calcTrainingStatistics(experimentLogs, experimentNames, logLocation):
    statisticsFilepath = os.path.join(logLocation, 'statistics_training.txt')
    with open(statisticsFilepath, 'w') as statisticsFile:
        # Calc the statistics/properties per experiment
        lastEpochValues = {}
        lastEpochStatistics = []
        for dataSource in dataSources:
            lastEpochValues[dataSource] = {}
            for tagName, metricsName in trainingMetrics:
                lastEpochValues[dataSource][metricsName] = {}

                for name, logs in experimentLogs.items():
                    tagLogs = logs[dataSource][tagName]
                    lastEpoch = tagLogs[-1]
                    
                    # Calculate some statistics
                    tagMeans = np.mean(tagLogs,axis=1)
                    tagStds = np.std(tagLogs,axis=1)

                    tagCi = [sciT.interval(alpha=0.95, df=len(tempData)-1, loc=np.mean(tempData), scale=sem(tempData)) for tempData in tagLogs]
                    tagCi = np.asarray(tagCi)

                    if len(lastEpoch)<4:
                        shapiroP = 100
                    else:
                        shapiroTest = shapiro(lastEpoch)
                        shapiroP = shapiroTest[1]

                    # Collect data
                    lastEpochValues[dataSource][metricsName][name] = lastEpoch
                    lastEpochStatistics.append([name,dataSource,metricsName, round(tagMeans[-1],4),round(tagStds[-1],7),shapiroP])
        
        df = pd.DataFrame(lastEpochStatistics, columns =['Experiment name','data source','metrics','mean','std', 'Shapiro P *'])
        df = df.sort_values(['data source','metrics'])
        
        statisticsFile.write('------- Evaluation of the last epoch per experiment -------\n')
        statisticsFile.write(df.to_string(header=True, index=False))
        statisticsFile.write('\n* A Shapiro P-value of 100 means there are not enough runs to calculte the p-value \n\n')

        # Compare statistics between experiments
        for dataSource in dataSources:
            statisticsFile.write('\n------- {} data: Wilcoxon rank-sum test -------\n'.format(dataSource))
            for tagName, metricsName in trainingMetrics:
                testMatrix = []
                compareEpochs = lastEpochValues[dataSource][metricsName]
                for firstIndex, firstFileProperties in enumerate(experimentNames):
                    rowResults = []
                    firstExperiment = compareEpochs[firstFileProperties['name']]
                    for secondIndex, secondFileProperties in enumerate(experimentNames):
                        if firstIndex == secondIndex: 
                            rowResults.append(0.0)
                            continue
                        secondExperiment = compareEpochs[secondFileProperties['name']]
                        
                        firstHigher = np.mean(firstExperiment)>np.mean(secondExperiment)
                        if firstHigher:
                            ranksumTest = ranksums(firstExperiment, secondExperiment, alternative='greater')
                        else:
                            ranksumTest = ranksums(firstExperiment, secondExperiment, alternative='less')
                        ranksumValue = ranksumTest[0]
                        ranksumP = ranksumTest[1]
                        rowResults.append(ranksumP)
                    testMatrix.append(rowResults)

                testDf = pd.DataFrame(testMatrix, index =[fileProperty['name'] for fileProperty in experimentNames],columns =[fileProperty['name'] for fileProperty in experimentNames])
                statisticsFile.write('{} {}:\n'.format(dataSource,metricsName))
                statisticsFile.write(testDf.to_string(header=True, index=True))
                statisticsFile.write('\n\n')

def EvaluateTraining(experimentNames, plotLocation, plotZoom=True,plotMinEpochs=False):
    logRoot = os.path.join(utils.functions.modelsLocation,'gan')
    experimentLogs,stepsPerEpoch = readAllLogs(experimentNames, logRoot,plotSteps=True)
    calcTrainingStatistics(experimentLogs,experimentNames,plotLocation)
    plotTrainingLossGraphs(experimentLogs, stepsPerEpoch,plotLocation)
    plotTrainingGraphs(experimentLogs, stepsPerEpoch,plotLocation,plotSteps=True, plotMaxEpoch=True, plotZoom = plotZoom)
    plotTrainingGraphs(experimentLogs, stepsPerEpoch,plotLocation,plotSteps=False, plotMaxEpoch=True, plotZoom = plotZoom) 
    
    if plotMinEpochs:
        plotLocMinSteps = os.path.join(plotLocation, 'min_steps')
        os.makedirs(plotLocMinSteps, exist_ok=True)
        plotTrainingGraphs(experimentLogs, stepsPerEpoch,plotLocMinSteps,plotSteps=True, plotMaxEpoch=False, plotZoom = plotZoom)
        plotTrainingGraphs(experimentLogs, stepsPerEpoch,plotLocMinSteps,plotSteps=False, plotMaxEpoch=False, plotZoom = plotZoom)




# Function for loading data TODO: make the functions below more compact/combine them
def loadData(model, run, valData):
    modelsRoot = os.path.join(utils.functions.modelsLocation, 'gan')

    # Load model properties and data
    name = model['name']
    experiment = model['exp']
    ganDataSettings, ganModelSettings, dinoModelSettings = ganExperiments.GetExperiment(experiment,False)
    ganName = utils.models.getGanModelName(ganModelSettings, ganDataSettings, dinoModelSettings)
    experimentDir = os.path.join(modelsRoot,ganName,'run_{}'.format(run))
    if valData:
        predictionsDir = os.path.join(experimentDir ,'predictions_val')
    else:
        predictionsDir = os.path.join(experimentDir ,'predictions')
    os.makedirs(predictionsDir, exist_ok=True)
    predictionsFilename = os.path.join(predictionsDir, 'predictions_last_epoch.pkl')
    if not os.path.exists(predictionsFilename):
        raise Exception("No predictions for {}. Load model and predict dataset first".format(predictionsFilename))
    with open(predictionsFilename, 'rb') as file:
        [epoch, dataX, colorImages,trueLabels, modelPredictions, loadedHash] = pickle.load(file)
    return name, dataX, colorImages,trueLabels,modelPredictions

def loadSampleData(dataX,trueLabels,colorImages, modelPredictions, sampleIdx):
    # Get all data and flip the y-axis
    sampleX = dataX[sampleIdx]
    colorImage = colorImages[sampleIdx]
    
    # Restore padding first row
    sampleX[0,:,:] = np.zeros((sampleX.shape[1],sampleX.shape[2]))
    colorImage[0,:,:] = np.zeros((colorImage.shape[1],colorImage.shape[2]))

    # Flip all y-axes
    sampleX = np.flipud(sampleX)
    colorImage = np.flipud(colorImage)
    trueLabel = np.flipud(trueLabels[sampleIdx,:,:,1]).astype(bool)
    modelPrediction = np.flipud(modelPredictions[sampleIdx,:,:,1])

    sampleXReal = sampleX[:,:,:4]
    sampleXImag = sampleX[:,:,4:]
    sampleXAmp = np.sqrt(sampleXReal**2 + sampleXImag**2)/3
    # 0 for XX, 1 for XY, 2 for YX, 3 for YY
    sampleXX = sampleXAmp[:,:,0]
    sampleXY = sampleXAmp[:,:,1]
    sampleYX = sampleXAmp[:,:,2]
    sampleYY = sampleXAmp[:,:,3]

    threshold = 0.5
    prediction = modelPrediction > threshold
    falsePositives = np.logical_and(prediction, np.logical_not(trueLabel))
    falseNegatives = np.logical_and(np.logical_not(prediction), trueLabel)

    return colorImage, sampleXX, sampleXY, sampleYX, sampleYY, trueLabel, prediction, falsePositives, falseNegatives

# Plot visualizations
def makeComparisonImage(saveLocation, dataX, referenceLabels, predictedLabels, referenceName, predictedName,nSamples):
    for sampleIdx in range(nSamples):
        sampleXX = dataX[sampleIdx][:,:,0]
        sampleXY = dataX[sampleIdx][:,:,1]
        sampleYX = dataX[sampleIdx][:,:,2]
        sampleYY = dataX[sampleIdx][:,:,3]
        sampleLabel = referenceLabels[sampleIdx,:,:,1]>0.5
        samplePrediction = predictedLabels[sampleIdx,:,:,1]>0.5
        sampleFp = np.logical_and(samplePrediction, np.logical_not(sampleLabel))
        sampleFn = np.logical_and(np.logical_not(samplePrediction), sampleLabel)

        fig,ax = plt.subplots(nrows=4, ncols=2,figsize=(14,9))

        # Plot original data
        ax[0,0].imshow(sampleXX)
        ax[0,0].set_title("XX")
        #ax[0,0].set_ylabel("Frequency channel")
        ax[1,0].imshow(sampleXY)
        ax[1,0].set_title("XY")
        ax[2,0].imshow(sampleYX)
        ax[2,0].set_title("YX")
        ax[3,0].imshow(sampleYY)
        ax[3,0].set_title("YY")

        ax[0,1].imshow(sampleLabel)
        ax[0,1].set_title(referenceName)
        ax[1,1].imshow(samplePrediction)
        ax[1,1].set_title(predictedName)
        ax[2,1].imshow(sampleFp)
        ax[2,1].set_title("FP")
        ax[3,1].imshow(sampleFn)
        ax[3,1].set_title("FN")

        for rowIdx, axRow in enumerate(ax):
            for columnIdx, axCol in enumerate(axRow):
                axCol.invert_yaxis()
                if rowIdx == 3:
                    axCol.set_xlabel("Time (s)")
                else:
                    axCol.set_xticks([])
                
                if columnIdx == 0:
                    axCol.set_ylabel("channel")
                else:
                    axCol.set_yticks([])

        plt.savefig(os.path.join(saveLocation, "{}.png".format(sampleIdx)), dpi=600, bbox_inches='tight')
        plt.close()

def VisualizePredictions(models, run=0, valData = True, resultsLocation = None):
    if resultsLocation is None:
        plotsLocation = utils.functions.getPlotLocation(datasetName, 'comparePredictions')
    else:
        plotsLocation = resultsLocation

    for model in models:
        modelPlotsLocation = os.path.join(plotsLocation, model['name'])
        os.makedirs(modelPlotsLocation, exist_ok=True)
        name, dataX, colorImages,trueLabels,modelPredictions = loadData(model, run, valData)
        makeComparisonImage(modelPlotsLocation, dataX, trueLabels, modelPredictions, "AOFlagger", model['name'], nSamples = 50)
 
def renderSamplePrediction(baseView,labelOptions,backgroundTransparancy, dataX, trueLabels, colorImages, modelPredictions, sampleIdx, upscaleFactor=1, nPaddingPixels=0):
    colorImage, sampleXX, sampleXY, sampleYX, sampleYY, trueLabel, prediction, falsePositives, falseNegatives = loadSampleData(dataX,trueLabels,colorImages, modelPredictions, sampleIdx)

    if baseView == "COLOR":
        baseImage = colorImage.copy()
        # add transparance axis
        baseImage = np.concatenate([baseImage, np.ones((baseImage.shape[0],baseImage.shape[1],1))], axis=-1)
    elif baseView == "XX":
        baseImage = sampleXX.copy()
    elif baseView == "XY":
        baseImage = sampleXY.copy()
    elif baseView == "YX":
        baseImage = sampleYX.copy()
    elif baseView == "YY":
        baseImage = sampleYY.copy()
    elif baseView == "MAX":
        stacked = [sampleXX,sampleXY,sampleYX,sampleYY]
        baseImage = np.max(stacked, axis=0)
    elif baseView == "MEAN":
        stacked = [sampleXX,sampleXY,sampleYX,sampleYY]
        baseImage = np.mean(stacked, axis=0)
    
    if baseView != "COLOR":
        viridis_colormap = cm.viridis
        baseImage = viridis_colormap(baseImage.astype(np.float32))

    colorAoFlagger = [0,1,0,1]
    colorPrediction = [0,0,1,1]
    colorFalsePositives = [1,0,0,1]
    colorFalseNegatives = [1,0,1,1]
    
    if "Top image" in labelOptions:
        labelsImage = baseImage.copy()
        labelsImage[:,:,3] *= backgroundTransparancy
    else:
        labelsImage = np.zeros_like(baseImage)

    if "AOFlagger" in labelOptions:
        labelsImage[trueLabel] = colorAoFlagger
    if "Predicted" in labelOptions:
        labelsImage[prediction] = colorPrediction
    if "FP" in labelOptions:
        labelsImage[falsePositives] = colorFalsePositives
    if "FN" in labelOptions:
        labelsImage[falseNegatives] = colorFalseNegatives
    if "Compare" in labelOptions:
        bothTrue = np.logical_and(prediction, trueLabel)
        labelsImage[bothTrue] = colorAoFlagger
        labelsImage[falsePositives] = colorFalsePositives
        labelsImage[falseNegatives] = colorFalseNegatives

    if 'Difference' in labelOptions:
        bothTrue = np.logical_and(prediction, trueLabel)
        #labelsImage[bothTrue] = colorAoFlagger
        labelsImage[falsePositives] = colorFalsePositives
        labelsImage[falseNegatives] = colorFalsePositives


    # Add padding around base image and upsize
    baseImage = np.pad(baseImage, ((nPaddingPixels,nPaddingPixels),(nPaddingPixels,nPaddingPixels),(0,0)), mode='constant', constant_values=0)
    baseImage = Image.fromarray((baseImage*255).astype(np.uint8))
    baseImage = baseImage.resize((baseImage.width*upscaleFactor, baseImage.height*upscaleFactor), Image.NEAREST)

    # Add padding around labels image and upsize
    labelsImage = np.pad(labelsImage, ((nPaddingPixels,nPaddingPixels),(nPaddingPixels,nPaddingPixels),(0,0)), mode='constant', constant_values=0)
    labelsImage = Image.fromarray((labelsImage*255).astype(np.uint8))
    labelsImage = labelsImage.resize((labelsImage.width*upscaleFactor, labelsImage.height*upscaleFactor), Image.NEAREST)

    return np.asarray(baseImage), np.asarray(labelsImage)
