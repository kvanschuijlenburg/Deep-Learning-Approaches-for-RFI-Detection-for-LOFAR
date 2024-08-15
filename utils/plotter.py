import math
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm

import utils as utils
from PIL import Image

matplotlib.use('agg')

class h5Plotter:
    def __init__(self, datasetName, plotLocation, padRowZero = True):
        self.plotLocation = plotLocation
        self.datasetName = datasetName
        self.padRowZero = padRowZero
        
        self.initArrays()

    def initArrays(self):
        self.startFrequencies = []
        self.channelFrequencies = []
        self.times = []
        self.observations = []
        self.labels = []
        self.setNames = []
        self.columnTitles = []
        self.rowTitles = []
        
    def calcDistance(self, posA, posB):
        distance = round(math.sqrt(np.sum(np.square(posA-posB))))
        return distance

    def plotMultiplot(self, antennaA, antennaB, layout : {}, nSubbands=None, normalizationMethod=12):
        self.normalizationMethod = normalizationMethod

        nColumns = 0
        for plotType in layout:
            if plotType['type'] != 'transparantBackground':
                nColumns += 1
        
        # Load data and prepare plotter
        self.initArrays()
        error = self.loadData(antennaA,antennaB, nSubbands=nSubbands)
        if error: return
        self.orderSubbands()
        self.initPlot(nColumns=nColumns)

        # Plot columns
        for columnIndex, parameters in enumerate(layout):
            plotType = parameters['type']
            
            mode='amplitude'
            polarization='rgb'
            polarizations= utils.constants.linearRepresentation

            if 'mode' in parameters: mode=parameters['mode']
            if 'polarization' in parameters: polarization=parameters['polarization']
            if 'polarizations' in parameters: polarizations=parameters['polarizations']

            if plotType=='visibility':
                self.plotStackedVisibilities(column=columnIndex, mode=mode, polarization=polarization)
            elif plotType =='histogram':
                self.plotStackedHistograms(column=columnIndex, mode=mode, polarizations=polarizations)
            elif plotType == 'labels':
                self.plotStackedLabels(column=columnIndex)
            elif plotType == 'transparantBackground':
                self.plotTransparantBackground(column=columnIndex)
            else:
                raise Exception("Invalid plot type. Please choose between visibility, histogram or labels")
        if nColumns > 0 : 
            self.plotplot()

    def loadData(self, antennaA, antennaB, nSubbands=None):
        h5SetsLocation = utils.functions.getH5SetLocation(self.datasetName)
        h5Sets = [filename for filename in os.listdir(h5SetsLocation) if filename.endswith('.h5')]
        for number, dataset in enumerate(h5Sets):
            if nSubbands is not None:
                if number >= nSubbands: break
            datasetFileName = os.path.join(h5SetsLocation, dataset)
            warning, metaData, chan_freq, time, observation, label = utils.functions.getObservation(datasetFileName,antennaA,antennaB)
            if warning: return True
            
            self.setNames.append(dataset)
            self.startFrequencies.append(chan_freq[0])
            self.channelFrequencies.extend(chan_freq)
            self.times.append(time)

            
            # Normalize
            scaledComplex = []
            for polarIndex in range(observation.shape[2]):
                scaledComplexPolar = utils.functions.normalizeComplex(observation[:,:,polarIndex], standardize=False)
                scaledComplexPolar[0,:] = scaledComplexPolar[1,:]
                scaledComplex.append(scaledComplexPolar)
            observation = np.asarray(scaledComplex).transpose(1,2,0).astype(np.complex64)

            self.labels.append(label)
            self.observations.append(observation)

        self.observations = utils.datasets.NormalizeComplex(np.asarray(self.observations),normalizationMethod=self.normalizationMethod)
        self.metaData = metaData

        return False
            
    def orderSubbands(self):
        self.observations=np.asarray(self.observations)
        self.labels=np.asarray(self.labels)

        self.freqStart = self.channelFrequencies[0]
        self.freqStop = self.channelFrequencies[-1]
        self.freqStepSize = self.channelFrequencies[1]-self.channelFrequencies[0]
        subBandwidth = self.freqStepSize*self.observations.shape[1]
        nPossibleSubbands = math.ceil((self.freqStop-self.freqStart)/subBandwidth) # 53
        indexedFrequencies = list(enumerate(self.startFrequencies))
        self.subbandOrder = []
        
        for subband in range(nPossibleSubbands):
            expectedStartFreq = self.freqStart + subband*subBandwidth
            subbandIndex = -1
            for pair in indexedFrequencies:
                if pair[1]==expectedStartFreq:
                    subbandIndex = pair[0]
            self.subbandOrder.append([subbandIndex,expectedStartFreq])

    def initPlot(self, nColumns):
        if nColumns > 0:
            inchPixelRatio = 0.5
            height = int((self.freqStop-self.freqStart)/self.freqStepSize)+1
            width = self.observations.shape[2]
            figureHeight = int(inchPixelRatio*height/10)
            figureWidth = int(inchPixelRatio*width*nColumns/10)
            nRows = len(self.subbandOrder)
            self.fig, self.axs = plt.subplots(nRows, nColumns, figsize=(figureWidth,figureHeight))

    def plotStackedVisibilities(self, column=1, mode='amplitude', polarization='rgb'):
        height = int((self.freqStop-self.freqStart)/self.freqStepSize)+1
        width = self.observations.shape[2]
        subBandHeight = self.observations.shape[1]

        nRows = len(self.subbandOrder)
        self.columnTitles.append('Visibility: ' + mode +', ' +polarization)

        for index,frequencyPair in enumerate(self.subbandOrder):
            currentStartFreq = frequencyPair[1]
            currentStopFreq = currentStartFreq + subBandHeight*self.freqStepSize

            if frequencyPair[0] == -1:
                # subplot does not exist
                plotImage = np.zeros((subBandHeight, width,3))
            else:
                observation = self.observations[frequencyPair[0]]
                if mode == 'amplitude':
                    plotImage = utils.datasets.Generators.UniversalDataGenerator.ConvertToImage(None,observation)
                else:
                    angle = np.angle(observation)
                    angleNormalized = (angle+math.pi)/(2*math.pi)

                    plotImage = np.zeros((angleNormalized.shape[0],angleNormalized.shape[1],3))
                    plotImage[:,:,0] = angleNormalized[:,:,0]
                    plotImage[:,:,1] = np.clip(0.5*angleNormalized[:,:,1] + 0.5*angleNormalized[:,:,2], 0.0, 1.0)
                    plotImage[:,:,2] = angleNormalized[:,:,3]
                if self.padRowZero:
                    plotImage[0] = plotImage[1]
            
            if nRows==1:
                ax = self.axs[column]
            else:
                currentRow = nRows-index-1
                ax = self.axs[currentRow, column]
            testMax = np.max(plotImage)
            if testMax > 1.0:
                print(testMax)
            ax.imshow(plotImage, aspect='auto', origin='lower', extent=[0, width, currentStartFreq, currentStopFreq], interpolation='nearest')
            ax.set_xticklabels([])

    def plotStackedLabels(self, column=2):
        height = int((self.freqStop-self.freqStart)/self.freqStepSize)+1
        width = self.observations.shape[2]
        subBandHeight = self.observations.shape[1]

        nRows = len(self.subbandOrder)
        self.columnTitles.append('Flags')

        for index,frequencyPair in enumerate(self.subbandOrder):
            currentStartFreq = frequencyPair[1]
            currentStopFreq = currentStartFreq + subBandHeight*self.freqStepSize

            if frequencyPair[0] == -1:
                # subplot does not exist
                labelsPlot = np.zeros((subBandHeight, width))
            else:
                labelsPlot = self.labels[frequencyPair[0]]            
            
            if nRows==1:
                ax = self.axs[column]
            else:
                currentRow = nRows-index-1
                ax = self.axs[currentRow, column]

            ax.imshow(labelsPlot, aspect='auto', origin='lower', extent=[0, width, currentStartFreq, currentStopFreq], interpolation='nearest')
            ax.set_xticklabels([])

    def plotStackedHistograms(self, column, mode, polarizations):
        width = self.observations.shape[2]
        subBandHeight = self.observations.shape[1]
        nPolarizations = self.observations.shape[3]
        nRows = len(self.subbandOrder)
        
        polarsText = ', '.join(polarizations)
        self.columnTitles.append('Histrogram: ' + mode +', ' + polarsText)
        for index,frequencyPair in enumerate(self.subbandOrder):
            if nRows==1:
                ax = self.axs[column]
            else:
                currentRow = nRows-index-1
                ax = self.axs[currentRow, column]

            if frequencyPair[0] == -1:
                # subplot does not exist
                ax.imshow(np.zeros((subBandHeight, width,3)), aspect='auto', origin='lower', interpolation='nearest')
            else:
                for polarization in polarizations:
                    polarIndex = utils.constants.linearRepresentation.index(polarization)
                    if mode=='amplitude':
                        selectedData = np.abs(self.observations[frequencyPair[0],:,:,polarIndex])
                        flattened = np.asarray(selectedData.flatten())
                        ax.hist(flattened,bins=100,alpha=0.1)
                    elif mode=='phase':
                        selectedData = np.angle(self.observations[frequencyPair[0],:,:,polarIndex])
                        flattened = np.asarray(selectedData.flatten())
                        ax.hist(flattened, bins=100, alpha=0.1)#,bins=100,alpha=0.1)
                    else: raise Exception('Invalid mode. Please choose amplitude or phase')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(0.0,1.0)
            
        if nRows > 1:
            for i in range(nRows):
                self.axs[i, 1].get_shared_x_axes().join(self.axs[i, 1], self.axs[i, 0])
    
    def plotTransparantBackground(self, column):
        height = int((self.freqStop-self.freqStart)/self.freqStepSize)+1
        width = self.observations.shape[2]
        subBandHeight = self.observations.shape[1]

        self.columnTitles.append('tranparant background')

        transparantBackground = []
        transparantBackgroundWhiteRfi = []

        backgroundValues = []
        RrfiValues = []

        for index,frequencyPair in enumerate(self.subbandOrder):
            currentStartFreq = frequencyPair[1]
            currentStopFreq = currentStartFreq + subBandHeight*self.freqStepSize

            if frequencyPair[0] == -1:
                # subplot does not exist
                plotImage = np.zeros((subBandHeight, width,3))
            else:
                observation = self.observations[frequencyPair[0]]
                labelsPlot = self.labels[frequencyPair[0]]
                
                backgroundValues.append(np.mean(np.abs(observation[1:][labelsPlot[1:]==False])))
                RrfiValues.append(np.mean(np.abs(observation[1:][labelsPlot[1:]==True])))
                
                plotImage = utils.datasets.Generators.UniversalDataGenerator.ConvertToImage(None,observation)
                if self.padRowZero:
                    plotImage[0] = plotImage[1]

                zeroImage = np.zeros((subBandHeight, width,4))
                
                transBackgroundImage = zeroImage.copy()
                whiteRfiImage = zeroImage.copy()
                
                transBackgroundImage[labelsPlot==True,3] = 1
                transBackgroundImage[labelsPlot==True,:3] = plotImage[labelsPlot==True]
                whiteRfiImage[labelsPlot==True] = [1,1,1,1]

                transparantBackground.extend(transBackgroundImage)
                transparantBackgroundWhiteRfi.extend(whiteRfiImage)
        
        transparantBackground = np.asarray(transparantBackground)
        transparantBackgroundWhiteRfi = np.asarray(transparantBackgroundWhiteRfi)

        # save as PIL image
        transparantBackground = (transparantBackground*255).astype(np.uint8)
        transparantBackgroundWhiteRfi = (transparantBackgroundWhiteRfi*255).astype(np.uint8)
        transparantBackgroundInverse = transparantBackground.copy()
        transparantBackgroundInverse[:,:,:3] = 255-transparantBackgroundInverse[:,:,:3]

        transparantBackground = Image.fromarray(transparantBackground, 'RGBA')
        transparantBackgroundWhiteRfi = Image.fromarray(transparantBackgroundWhiteRfi, 'RGBA')
        transparantBackgroundInverse = Image.fromarray(transparantBackgroundInverse, 'RGBA')


        [antennaA, antennaB, posA, posB] = self.metaData
        saveLocation = os.path.join(self.plotLocation,'Transparant')
        os.makedirs(saveLocation, exist_ok=True)
        transparantBackground.save(os.path.join(saveLocation,'{}_{}_transparantBackground.png').format(antennaA,antennaB))
        transparantBackgroundInverse.save(os.path.join(saveLocation,'{}_{}_transparantBackgroundInverse.png').format(antennaA,antennaB))
        transparantBackgroundWhiteRfi.save(os.path.join(saveLocation,'{}_{}_whiteRfi.png'.format(antennaA,antennaB)))

    def plotplot(self):
        [antennaA, antennaB, posA, posB] = self.metaData
        distance=self.calcDistance(posA, posB)
        self.fig.subplots_adjust(hspace=0)
       
        for _, frequencyPair in enumerate(self.subbandOrder):           
            if frequencyPair[0] == -1:
                rowText='N/A'
            else:
                rowText=self.setNames[frequencyPair[0]]
            self.rowTitles.append(rowText)
        self.rowTitles.reverse()

        pad = 5 # in points
        axes = self.axs
        for ax, col in zip(axes[0], self.columnTitles):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')

        for ax, row in zip(axes[:,0], self.rowTitles):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

        figureName = 'Baseline %s_%s' %(antennaA, antennaB)
        plt.savefig(os.path.join(self.plotLocation,figureName), dpi=50, bbox_inches='tight')        
        plt.cla()
        plt.clf()
        plt.close('all')   

# def ObservationsToColor(observation, mode):
#     observations = np.asarray(observation)
#     if mode == 'amplitude':
#         modeIndex = 0
#     elif mode == 'phase':
#         modeIndex = 1
#     else:
#         raise Exception('Invalid mode. Please choose amplitude or phase')

#     observationMode = observations[:,:,:,modeIndex]
#     modeColor = np.zeros((observationMode.shape[0],observationMode.shape[1],3))
#     modeColor[:,:,0] = observationMode[:,:,0]
#     modeColor[:,:,1] = np.clip(0.5*observationMode[:,:,1] + 0.5*observationMode[:,:,2], 0.0, 1.0)
#     modeColor[:,:,2] = observationMode[:,:,3]
#     return modeColor

# def channelsToColor(observation, mode):
#     observations = np.asarray(observation)
#     if mode == 'amplitude':
#         modeIndex = 0
#     elif mode == 'phase':
#         modeIndex = 1
#     else:
#         raise Exception('Invalid mode. Please choose amplitude or phase')

#     modeColor = np.zeros((observations.shape[0],observations.shape[1],3))
#     if observation.shape[2] == 8:
#         modeColor[:,:,0] = observations[:,:,modeIndex]
#         modeColor[:,:,1] = 0.5*observations[:,:,2+modeIndex] + 0.5*observations[:,:,4+modeIndex]
#         modeColor[:,:,2] = observations[:,:,6+modeIndex]
#     elif observation.shape[2] == 4:
#         modeColor[:,:,0] = observations[:,:,0]
#         modeColor[:,:,1] = 0.5*observations[:,:,1] + 0.5*observations[:,:,2]
#         modeColor[:,:,2] = observations[:,:,3]
#     return modeColor

def plotLabels(labels, titles = None, saveFileName = None, rowTitles = None, colTitles = None):
    nRows = labels.shape[0]
    nColumns = labels.shape[3]
    figSize = labels[0].shape

    # Create a 2D plot
    dpi=300
    factor=10
    subPlotWidht=factor*figSize[1]/dpi
    subPlotHeight=factor*figSize[0]/dpi
    figureSize = (subPlotWidht*nColumns, subPlotHeight*nRows)

    fig,axes = plt.subplots(nRows,nColumns, figsize=figureSize)

    for rowIndex, prediction in enumerate(labels):
        for columnIndex in range(nColumns):
            oneImage = prediction[:,:,columnIndex]
            if nColumns >1:
                ax = axes[rowIndex,columnIndex]
            else:
                ax = axes[rowIndex]
            ax.imshow(oneImage, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if titles is not None:
                title = str(titles[rowIndex,0]) + '-' + str(titles[rowIndex,1])
                ax.set_title(title)

    pad = 50 # in points
    if colTitles is not None:
        for ax, col in zip(axes[0], colTitles):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')

    if rowTitles is not None:
        if nColumns >1:
            for ax, rowText in zip(axes[:,0], rowTitles):
                ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
        else:
            for ax, rowText in zip(axes, rowTitles):
                rowText = str(rowText)
                ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

    # Create a date formatter for the x-axis labels
    plt.tight_layout()
    if saveFileName == None:
        plt.show()
    else:
        plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')

def plotGeneratedPredictionsLabels(generatedBatch, realBatch, titles = None, saveFileName = None, rowTitles = None, colTitles = None):
    nRows = generatedBatch.shape[0]
    nColumns = 2

    # Create a 2D plot
    dpi=300
    subPlotWidht=nColumns * 8
    subPlotHeight=nRows * 2
    figureSize = (subPlotWidht, subPlotHeight)

    fig,axes = plt.subplots(nRows,nColumns, figsize=figureSize)

    for rowIndex in range(nRows):
        generated = generatedBatch[rowIndex]
        real = realBatch[rowIndex]

        generatedMin = np.min(generated)
        generatedMax = np.max(generated)
        generatedNormalized = (generated-generatedMin)/(generatedMax-generatedMin)

        axes[rowIndex,0].imshow(generatedNormalized, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')
        axes[rowIndex,1].imshow(real, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')

        axes[rowIndex,0].set_xticklabels([])
        axes[rowIndex,0].set_yticklabels([])
        axes[rowIndex,1].set_xticklabels([])
        axes[rowIndex,1].set_yticklabels([])

        axes[rowIndex,0].set_title('generated')
        axes[rowIndex,1].set_title('real')

    pad = 50 # in points
    if rowTitles is not None:
        if nColumns >1:
            for ax, rowText in zip(axes[:,0], rowTitles):
                ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
        else:
            for ax, rowText in zip(axes, rowTitles):
                rowText = str(rowText)
                ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

    # Create a date formatter for the x-axis labels
    plt.tight_layout()
    if saveFileName == None:
        plt.show()
    else:
        plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')

# def plotPredictions(labels, titles = None, saveFileName = None, rowTitles = None, colTitles = None):
#     nRows = labels.shape[0]

#     if len(labels.shape)==3:
#         nColumns = 1
#         figSize = labels[0].shape
#         convertToColor = False
#     else:
#         nColumns = labels.shape[1]
#         figSize = labels[0].shape[1:]
#         convertToColor = labels[0].shape[-1] == 4

#     # Create a 2D plot
#     dpi=300
#     factor=10
#     subPlotWidht=factor*figSize[1]/dpi
#     subPlotHeight=factor*figSize[0]/dpi
#     figureSize = (subPlotWidht*nColumns, subPlotHeight*nRows)

#     fig,axes = plt.subplots(nRows,nColumns, figsize=figureSize)

#     for rowIndex, prediction in enumerate(labels):
#         for columnIndex in range(nColumns):

#             if nColumns >1:
#                 oneImage = prediction[columnIndex, :,:]
#                 ax = axes[rowIndex,columnIndex]
#             else:
#                 oneImage = prediction
#                 ax = axes[rowIndex]
#             if convertToColor:
#                 tempImage = oneImage.copy()
#                 oneImage = np.zeros((tempImage.shape[0], tempImage.shape[1], 3))
#                 oneImage[:,:,0] = tempImage[:,:,0]
#                 oneImage[:,:,1] = tempImage[:,:,1]*0.5 + tempImage[:,:,2]*0.5
#                 oneImage[:,:,2] = tempImage[:,:,3]
#             ax.imshow(oneImage, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')

#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#             # ax.set
#             # ax.axis('off')
#             if titles is not None:
#                 if nColumns >1:
#                     title = str(titles[rowIndex,columnIndex])
#                 else:
#                     title = str(titles[rowIndex])

#                 ax.set_title(title)

#     pad = 50 # in points
#     if colTitles is not None:
#         for ax, col in zip(axes[0], colTitles):
#             ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')

#     if rowTitles is not None:
#         if nColumns >1:
#             for ax, rowText in zip(axes[:,0], rowTitles):
#                 ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
#         else:
#             for ax, rowText in zip(axes, rowTitles):
#                 rowText = str(rowText)
#                 ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

#     # Create a date formatter for the x-axis labels
#     plt.tight_layout()
#     if saveFileName == None:
#         plt.show()
#     else:
#         plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')

#     plt.close()

# def plotGraph(dataX, dataY, graphNames = None, graphTitle = None, xLabel = None, xAxisInt = True, saveFileName = None):   
#     dpi=300

#     plt.figure(figsize=(10, 10))
    
    
#     if xAxisInt:
#         ax = plt.figure().gca()
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#     if len(dataY.shape)>1:
#         # more than one plot
#         for graphIndex in range(dataY.shape[0]):
#             if graphNames is not None:
#                 plt.plot(dataX, dataY[graphIndex], label=graphNames[graphIndex])
#             else:
#                 plt.plot(dataX, dataY[graphIndex])
#     else:
#         plt.plot(dataX, dataY)
    
#     if graphNames is not None:
#         plt.legend()
    
#     if graphTitle is not None:
#         plt.title(graphTitle)

#     if xLabel is not None:
#         plt.xlabel(xLabel)

#     # Save or show figure
#     plt.tight_layout()
#     if saveFileName == None:
#         plt.show()
#     else:
#         plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')
#     plt.close()

# def plotCalcHistogramsPredictions(labels, titles = None, saveFileName = None, rowTitles = None, colTitles = None):
#     nRows = labels.shape[0]
#     figSize = labels[0].shape

#     # Create a 2D plot
#     dpi=300
#     factor=10
#     subPlotWidht=factor*figSize[1]/dpi
#     subPlotHeight=factor*figSize[0]/dpi
#     figureSize = (subPlotWidht*2, subPlotHeight*nRows)

#     fig,axes = plt.subplots(nRows,2, figsize=figureSize)

#     for rowIndex, prediction in enumerate(labels):
#         oneImage = prediction
#         ax = axes[rowIndex, 0]
#         ax.imshow(oneImage, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])

#         if titles is not None:
#             title = str(titles[rowIndex])
#             ax.set_title(title)
        
#         ax = axes[rowIndex, 1]
#         flattened = np.asarray(oneImage.flatten())
#         ax.hist(flattened, bins=100)
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])

#         if titles is not None:
#             title = str(titles[rowIndex])
#             ax.set_title(title)

#     # Create a date formatter for the x-axis labels
#     plt.tight_layout()
#     if saveFileName == None:
#         plt.show()
#     else:
#         plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')

#     plt.close()

# def multiPlot(columnList, plotSettings = {}, saveFileName = None):
#     '''
#     For each dict in column:
#     ['data'] = row x height x width x channels
#     ['data'] = height x width x channels
#     ['data'] = row x height x width
#     ['data'] = height x width
#     '''
#     columns = []

#     for columnIndex in range(len(columnList)):
#         if 'multiData' in columnList[columnIndex]:
#             multiColumnTitles = None
#             if 'colNames' in columnList[columnIndex]:
#                 multiColumnTitles = columnList[columnIndex]['colNames']
#             elif 'colName' in columnList[columnIndex]:
#                 multiColumnTitles = []
#                 colBaseName = columnList[columnIndex]['colName']
#                 for multiNameIndex in range(len(columnList[columnIndex]['multiData'])):
#                     multiColumnTitles.append(colBaseName + ' {}'.format(multiNameIndex))

#             newColumn = {}
#             for multiColumnIndex, columnData in enumerate(columnList[columnIndex]['multiData']):
#                 newColumn['data'] = columnData
#                 if multiColumnTitles is not None:
#                     newColumn['colName'] = multiColumnTitles[multiColumnIndex]
#                 columns.append(newColumn.copy())
#         if 'data' in columnList[columnIndex]:
#             columns.append(columnList[columnIndex])

#     nColumns = len(columns)
#     nRows = 0
#     subFigureHeight = 0
#     subFigureWidth = 0

#     # For each column, reshape it to: row x height x width x channels
#     for columnIndex in range(len(columns)):
#         # First determine dimensions of the plot
#         columnData = np.asarray(columns[columnIndex]['data'])
#         dataShape = columnData.shape

#         if len(dataShape) == 2:
#             reshapedData = np.reshape(columnData,(1,dataShape[0],dataShape[1],1))
        
#         elif len(dataShape) == 3:
#             if dataShape[-1] <= 8:
#                 # One row and multiple channels
#                 reshapedData = np.reshape(columnData,(1,dataShape[0],dataShape[1],dataShape[2]))
#             else:
#                 # Multiple rows and one channels
#                 reshapedData = np.reshape(columnData,(dataShape[0],dataShape[1],dataShape[2],1))
        
#         elif len(dataShape) == 4:
#             reshapedData = columnData
        
#         else:
#             raise Exception("shape of column data undefined")
        
#         columns[columnIndex]['data'] = reshapedData

#         if reshapedData.shape[1] > subFigureHeight:
#             subFigureHeight = reshapedData.shape[1]

#         if reshapedData.shape[2] > subFigureWidth:
#             subFigureWidth = reshapedData.shape[2]

#         # Check if the number of rows are equal
#         columnRows = columns[columnIndex]['data'].shape[0]
#         if nRows > 0 and nRows != columnRows:
#             raise Exception("number of rows changed. Not square")
#         nRows = columnRows
    
#     if nRows == 0: raise Exception("No rows found")

#     # Create a 2D plot
#     dpi=300
#     factor=10
#     subPlotHeight=factor*subFigureHeight/dpi
#     subPlotWidht=factor*subFigureWidth/dpi
#     figureSize = (subPlotWidht*nColumns, subPlotHeight*nRows)

#     fig,axes = plt.subplots(nRows,nColumns, figsize=figureSize)

#     colNames = None
#     rowNames = None

#     for columnIndex, column in enumerate(columns):
#         columnData = column['data']
#         if nColumns == 1:
#             columnAxes = axes
#         else:
#             columnAxes = axes[:,columnIndex]
#         for rowIndex, image in enumerate(columnData):
#             # image has shape height x width x channels. Channels is >= 1
#             if nRows == 1:
#                 ax = columnAxes
#             else:
#                 ax = columnAxes[rowIndex]

#             if image.shape[-1] == 4:
#                 tempImage = np.zeros((image.shape[0], image.shape[1], 3))
#                 tempImage[:,:,0] = image[:,:,0]
#                 tempImage[:,:,1] = image[:,:,1]*0.5 + image[:,:,2]*0.5
#                 tempImage[:,:,2] = image[:,:,3]
#                 image = tempImage
            
#             if image.shape[-1] == 2:
#                 tempImage = np.zeros((image.shape[0], image.shape[1], 3))
#                 tempImage[:,:,0] = image[:,:,0]
#                 tempImage[:,:,1] = image[:,:,1]
#                 # tempImage[:,:,2] = image[:,:,3]
#                 image = tempImage

#             ax.imshow(image, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])

#             if 'titles' in column:
#                 if nRows >1:
#                     title = str(column['titles'][rowIndex])
#                 else:
#                     title = str(column['titles'])
#                 ax.set_title(title)
            
#             if 'colName' in column:
#                 if colNames is None:
#                     colNames = [""]*nColumns
#                 colNames[columnIndex] = column['colName']
    
#     if 'rowNames' in plotSettings:
#         #if rowTitles is None:
#         if nRows == 1:
#             if isinstance(column['rowNames'], str):
#                 rowNames = [column['rowNames']]
#             else:
#                 rowNames = column['rowNames']
#         else:
#             rowNames = column['rowNames']

#     pad = 50 # in points
#     if colNames is not None:
#         for ax, col in zip(axes[0], colNames):
#             ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large', ha='center', va='baseline')

#     if rowNames is not None:
#         if nColumns >1:
#             for ax, rowText in zip(axes[:,0], rowNames):
#                 ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')
#         else:
#             for ax, rowText in zip(axes, rowNames):
#                 rowText = str(rowText)
#                 ax.annotate(rowText, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label, textcoords='offset points', size='large', ha='right', va='center')

#     # Create a date formatter for the x-axis labels
#     plt.tight_layout()
#     if saveFileName == None:
#         plt.show()
#     else:
#         plt.savefig(saveFileName, dpi=dpi, bbox_inches='tight')
#     plt.close()

def tsneVisualization(tsneFeatures, dataX, plotDirectory, plotTitle, dpi=100):
    whiteSpace = 64
    lineAroundImage = 1

    backgroundColor = 255# 220
    canvasSize = 6000#10000
    brightness = 1.5

    canvasXmax = canvasSize
    canvasYmax = canvasSize

    canvas = np.ones((canvasYmax+dataX.shape[1]+2*whiteSpace, canvasXmax+dataX.shape[2]+2*whiteSpace,3),dtype=np.uint8)*backgroundColor

    embeddingX = tsneFeatures[:,0]
    embeddingY = tsneFeatures[:,1]

    ratioX = canvasXmax/(np.max(embeddingX)-np.min(embeddingX))
    offsetX = -np.min(embeddingX)*ratioX+whiteSpace
    startX = np.round(offsetX + ratioX*embeddingX).astype(int)

    ratioY = canvasYmax/(np.max(embeddingY)-np.min(embeddingY))
    offsetY = -np.min(embeddingY)*ratioY+whiteSpace
    startY = np.round(offsetY + ratioY*embeddingY).astype(int)

    canvas[startY, startX,:] = [0,0,0]

    dataX = (dataX*255).astype(np.uint8)
    sorted_indices = np.argsort(tsneFeatures.sum(axis=1))[::-1]
    for dataIndex in sorted_indices:
        xPos = startX[dataIndex]
        yPos = startY[dataIndex]
        imageX = dataX[dataIndex]
        imageX = np.clip(imageX*brightness,0,255)
        plotImage = np.ones((imageX.shape[0]+lineAroundImage*2,imageX.shape[1]+lineAroundImage*2,3),dtype=np.uint8)*0
        plotImage[lineAroundImage:lineAroundImage+imageX.shape[0],lineAroundImage:lineAroundImage+imageX.shape[1],:] = imageX
        canvas[yPos:yPos+plotImage.shape[0],xPos:xPos+plotImage.shape[1],:] = plotImage

    canvas = np.flip(canvas, axis=0)

    # save canvas as PIL
    canvas_image = Image.fromarray(canvas)
    canvas_image.save(os.path.join(plotDirectory, plotTitle + ".png"))

def tsneClusterScatter(tsneFeatures, plotDirectory, plotTitle, clusters,dpi=100, plotCenters = False, legend = None, remappedLabels = None, colorMap = None):
    #plt.figure(figsize=(10, 10))
    plt.figure(figsize=(6, 6))
    
    if colorMap is None:
        colors = cm.nipy_spectral(clusters.astype(float) / len(np.unique(clusters)))
    else:
        colors = []
        for cluster in clusters:
            colors.append(colorMap[cluster])

        colors=np.asarray(colors)

    if remappedLabels is not None:
        labelIntersection = clusters == remappedLabels[0]
        if len(remappedLabels)>0:
            for newLabelsIndex in range(1,len(remappedLabels)):
                tempIntersection = clusters == remappedLabels[newLabelsIndex]
                labelIntersection = np.logical_and(labelIntersection, tempIntersection)
        labelDifference = labelIntersection == False
    
    if legend is None:
        if remappedLabels is None:
            plt.scatter(tsneFeatures[:, 0], tsneFeatures[:, 1], marker=".", s=50, alpha=1, lw=0, c=colors, edgecolor="k")
        else:
            plt.scatter(tsneFeatures[labelIntersection, 0], tsneFeatures[labelIntersection, 1], marker=".", s=50, alpha=1, lw=0, c=colors[labelIntersection], edgecolor="k")
            plt.scatter(tsneFeatures[labelDifference, 0], tsneFeatures[labelDifference, 1], marker=".", s=50, alpha=0.05, lw=0, c=colors[labelDifference], edgecolor="k")

    else:
        for clusterIdx in np.unique(clusters):
            clusterSamples = np.where(clusters == clusterIdx)
            plt.scatter(tsneFeatures[clusterSamples, 0], tsneFeatures[clusterSamples, 1], marker=".", s=50, alpha=1, lw=0, c=colors[clusterSamples], edgecolor="k", label=legend[clusterIdx])
        plt.legend()

    if plotCenters:
        centers = []
        for i in range(len(np.unique(clusters))):
            center = np.mean(tsneFeatures[clusters == i], axis=0)
            centers.append(center)
        centers = np.asarray(centers)
        plt.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=0.7,s=200,edgecolor="k",)
        for i, c in enumerate(centers):
            plt.annotate(str(i), (c[0], c[1]-0.5), fontsize=10, color='black', ha='center', va='center')

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    os.makedirs(plotDirectory, exist_ok=True)
    plt.savefig(os.path.join(plotDirectory, plotTitle), dpi=dpi, bbox_inches='tight')
    plt.close()

def tsneScatter(tsneFeatures, plotDirectory, plotTitle, dpi=100):
    plt.figure()
    plt.scatter(tsneFeatures[:, 0], tsneFeatures[:, 1],s=10)
    plt.xlabel("tsne 1")
    plt.ylabel("tsne 2")
    plt.savefig(os.path.join(plotDirectory, plotTitle), dpi=dpi, bbox_inches='tight')
    plt.close()

def tsneClustersWithSilhouette(tsneFeatures, plotDirectory, plotTitle, clusters, kClusters,sampleSilhouetteValue,  silhouette_avg, dpi=300, plotCenters = False):
        # This function is based on code written by scikit-learn authors found at
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#:~:text=The%20silhouette%20plot%20displays%20a,of%20%5B%2D1%2C%201%5D.
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(21, 10)

        # Silhouette plot
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
        ax1.set_title("Silhouette plot for the various clusters.")
        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster label")
        
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks

        # 2nd Plot showing the actual clusters formed 
        colors = cm.nipy_spectral(clusters.astype(float) / len(np.unique(clusters)))
        colors=np.asarray(colors)
        ax2.scatter(tsneFeatures[:, 0], tsneFeatures[:, 1], marker=".", s=50, alpha=1, lw=0, c=colors, edgecolor="k")

        if plotCenters:
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

        #dpi=300
        plt.savefig(os.path.join(plotDirectory, plotTitle), dpi=dpi, bbox_inches='tight')
        plt.close()