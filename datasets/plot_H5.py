import utils as utils

datasetName = 'LOFAR_L2014581 (recording)'

def plotH5Set(antennasA = None, antennasB = None, plotType = 'rgb', plotLocation = None):
    if plotLocation is None:
        plotLocation = utils.functions.getPlotLocation(datasetName, 'h5')
        
    # Data selection
    if antennasA is None:
        antennasA = ["CS002HBA1","CS003HBA0","CS003HBA1","CS004HBA0","CS004HBA1","CS005HBA0","CS005HBA1","CS006HBA0","CS006HBA1","CS007HBA0","CS007HBA1"]

    if antennasB is None:
        antennasB = antennasA

    plotter = utils.plotter.h5Plotter(datasetName,plotLocation)

    if plotType == 'transparant':
        plotLayout = [{'type':'transparantBackground'}]

    if plotType == 'rgb':
        plotLayout = [{'type':'visibility','mode':'amplitude','polarization':'rgb'},
                      {'type':'visibility','mode':'phase','polarization':'rgb'}
                      ]
    if plotType == 'rgbLabels':
        plotLayout = [{'type':'visibility','mode':'amplitude','polarization':'rgb'},
                      {'type':'visibility','mode':'phase','polarization':'rgb'},
                      {'type':'labels'},
                      ]        
    elif plotType == 'rgbHistograms':
        plotLayout = [{'type':'visibility','mode':'amplitude','polarization':'rgb'},
                      {'type':'histogram','mode':'amplitude','polarizations':['XX', 'XY', 'YX','YY']},
                      {'type':'visibility','mode':'phase','polarization':'rgb'},
                      {'type':'histogram','mode':'phase','polarizations':['XX', 'XY', 'YX','YY']},
                      {'type':'labels'},
                      ]
    elif plotType == 'all':
        plotLayout = [{'type':'visibility','mode':'amplitude','polarization':'XX'},
                    {'type':'histogram','mode':'amplitude','polarizations':['XX']},
                    {'type':'visibility','mode':'phase','polarization':'XX'},
                    {'type':'histogram','mode':'phase','polarizations':['XX']},
                    {'type':'visibility','mode':'amplitude','polarization':'XY'},
                    {'type':'histogram','mode':'amplitude','polarizations':['XY']},
                    {'type':'visibility','mode':'phase','polarization':'XY'},
                    {'type':'histogram','mode':'phase','polarizations':['XY']},
                    {'type':'visibility','mode':'amplitude','polarization':'YX'},
                    {'type':'histogram','mode':'amplitude','polarizations':['YX']},
                    {'type':'visibility','mode':'phase','polarization':'YX'},
                    {'type':'histogram','mode':'phase','polarizations':['YX']},
                    {'type':'visibility','mode':'amplitude','polarization':'YY'},
                    {'type':'histogram','mode':'amplitude','polarizations':['YY']},
                    {'type':'visibility','mode':'phase','polarization':'YY'},
                    {'type':'histogram','mode':'phase','polarizations':['YY']},
                    {'type':'labels'}
                    ]
        
    for antennaA in antennasA:
        for antennaB in antennasB:
            if antennaB == antennaA: continue
            plotter.plotMultiplot(antennaA,antennaB,plotLayout, normalizationMethod=12)