from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .glover.methods import functions
from .glover import viewer

class gloverAnalyzer:

    """
    A class for calculating quality assurance metrics on fMRI data based on the "Report on a Multicenter fMRI Quality Assurance Protocol" paper by Friedman and Glover.

    Parameters:
    - timeseries_data (numpy.ndarray): A 3D array containing time series data.

    Attributes:
    - timeseries (numpy.ndarray): The input time series data.
    - signalImage (numpy.ndarray): Mean image calculated from the time series.
    - tempFlucImage (numpy.ndarray): Standard deviation image of detrended time series data.
    - sfnrImage (numpy.ndarray): Signal-to-Fluctuation-Noise Ratio (SFNR) image.
    - staticSpatialNoiseImage (numpy.ndarray): Image representing static spatial noise.
    - snrImage (numpy.ndarray): Signal-to-Noise Ratio (SNR) image.
    - snrSV (float): SNR Summary Value.
    - sfnrSV (float): SFNR Summary Value.
    - percentFluc (float): Percent Fluctuation.
    - residualsSVs (numpy.ndarray): Summary Values of residuals from detrended signal.
    - residualsSVsFT (numpy.ndarray): Fast Fourier Transform of residuals Summary Values.
    - drift (float): Drift percentage.

    Methods:
    - weisskoffAnalysis(maxROI, showPlot=False): Perform Weisskoff analysis on the data and return the coefficient of variation.

    """

    def __init__(self, timeseries_data):
        assert len(timeseries_data.shape) == 3,"Data has to have only 3 dimensions (time,space,space)"
        self.timeseries = timeseries_data

        self.viewer = viewer
        # self.functions = functions
        # self.debug = self.MiscFunctions()

        self._signalImage = None
        self._tempflucnoiseImage = None
        self._sfnrImage = None
        self._spatialnoiseImage = None
        self._snrImage = None

        self._snrSV = None
        self._sfnrSV = None

        self._percentFluc = None
        self._drift = None
        self._residualsSVs = None
        self._residualsSVsFT = None

    @property
    def signalImage(self):
        if self._signalImage is None:
            # Calculate the mean image
            self._signalImage = np.mean(self.timeseries, axis=0)
        return self._signalImage

    @property
    def tempFlucImage(self):
        if self._tempflucnoiseImage is None:
            detrend = functions.detrend_image(self.timeseries)
            self._tempflucnoiseImage = np.std(detrend, axis=0)
        return self._tempflucnoiseImage

    @property
    def sfnrImage(self):
        if self._sfnrImage is None:
            self._sfnrImage = np.divide(self.signalImage,self.tempFlucImage + 0.05 * np.max(self.tempFlucImage))
        return self._sfnrImage

    @property
    def staticSpatialNoiseImage(self):
        if self._spatialnoiseImage is None:
            sum_even, sum_odd = functions.addPixelwiseSums(self.timeseries)
            self._spatialnoiseImage = np.abs(sum_even - sum_odd) #/ self.timeseries.shape[0] # NICHT GANZ SICHER!!!!!!
        return self._spatialnoiseImage

    @property
    def snrImage(self):
        if self._snrImage is None:
            self._snrImage = np.divide(self.signalImage, self.staticSpatialNoiseImage + 0.5 * np.max(self.staticSpatialNoiseImage))
        return self._snrImage
    
    @property
    def snrSV(self):
        if self._snrSV is None:
            signal_SV = functions.summaryValue(self.signalImage)
            SD_spatialNoiseSV = np.std(functions.roi(self.staticSpatialNoiseImage))
            self._snrSV = np.round(signal_SV / SD_spatialNoiseSV / self.timeseries.shape[0],2)
        return self._snrSV
    
    @property
    def sfnrSV(self):
        if self._sfnrSV is None:
            self._sfnrSV = np.round(functions.summaryValue(self.sfnrImage),2)
        return self._sfnrSV
    
    @property
    def percentFluc(self):
        if self._percentFluc is None:
            meanImageIntensity = functions.summaryValue(self.signalImage)
            meanStandDev = functions.summaryValue(self.tempFlucImage)
            self._percentFluc = np.round(100 * meanStandDev / meanImageIntensity,2)
        return self._percentFluc

    @property
    def residualsSVs(self):
        if self._residualsSVs is None:
            detrendedSignalImage = functions.detrend_image(functions.roi(self.timeseries))
            residualsSV = np.empty(detrendedSignalImage.shape[0])
            for timestep in range(residualsSV.shape[0]):
                residualsSV[timestep] = functions.summaryValue(detrendedSignalImage[timestep,:,:])
            self._residualsSVs = residualsSV
        return self._residualsSVs

    @property
    def residualsSVsFT(self):
        if self._residualsSVsFT is None:
            self._residualsSVsFT = np.abs(np.fft.fftshift(np.fft.fft(self.residualsSVs)))
        return self._residualsSVsFT
    
    @property
    def drift(self):
        if self._drift is None:
            timeseries_SVs = np.empty(self.timeseries.shape[0])
            for timestep in range(self.timeseries.shape[0]):
                timeseries_SVs[timestep] = functions.summaryValue(self.timeseries[timestep,:,:])
            fit = functions.quad_fit(timeseries_SVs)
            self._drift = np.round(100 * (fit[0] - fit[-1]) / functions.summaryValue(self.signalImage),2)
        return self._drift


    def weisskoffAnalysis(self, maxROI=21, showPlot=True):
        coeffOfVar = np.empty((maxROI,2))
        theory_best = np.empty((maxROI,2))
        meanImage = self.signalImage

        for roi in range(maxROI):
            #print(roi+1)
            mi_sv = functions.summaryValue(meanImage,roi+1)
            
            sd_sv = functions.summaryValue_over_time(self.timeseries, roi+1)
            sd_sv = np.std(sd_sv)

            coeffOfVar[roi,0] = 100 * (sd_sv / mi_sv)
            coeffOfVar[roi,1] = roi + 1

            theory_best[roi,0] = coeffOfVar[0,0] / (roi + 1)
            theory_best[roi,1] = roi + 1
        #print(coeffOfVar)
        
        rdc = np.round(coeffOfVar[0,0] / coeffOfVar[-1,0],2)

        if showPlot:
            plt.figure()
            plt.plot(coeffOfVar[:, 1], coeffOfVar[:, 0],label=f"Exp. Data RDC={rdc}")
            plt.plot(theory_best[:,1],theory_best[:,0],label="Theory") 
            plt.legend()
            plt.yscale('log')  # Set the y-axis to a logarithmic scale
            plt.xscale('log')  # Set the x-axis to a logarithmic scale
            plt.xlabel('Logarithmic ROI Index')
            plt.ylabel('Logarithmic Coefficient of Variation (%)')
            plt.title(f'Weisskoff Analysis, RDC: {rdc}')
            plt.grid(True)
            plt.show()

        return coeffOfVar


    def weisskoffAnalysisDebug(self,meanImage, timeSeries, maxROI=21, showPlot=True):
        coeffOfVar = np.empty((maxROI,2))
        theory_best = np.empty((maxROI,2))
        meanImage = meanImage
        sdImage = timeSeries
        for roi in range(maxROI):
            #print(roi+1)
            mi_sv = functions.summaryValue(meanImage,roi+1)

            sd_sv = functions.summaryValue_over_time(sdImage, roi+1)
            sd_sv = np.std(sd_sv)

            coeffOfVar[roi,0] = 100 * (sd_sv / mi_sv)
            coeffOfVar[roi,1] = roi + 1

            theory_best[roi,0] = coeffOfVar[0,0] / (roi + 1)
            theory_best[roi,1] = roi + 1
        #print(coeffOfVar)
        
        rdc = np.round(coeffOfVar[0,0] / coeffOfVar[-1,0],2)

        if showPlot:
            plt.figure()
            plt.plot(coeffOfVar[:, 1], coeffOfVar[:, 0],label=f"Exp. Data RDC={rdc}")
            plt.plot(theory_best[:,1],theory_best[:,0],label="Theory") 
            plt.legend()
            plt.yscale('log')  # Set the y-axis to a logarithmic scale
            plt.xscale('log')  # Set the x-axis to a logarithmic scale
            plt.xlabel('Logarithmic ROI Index')
            plt.ylabel('Logarithmic Coefficient of Variation (%)')
            plt.title(f'Weisskoff Analysis, RDC: {rdc}')
            plt.grid(True)
            plt.show()

        return coeffOfVar
