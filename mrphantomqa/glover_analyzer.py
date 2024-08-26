import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

from datetime import datetime
from fpdf import FPDF

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

    def __init__(self, timeseries_data, SVroi=21):
        assert len(timeseries_data.imagedata[:,9,:,:].shape) == 3,"Data has to have only 3 dimensions (time,space,space)"
        self.timeseries = timeseries_data.imagedata[:,9,:,:]
        self.metadata = timeseries_data.metadata
        self.SVroi = SVroi

        self.viewer = viewer

        self.scannername        = self.metadata[0x00080080].value
        self.creationdate       = self.metadata[0x00080012].value

        # self.viewer = viewer
        # self.functions = functions
        # self.debug = self.MiscFunctions()

        self._signalImage           = None
        self._tempflucnoiseImage    = None
        self._sfnrImage             = None
        self._spatialnoiseImage     = None
        self._snrImage              = None

        self._snrSV                 = None
        self._sfnrSV                = None

        self._percentFluc           = None
        self._drift                 = None
        self._residualsSVs          = None
        self._residualsSVsFT        = None
        self.rdc                    = None

        self._data_organized        = None
        self.longtermdata       = {}

        self.dirs               = {     # KEEP OS.SEP!!!
            "png"   : "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/glover" + f"{os.sep}",
            "csv"   : "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/glover" + f"{os.sep}",
            "srp"   : "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/glover" + f"{os.sep}",
            "lrp"   : "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/glover" + f"{os.sep}"
        }

        for filetype, dir_to_save_to in self.dirs.items():
            if not os.path.exists(dir_to_save_to):
                os.makedirs(dir_to_save_to, exist_ok=True)


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
            signal_SV = functions.summaryValue(self.signalImage, self.SVroi)
            SD_spatialNoiseSV = np.std(functions.roi(self.staticSpatialNoiseImage, self.SVroi))
            self._snrSV = np.round(signal_SV / SD_spatialNoiseSV / self.timeseries.shape[0],2)
        return self._snrSV
    
    @property
    def sfnrSV(self):
        if self._sfnrSV is None:
            self._sfnrSV = np.round(functions.summaryValue(self.sfnrImage, self.SVroi),2)
        return self._sfnrSV
    
    @property
    def percentFluc(self):
        if self._percentFluc is None:
            meanImageIntensity = functions.summaryValue(self.signalImage, self.SVroi)
            meanStandDev = functions.summaryValue(self.tempFlucImage, self.SVroi)
            self._percentFluc = np.round(100 * meanStandDev / meanImageIntensity,2)
        return self._percentFluc

    @property
    def residualsSVs(self):
        if self._residualsSVs is None:
            detrendedSignalImage = functions.detrend_image(functions.roi(self.timeseries, self.SVroi))
            residualsSV = np.empty(detrendedSignalImage.shape[0])
            for timestep in range(residualsSV.shape[0]):
                residualsSV[timestep] = functions.summaryValue(detrendedSignalImage[timestep,:,:], self.SVroi)
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
                timeseries_SVs[timestep] = functions.summaryValue(self.timeseries[timestep,:,:], self.SVroi)
            fit = functions.quad_fit(timeseries_SVs)
            self._drift = np.round(100 * (fit[0] - fit[-1]) / functions.summaryValue(self.signalImage, self.SVroi),2)
        return self._drift


    def weisskoffAnalysis(self, maxROI=21, showPlot=False):
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

        self.rdc = rdc
        return rdc


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


    @property
    def data_organized(self):
        # Organize data. All tests have to have ran otherwise you get an error.
        if self._data_organized is None:
            self._data_organized = {
                "SNR SV": {
                    "result": self.snrSV,
                    "display_range": [0,2],
                },
                "SFNR SV": {
                    "result": self.sfnrSV,
                    "display_range": [0,150],
                },
                "PF": {
                    "result": self.percentFluc,
                    "display_range": [0,150],
                },
                "Drift": {
                    "result": self.drift,
                    "display_range": [0,150],
                }
            }
        
        return self._data_organized
    
    def add2csv(self):
        # csv header and data
        savedata = {
            "Date of measurement":  f"{self.metadata[0x00080020].value}",
            "Time of measurement":  f"{self.metadata[0x00080031].value}",
            "Time of evaluation":   f"{datetime.now()}"
        }
        for testname, prop in self.data_organized.items():
            savedata[testname] = prop["result"]

        csv_filename = self.dirs["csv"] + f'{self.scannername}_glover.csv'
        write_header = not os.path.isfile(csv_filename)
        
        # Write CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            if write_header:
                header = savedata.keys()
                csv_writer.writerow(header)
            writedata = [savedata[key] for key in savedata.keys()]
            csv_writer.writerow(writedata)

    def _readcsv(self):
        csv_filename = self.dirs["csv"] + f'{self.scannername}_glover.csv'
        with open(csv_filename, 'r') as file:
            reader = csv.DictReader(file)
            # Collect all rows in a list
            rows = [row for row in reader]
            # Sort rows by the 'Date of measurement' column
            rows = sorted(rows, key=lambda row: row['Date of measurement'])

            for row in rows:
                for column in reader.fieldnames:
                    if column in row:
                        if column not in self.longtermdata:
                            self.longtermdata[column] = []
                        self.longtermdata[column].append(row[column])

    def create_longterm_report(self):
        self._readcsv()

        for testname, value in self.data_organized.items():
            xdata_raw = self.longtermdata["Date of measurement"]
            dates = [datetime.strptime(date, '%Y%m%d') for date in xdata_raw]

            ydata = [float(i) for i in self.longtermdata[testname]]

            # Plot the data
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.gcf().autofmt_xdate()  # Auto format date labels
            plt.plot(dates, ydata, marker='o')
            plt.ylim(value["display_range"])
            # plt.ylim([0,10])

            # Adding titles and labels
            plt.title(f'Longitudinal plot for {testname}')
            plt.ylabel('Testresult')

            # Show the plot
            # plt.show()
            # Save figure
            plt.savefig(self.dirs["png"]+f"Longterm_{testname}_glover.png")
            plt.close()

        

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.set_font("Arial", size=12)
        x_offset = 10
        y_offset = 30
        width = 90
        height = 70

        for i, testname in enumerate(self.data_organized.keys()):
            if i % 3 == 0:
                pdf.add_page()
                y_offset = 30
            pdf.image(self.dirs["png"]+f"Longterm_{testname}_glover.png", x=x_offset, y=y_offset, w=width, h=height)
            pdf.set_xy(x_offset, y_offset + height + 5)
            # pdf.cell(200, 100, testname, ln=True)
            y_offset += height + 20
        
        pdf.output(self.dirs["lrp"] + f'{self.scannername}_longterm_report.pdf')

    def runall(self):
        self.resolution(False, True)
        self.low_contrast(False, True)
        self.uniformity(False, True)
        self.size(False, True)
        self.grid(False, True)
        self.thickness(False, True)
        self.position(False, True)
        self.ghosting(False, True)

        self.add2csv()
        self.create_report()
        self.create_longterm_report()
