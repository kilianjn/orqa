import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

from datetime import datetime
from fpdf import FPDF

from .acr.methods import functions as acrfunc
from .utils.methods import functions as utilfunc

class acrAnalyzer:
    def __init__(self, data, workdir) -> None:
        self.imagedata_loc  = data.imagedata[0] if hasattr(data, 'imagedata') else None
        self.metadata       = data.metadata if hasattr(data, 'metadata') else None
        
        self.scannername    = self.metadata[0x00080080].value
        self.creationdate   = self.metadata[0x00080012].value
        
        self.pixelSpacing   = [1,1]
        # Get correct pixel spacing
        if self.metadata.get(0x52009230) is not None: # Case Enhanced save
            self.pixelSpacing = self.metadata[0x52009230][0][0x00289110][0][0x00280030].value
        if self.metadata.get(0x00280030) is not None: # Case Interoperability
            self.pixelSpacing = [float(i) for i in self.metadata[0x00280030].value]

        self.res_GA         = None
        self.res_HCSR       = None
        self.res_STA        = None
        self.res_SPA        = None
        self.res_IIU        = None
        self.res_PGA        = None
        self.res_LCOD       = None

        self.workdir            = workdir
        while True:
            if os.path.exists(self.workdir):
                break
            else:
                print("Path does not exist. Try anew.")
                self.workdir = str(input("Type the path of the desired working directory: \n"))

        self._data_organized    = None  # All important data and metadata is stored here to create the corresponding reports.

        self.longtermdata       = {}    # Data from CSV files is stored here.

        # Make folders
        self.dirs               = {     # KEEP OS.SEP!!!
            "png"   : f"{self.workdir}" + f"{os.sep}acr_reports{os.sep}{self.scannername}{os.sep}imgs{os.sep}",
            "csv"   : f"{self.workdir}" + f"{os.sep}acr_reports{os.sep}{self.scannername}{os.sep}",
            "srp"   : f"{self.workdir}" + f"{os.sep}acr_reports{os.sep}{self.scannername}{os.sep}single_reports{os.sep}",
            "lrp"   : f"{self.workdir}" + f"{os.sep}acr_reports{os.sep}{self.scannername}{os.sep}"
        }

        for filetype, dir_to_save_to in self.dirs.items():
            if not os.path.exists(dir_to_save_to):
                os.makedirs(dir_to_save_to, exist_ok=True)

    def geometric_accuracy(self, showplot=False, print=False):
        #x, bins = methods.getHistogram(self.imagedata_loc)
        peak_value = utilfunc.getThreshold.findPeak(self.imagedata_loc[0],10)
        thldImage = utilfunc.createThresholdImage(self.imagedata_loc[0], peak_value/2)
        comy, comx = utilfunc.findCenter.centerOfMassFilled(thldImage)

        measureResults1 = acrfunc.ga.measureDistance(thldImage, (comy,comx), 45, self.pixelSpacing)
        measureResults2 = acrfunc.ga.measureDistance(thldImage, (comy,comx), -45, self.pixelSpacing)
        measureResults3 = acrfunc.ga.measureDistance(thldImage, (comy,comx), 0, self.pixelSpacing)

        if showplot or print:
            plt.imshow(self.imagedata_loc[0], cmap='gray')
            for len, coord in [measureResults1, measureResults2,measureResults3]:
                plt.scatter(coord[:, 1], coord[:, 0], c='red', marker='x')
                plt.plot(coord[:, 1], coord[:, 0], label=f"Length = {len}")
                plt.scatter(comx, comy, c='blue', marker='x')
            plt.legend()
            if showplot:
                plt.show()
            if print:
                plt.savefig(self.dirs["png"]+"ga_acr.png")
                plt.close()

        self.res_GA = np.round(np.mean([measureResults1[0],measureResults2[0],measureResults3[0]]),1)
            

    def high_contrast_spatial_resolution(self, showplot=False, print=False):
        self.res_HCSR = acrfunc.hcsr.cutoutRelevantSquare(self.imagedata_loc[0],showplot)

        return self.res_HCSR
    
    def slice_thickness_accuracy(self,showplot=False, print=False):
        thldvalue = utilfunc.getThreshold.findPeak(self.imagedata_loc[0], 10)
        thldimg = utilfunc.createThresholdImage(self.imagedata_loc[0], thldvalue/2)
        center = utilfunc.findCenter.centerOfMassFilled(thldimg)

        coord = utilfunc.getAreaBoundaries(thldimg, center)
        img = utilfunc.createSubarray(self.imagedata_loc[0], coord)
        img = img[:,int(img.shape[1]*0.15):int(img.shape[1]*0.85)]
        minrow, maxrow = acrfunc.sta.getrows(img)
        img = img[minrow:maxrow]        

        length, coords, border = acrfunc.sta.measureLength(img, self.pixelSpacing[1]) # Anstatt image border einsetzen.
        thld = utilfunc.getThreshold.otsuMethod(img)
        thld_rect = utilfunc.createThresholdImage(img, thld)
        self.res_STA = np.round(length,1)

        if showplot or print:
            plt.subplot(211)
            plt.imshow(thld_rect)
            plt.hlines(border,0,img.shape[1]-1,colors="green",alpha=0.5)
            plt.subplot(212)
            y_half = img.shape[0]/2
            plt.vlines(coords[0],ymin=0,ymax=y_half)
            plt.vlines(coords[1],ymin=0,ymax=y_half)
            plt.vlines(coords[2],ymin=y_half,ymax=y_half*2, color="red")
            plt.vlines(coords[3],ymin=y_half,ymax=y_half*2, color="red")
            plt.imshow(img, cmap="bone")
            if showplot:
                plt.show()
            if print:
                plt.savefig(self.dirs["png"]+"sta_acr.png")
                plt.close()



    def slice_position_accuracy(self, showplot=False, print=False):
        thld = utilfunc.getThreshold.findPeak(self.imagedata_loc[0],10)
        thld_image = utilfunc.createThresholdImage(self.imagedata_loc[0], thld/3)
        center = utilfunc.findCenter.centerOfMassFilled(thld_image)

        measureResults1 = acrfunc.spa.getPositionDifference(thld_image, center)

        thld = utilfunc.getThreshold.findPeak(self.imagedata_loc[10],10)
        thld_image = utilfunc.createThresholdImage(self.imagedata_loc[10], thld/3)
        center = utilfunc.findCenter.centerOfMassFilled(thld_image)

        measureResults2 = acrfunc.spa.getPositionDifference(thld_image, center)

        if showplot or print:
            dataset = (measureResults1, measureResults2)
            fig, axlist = plt.subplots(2,1)  
            for data,ax in zip(dataset,axlist):  
                ax.set_title(f"difference: {data[0]}")
                ax.imshow(data[1])
                ax.vlines(data[2], 0, data[1].shape[0]-1, linestyles="dotted")
            if showplot:
                plt.show()
            if print:
                plt.savefig(self.dirs["png"]+"spa_acr.png")
                plt.close()

        self.res_SPA = np.round(np.mean([measureResults1[0],measureResults2[0]]),1)

        pass

    def image_intensity_uniformity(self, showplot=False, print=False):
        thld = utilfunc.getThreshold.findPeak(self.imagedata_loc[6],10)
        thldimg = utilfunc.createThresholdImage(self.imagedata_loc[6], thld/2)
        center = utilfunc.findCenter.centerOfMassFilled(thldimg)

        kernelsize = 11
        maxValue, minValue, maxCoord, minCoord, convImgROI = acrfunc.iiu.searchForCircularSpots(self.imagedata_loc[6],center, 60, kernelsize)

        if showplot or print:
            plt.imshow(convImgROI)
            plt.scatter(maxCoord[1],maxCoord[0],facecolors='none',edgecolors='r', s=kernelsize**2, label=f"max: {maxValue}")
            plt.scatter(minCoord[1],minCoord[0],facecolors='none',edgecolors='b', s=kernelsize**2, label=f"min: {minValue}")
            plt.legend()
            if showplot:
                plt.show()
            if print:
                plt.savefig(self.dirs["png"]+"iiu_acr.png")
                plt.close()
        self.res_IIU = np.round(100 * (1-(maxValue-minValue)/(maxValue+minValue)),1)
        return
    
    def percent_ghosting_ratio(self, showplot=False, print=False):
        thld = utilfunc.getThreshold.findPeak(self.imagedata_loc[6],10)
        thldimg = utilfunc.createThresholdImage(self.imagedata_loc[6], thld/2)
        center = utilfunc.findCenter.centerOfMassFilled(thldimg)

        result, imagemasks, data = acrfunc.psg.calcPSG(self.imagedata_loc[6],thldimg, center)

        resultText = f"center:{data[0]}\ntop:{data[1]}\nbot:{data[3]}\nright:{data[2]}\nleft:{data[4]}"

        if showplot or print:
            plt.imshow(self.imagedata_loc[6], cmap="bone")
            for i in imagemasks:
                plt.imshow(np.ma.masked_array(self.imagedata_loc[6], i))
            plt.gcf().text(0.81, 0.5, resultText, fontsize=10)
            if showplot:
                plt.show()
            if print:
                plt.savefig(self.dirs["png"]+"pgr_acr.png")
                plt.close()

        self.res_PGA = np.round(result,1)
        return

    def low_contrast_object_detectibility(self, method="peaks", showplot=False, print=False):
        # spokesIm11 = methods.questionWindow(analyzer.imagedata_loc[10],"Count spokes")
        measuedResults = acrfunc.lcod.calcLCOD(self.imagedata_loc, method)

        if showplot or print:
            fig, axlist = plt.subplots(1,4)  
            for data,ax in zip(measuedResults,axlist):
                ax.imshow(data[1])
                ax.set_title(f"{data[0]}")
                ax.hlines(data[2], 0, int(data[1].shape[1]-1), colors="red", alpha=0.3)
                ax.vlines(0.14*data[1].shape[1], 0, 359, colors="orange", alpha=0.5)
                ax.vlines(0.44*data[1].shape[1], 0, 359, colors="orange", alpha=0.5)
                ax.vlines(0.75*data[1].shape[1], 0, 359, colors="orange", alpha=0.5)
            if showplot:
                plt.show()
            if print:
                plt.savefig(self.dirs["png"]+"lcod_acr.png")
                plt.close()

        self.res_LCOD = sum([measuedResults[i][0] for i in range(len(measuedResults))])

    @property
    def data_organized(self):
        # Organize data. All tests have to have ran otherwise you get an error.
        if self._data_organized is None:
            self._data_organized = {
                "GA": {
                    "result": self.res_GA,
                    # "deviation": self.res_GA_SD,
                    "criteria": {"min": 185,"max": 195},
                    "unit": "mm",
                    "image": self.dirs["png"]+"ga_acr.png",
                    "display_range": [180,200],
                },
                "LCOD": {
                    "result": self.res_LCOD,
                    "criteria": {"min": 32, "max": 40},
                    "unit": "spokes",
                    "image": self.dirs["png"]+"lcod_acr.png",
                    "display_range": [25  ,41],
                },
                "IIU": {
                    "result": self.res_IIU,
                    "criteria": {"min": 80, "max": 100},
                    "unit": "%",
                    "image": self.dirs["png"]+"iiu_acr.png",
                    "display_range": [0  ,100],
                },
                "STA": {
                    "result": self.res_STA,
                    "criteria": {"min": 4, "max": 6},
                    "unit": "mm",
                    "image": self.dirs["png"]+"sta_acr.png",
                    "display_range": [0  ,15 ],
                },
                "SPA": {
                    "result": self.res_SPA,
                    "criteria": {"min":-4,"max": 4},
                    "unit": "mm",
                    "image": self.dirs["png"]+"sta_acr.png",
                    "display_range": [-10 ,10  ],
                },
                "PGR": {
                    "result": self.res_PGA,
                    "criteria": {"min": 0, "max": 5},
                    "unit": "%",
                    "image": self.dirs["png"]+"pgr_acr.png",
                    "display_range": [0  ,100],
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

        csv_filename = self.dirs["csv"] + f'{self.scannername}_acr.csv'
        write_header = not os.path.isfile(csv_filename)
        
        # Write CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            if write_header:
                header = savedata.keys()
                csv_writer.writerow(header)
            writedata = [savedata[key] for key in savedata.keys()]
            csv_writer.writerow(writedata)

    def _check_criteria(self, test_name, value):
        criteria = self.data_organized.get(test_name).get("criteria")
        if not criteria:
            return False

        if "min" in criteria and value <= criteria["min"]:
            return False
        if "max" in criteria and value >= criteria["max"]:
            return False

        return True

    def create_report(self):
        combined_results_mapping = self.data_organized

        # Creation of PDF
        ## Initialize the PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        ## Add a title page
        pdf.add_page()
        pdf.set_font("Arial", size=24)
        pdf.cell(200, 10, "ACR QA Report", ln=True, align='C')

        ## Add summary table
        pdf.set_font("Arial", size=12)
        pdf.ln(20)  # Add some space
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, "Summary Table", ln=True, align='L')
        pdf.set_font("Arial", size=12)

        pdf.cell(40, 10, "Test", 1)
        pdf.cell(40, 10, "Result", 1)
        pdf.cell(40, 10, "Criteria", 1)
        pdf.cell(40, 10, "Pass/Fail", 1)
        pdf.ln()

        ## Fill Table
        for key, value in combined_results_mapping.items():
            result = value["result"]
            unit = value["unit"]
            deviation = value.get("deviation")

            criteria = f'{value.get("criteria").get("min")} <= {value.get("criteria").get("max")} {unit}'

            ### Format the result with deviation if it exists
            result_str = f"{result}{unit}"
            if deviation is not None:
                result_str += f"+-{deviation}{unit}"

            pass_fail = "Pass" if self._check_criteria(key, result) else "Fail"
            
            pdf.cell(40, 10, key, 1)
            pdf.cell(40, 10, result_str, 1)
            pdf.cell(40, 10, criteria, 1)
            pdf.cell(40, 10, pass_fail, 1)
            pdf.ln()
        
        ## Add pages with images
        pdf.set_font("Arial", size=12)

        for key, value in combined_results_mapping.items():
            image_filename = value["image"]
            metric_name = key
            metric_value = value["result"]
            metric_unit = value["unit"]

            metric_deviation = f'+-{value["deviation"]}' if value.get("deviation") else ""

            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(200, 10, key, ln=True, align='L')
            pdf.image(image_filename, x=80, y=20, w=120)
            # pdf.set_xy(20, 30)
            pdf.ln(20)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"{metric_name}: {metric_value}{metric_deviation} {metric_unit}", ln=True)

        # Save the PDF
        pdf.output(self.dirs["srp"] + f"{self.scannername}_{self.creationdate}_QAreport.pdf")

    def _readcsv(self):
        csv_filename = self.dirs["csv"] + f'{self.scannername}_acr.csv'
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
            plt.savefig(self.dirs["png"]+f"Longterm_{testname}_acr.png")
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
            pdf.image(self.dirs["png"]+f"Longterm_{testname}_acr.png", x=x_offset, y=y_offset, w=width, h=height)
            pdf.set_xy(x_offset, y_offset + height + 5)
            # pdf.cell(200, 100, testname, ln=True)
            y_offset += height + 20
        
        pdf.output(self.dirs["lrp"] + f'{self.scannername}_longterm_report_acr.pdf')

    def runall(self):
        self.geometric_accuracy(False, True) # Done.
        self.slice_position_accuracy(False, True) # Done.
        self.image_intensity_uniformity(False, True) # Done.
        self.percent_ghosting_ratio(False, True) # Done.
        self.low_contrast_object_detectibility("edges1", False, True) # Done.
        self.slice_thickness_accuracy(False, True) # Done.

        self.add2csv()
        self.create_report()
        self.create_longterm_report()