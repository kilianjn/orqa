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
    def __init__(self, data) -> None:
        self.imagedata_loc  = data.imagedata[0]
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

        self.results        = {}

        self.criteria       = {
            "Geometric Accuracy":                 {"min": 142,"max": 148},
            "Low Contrast Object Detectability":  {"min": 6},
            "Image Intensity Uniformity":         {"min": 80},
            "Slice Thickness":                    {"min": 4, "max": 6},
            "Slice Position":                     {"min":-4,"max": 4},
            "Percent Signal Ghosting":            {"max": 5}
        }

        self.longtermdata   = {}

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
                plt.savefig("test1.png")
                plt.close()

        self.res_GA = np.mean([measureResults1[0],measureResults2[0],measureResults3[0]])
            

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
        self.res_STA = length

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
                plt.savefig("test6.png")
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
                plt.savefig("test2.png")
                plt.close()

        self.res_SPA = np.mean([measureResults1[0],measureResults2[0]])

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
                plt.savefig("test3.png")
                plt.close()
        self.res_IIU = 100 * (1-(maxValue-minValue)/(maxValue+minValue))
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
                plt.savefig("test4.png")
                plt.close()

        self.res_PGA = result
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
                plt.savefig("test5.png")
                plt.close()

        self.res_LCOD = sum([measuedResults[i][0] for i in range(len(measuedResults))])



    def add2csv(self):
        # csv header and data
        self.results = {
            "Date of measurement":                  f"{self.metadata[0x00080020].value}",
            "Time of measurement":                  f"{self.metadata[0x00080031].value}",
            "Time of evaluation":                   f"{datetime.now()}",
            "Geometric Accuracy":                   f"{self.res_GA}",
            "Low Contrast Object Detectability":    f"{self.res_LCOD}",
            "Image Intensity Uniformity":           f"{self.res_IIU}",
            "Slice Thickness":                      f"{self.res_STA}",
            "Slice Position":                       f"{self.res_SPA}",
            "Percent Signal Ghosting":              f"{self.res_PGA}"
        }

        csv_filename = f'{self.scannername}_acr.csv'
        write_header = not os.path.isfile(csv_filename)
        
        # Write CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            if write_header:
                header = self.results.keys()
                csv_writer.writerow(header)
            writedata = [self.results[key] for key in self.results.keys()]
            csv_writer.writerow(writedata)


    def _check_criteria(self, test_name, value):
        criteria = self.criteria.get(test_name)
        if not criteria:
            return False

        if "min" in criteria and value <= criteria["min"]:
            return False
        if "max" in criteria and value >= criteria["max"]:
            return False

        return True

    @property
    def _results_organized(self):
        combined_results_mapping = {
            "Resolution": {
                "result": self.res_RES,
                "deviation": self.res_RES_SD,
                "criteria": f'{self.criteria["Resolution"]["min"]} - {self.criteria["Resolution"]["max"]} mm',
                "unit": "mm",
                "image": "francis_res.png",
            },
            "Diameter": {
                "result": self.res_GA,
                "deviation": self.res_GA_SD,
                "criteria": f'{self.criteria["Diameter"]["min"]} - {self.criteria["Diameter"]["max"]} mm',
                "unit": "mm",
                "image": "francis_size.png",
            },
            "Low Contrast": {
                "result": self.res_LCOD,
                "criteria": f'> {self.criteria["Low Contrast"]["min"]} spokes',
                "unit": "spokes",
                "image": "francis_contrast.png",
            },
            "Image Uniformity": {
                "result": self.res_IIU,
                "criteria": f'> {self.criteria["Image Uniformity"]["min"]}%',
                "unit": "%",
                "image": "francis_uniformity.png",
            },
            "Slice Thickness": {
                "result": self.res_STA,
                "criteria": f'{self.criteria["Slice Thickness"]["min"]} - {self.criteria["Slice Thickness"]["max"]} mm',
                "unit": "mm",
                "image": "francis_thickness.png",
            },
            "Slice Position": {
                "result": self.res_SPA,
                "criteria": f'{self.criteria["Slice Position"]["min"]} - {self.criteria["Slice Position"]["max"]} mm',
                "unit": "mm",
                "image": "francis_position.png",
            },
            "Grid Angle": {
                "result": self.res_Grid_angle,
                "criteria": f'{self.criteria["Grid Angle"]["min"]} - {self.criteria["Grid Angle"]["max"]} degrees',
                "unit": "degrees",
                "image": "francis_grid.png",
            },
            "Grid Size": {
                "result": self.res_Grid_size,
                "criteria": f'{self.criteria["Grid Size"]["min"]} - {self.criteria["Grid Size"]["max"]} mm2',
                "unit": "mm2",
                "image": "francis_grid.png",
            },
            "Ghosting": {
                "result": self.res_Ghosting,
                "criteria": f'< {self.criteria["Ghosting"]["max"]}%',
                "unit": "%",
                "image": "francis_ghosting.png",
            }
        }
        return combined_results_mapping

    def create_report(self):
        combined_results_mapping = self._results_organized

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
            criteria = value["criteria"]
            unit = value["unit"]
            deviation = value.get("deviation")

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
        pdf.output(f"{self.scannername}_{self.creationdate}_QAreport.pdf")


    def _readcsv(self):
        csv_filename = f'{self.scannername}_acr.csv'
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

        tests_and_yrange = {   
            "Resolution":           [0.3,2  ],
            "Geometric Accuracy":   [140,150],
            "Low Contrast":         [0  ,9  ],
            "Image Uniformity":     [0  ,100],
            "Slice Thickness":      [0  ,10 ],
            "Slice Position":       [-5 ,5  ],
            "Grid Size":            [28 ,44 ],
            "Ghosting":             [0  ,100]
        }

        for testname in tests_and_yrange.keys():
            xdata_raw = self.longtermdata["Date of measurement"]
            dates = [datetime.strptime(date, '%Y%m%d') for date in xdata_raw]

            ydata = [float(i) for i in self.longtermdata[testname]]

            # Plot the data
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.gcf().autofmt_xdate()  # Auto format date labels
            plt.plot(dates, ydata, marker='o')
            plt.ylim(tests_and_yrange[testname])
            # plt.ylim([0,10])

            # Adding titles and labels
            plt.title(f'Longitudinal plot for {testname}')
            plt.ylabel('Testresult')

            # Show the plot
            # plt.show()
            # Save figure
            plt.savefig(f"Longterm_{testname}.png")
            plt.close()

        

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # # Add a title page
        # pdf.add_page()
        # pdf.set_font("Arial", size=24)
        # pdf.cell(200, 10, "Francis longterm Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        x_offset = 10
        y_offset = 30
        width = 90
        height = 70

        for i, testname in enumerate(tests_and_yrange.keys()):
            if i % 3 == 0:
                pdf.add_page()
                y_offset = 30
            pdf.image(f"Longterm_{testname}.png", x=x_offset, y=y_offset, w=width, h=height)
            pdf.set_xy(x_offset, y_offset + height + 5)
            # pdf.cell(200, 100, testname, ln=True)
            y_offset += height + 20
        
        pdf.output(f'{self.scannername}_longterm_report.pdf')

