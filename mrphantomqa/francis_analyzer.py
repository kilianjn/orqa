from .utils.methods import functions as utilfunc
from .francis.methods import functions as francisfunc

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from datetime import datetime
from fpdf import FPDF

class francisAnalyzer:
    def __init__(self, data) -> None:
        self.imagedata      = data.imagedata[0]
        self.metadata       = data.metadata if hasattr(data, 'metadata') else None

        self.scannername    = self.metadata[0x00080080].value
        self.creationdate   = self.metadata[0x00080012].value
        self.spacing        = self.metadata[0x52009230][0][0x00289110][0][0x00280030].value

        self.res_RES        = None  # Resolution
        self.res_RES_SD     = None  # Resolution SD
        self.res_GA         = None  # Geometric lenghts
        self.res_GA_SD      = None  # Geometric lengths SD
        self.res_LCOD       = None  # Low contrast
        self.res_IIU        = None  # Image uniformity
        self.res_STA        = None  # Slice thickness
        self.res_SPA        = None  # Slice position
        self.res_Grid_size  = None  # Grid size
        self.res_Grid_angle = None  # Grid angle
        self.res_Ghosting   = None  # Percent Ghosting Ratio

        self.results        = []
        self.criteria       = {
            "Resolution": {"min": 0.8, "max": 1.2},
            "ResolutionSD": {"max": 2},
            "Diameter": {"min": 142,"max": 148},
            "Diameter SD": {"max": 2},
            "Low Contrast": {"min": 6},
            "Image Uniformity": {"min": 80},
            "Slice Thickness": {"min": 4, "max": 6},
            "Slice Position": {"min":-4,"max": 4},
            "Grid Angle": {"min": 87,"max": 93},
            "Grid Size": {"min":28, "max": 44},
            "Ghosting": {"max": 5}
        }

        self.longtermdata = {}

    def resolution(self, showplot=False, savefig=False):
        
        # Interpolation for smoother lines
        img = utilfunc.interpolateImage(self.imagedata[2])

        # Initial location of centerpoint
        thld = int(0.8 * utilfunc.getThreshold.findPeak(img,11))
        thld_img = utilfunc.createThresholdImage(img,thld)
        centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)
        
        diameter = np.mean([utilfunc.measureDistance(thld_img,centerpoint,x) for x in [45,135]])

        # Finding exact center using small mask over centerdot
        centermask = utilfunc.circularROI(thld_img,centerpoint,int(0.04*diameter))
        tmp_center_thldImg = thld_img.copy()
        tmp_center_thldImg[centermask] = 0
        centerpoint = utilfunc.findCenter.centerOfMass(tmp_center_thldImg)

        # Donutmask
        mask = utilfunc.circularROI(thld_img,centerpoint,int(0.41*diameter))
        innermask = utilfunc.circularROI(thld_img,centerpoint,int(0.03*diameter))
        mask1 = np.ma.mask_or(mask, ~innermask)
        img_donutmask = np.ma.masked_array(img,mask1)

        # Creating Threshold image using only donutimage for the choice of threshold
        thld1 = utilfunc.getThreshold.otsuMethod(img_donutmask)
        thldimg1 = utilfunc.createThresholdImage(img_donutmask,thld1)

        # Finding the longest length of each line along the trinsngles
        longestLength = []
        llCords = []
        for i in range(12):
            tempMaxLength = 0
            tempCoords = 0
            for j in range(-92+30*i, -88+30*i): # range of angles to account for small angle offsets
                length, points = francisfunc.res.measureDistance(thldimg1,centerpoint,j,[x/2 for x in self.spacing])
                if length > tempMaxLength:
                    tempMaxLength = length
                    tempCoords = points
            longestLength.append(tempMaxLength)
            llCords.append(tempCoords)
        if showplot or savefig:
            plt.imshow(img)
            plt.scatter(centerpoint[1],centerpoint[0], marker="x", color="red")
            for i in range(len(llCords)):
                plt.plot(llCords[i][:,1],llCords[i][:,0], color="red")
            if showplot:
                plt.show()
            elif savefig:
                plt.savefig("francis_res.png")
                plt.close()
        
        self.res_RES = np.round((5 - (np.median(longestLength)/50 * 5))*2, 2)
        self.res_RES_SD = np.round((5 - (np.std(longestLength)/50 * 5))*2, 2)
        return

    def low_contrast(self, showplot=False, savefig=False):

        img = utilfunc.interpolateImage(self.imagedata[4])

        thld = int(utilfunc.getThreshold.findPeak(img,10))
        thld_img = utilfunc.createThresholdImage(img,thld)
        centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)

        images = []

        ## Cutout center and get appropiate masks
        mask = utilfunc.cutoutStructureMask(thld_img, centerpoint)
        mask = utilfunc.removeHoles(thld_img)
        thldMaskedImg = np.ma.masked_array(thld_img, mask)

        ## Data for mask
        center_new = utilfunc.findCenter.centerOfMass(mask)
        diameter = utilfunc.measureDistance(mask,center_new,0,[1,1])
        radius = diameter/2

        circMask = utilfunc.circularROI(img, center_new,int(0.65*radius))

        # Actual Computation
        cutoutImage = np.ma.masked_array(img, circMask)
        lineArraysByAngle, _ = francisfunc.lcod.radialTrafo_LCOD(cutoutImage, center_new, -105)

        # Generate edgeimage
        edgeImage = []
        for angle in range(lineArraysByAngle.shape[0]):
            edgeImage.append(np.convolve(lineArraysByAngle[angle], [1,0,-1], "valid"))
        edgeImage = np.array(edgeImage)
        lineArraysByAngle = edgeImage[:,:int(edgeImage.shape[1]*0.95)]

        images.append(lineArraysByAngle)

        countedSpokes, spokePosition = francisfunc.lcod.countEdges(lineArraysByAngle)


        if showplot or savefig:
            plt.subplot(121)
            plt.imshow(img, cmap="gray")
            plt.imshow(cutoutImage)

            plt.subplot(122)
            plt.imshow(lineArraysByAngle)
            plt.title(f"{countedSpokes} spokes counted")
            for i in spokePosition:
                plt.hlines(i, 0, int(lineArraysByAngle.shape[1]-1), colors="red", alpha=0.3)
            if showplot:
                plt.show()
            if savefig:
                plt.savefig("francis_contrast.png")
                plt.close()
        
        self.res_LCOD = countedSpokes

    def uniformity(self, showplot=False, savefig=False):
        img = self.imagedata[8]
        thld = int(0.8 * utilfunc.getThreshold.otsuMethod(img))
        thld_img = utilfunc.createThresholdImage(img,thld)
        centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)

        diameter = np.mean([utilfunc.measureDistance(thld_img,centerpoint,x) for x in [45,135]])
        roiDiameter = int(0.32*diameter)
        centermask = utilfunc.circularROI(img, centerpoint, roiDiameter)


        kernelsize = int(0.15 * roiDiameter)
        kernel = np.ones((kernelsize,kernelsize))/(kernelsize**2)
        convolvedImg = utilfunc.convolveImage(img, kernel)

        convolvedMaskedImg = np.ma.masked_array(convolvedImg, centermask)

        maxValue, minValue, maxCoord, minCoord = francisfunc.iiu.searchForCircularSpots(convolvedMaskedImg)
        if showplot or savefig:
            plt.imshow(img, cmap="gray")
            plt.imshow(convolvedMaskedImg)
            plt.scatter(maxCoord[1],maxCoord[0],facecolors='none',edgecolors='r', s=10**2, label=f"max: {maxValue}")
            plt.scatter(minCoord[1],minCoord[0],facecolors='none',edgecolors='b', s=10**2, label=f"min: {minValue}")
            plt.legend()
            if showplot:
                plt.show()
            if savefig:
                plt.savefig("francis_uniformity.png")
                plt.close()

        self.res_IIU = np.round(100 * (1-(maxValue-minValue)/(maxValue+minValue)),2)

    def thickness(self, showplot=False, savefig=False):
        img = self.imagedata[6]
        rect_img = francisfunc.sta.cutoutRect(img)

        length, coords, border = francisfunc.sta.measureLength(rect_img, self.spacing[1]) # Anstatt image border einsetzen.
        self.res_STA = np.round(length,2)

        if showplot or print:
            y_half = rect_img.shape[0]/2
            plt.vlines(coords[0],ymin=0,ymax=y_half)
            plt.vlines(coords[1],ymin=0,ymax=y_half)
            plt.vlines(coords[2],ymin=y_half,ymax=y_half*2, color="red")
            plt.vlines(coords[3],ymin=y_half,ymax=y_half*2, color="red")
            plt.imshow(rect_img, cmap="bone")
            if showplot:
                plt.show()
            if savefig:
                plt.savefig("francis_thickness.png")
                plt.close()
        pass

    def position(self, showplot=False, savefig=False):
        img = self.imagedata[6]
        rectimg = francisfunc.spa.cutoutRect(img)
        length_diff, lengths = francisfunc.spa.getPositionDifference(rectimg)

        if showplot or savefig:
            plt.imshow(rectimg)
            plt.hlines(lengths[0]-0.5 ,0,int(rectimg.shape[1]/2))
            plt.hlines(lengths[1]-0.5,int(rectimg.shape[1]/2), rectimg.shape[1]-1)
            plt.vlines(int(rectimg.shape[1]/2), lengths[0]-0.5, lengths[1]-0.5, colors="red")
            if showplot:
                plt.show()
            elif savefig:
                plt.savefig("francis_position")
                plt.close()

        self.res_SPA = np.round(length_diff * self.spacing[0],2)
        
        pass

    def grid(self, showplot=False, savefig=False):
        img = self.imagedata[0]
        img_grid_pre = francisfunc.grid.cutoutSquare(img)

        img_grid = francisfunc.grid.imagePreProcessing(img_grid_pre, False)

        ## Grid Detection using the Hough Transform
        dist_lines, angle_cross, lines = francisfunc.grid.gridDetect(img_grid)

        squaresize = (self.spacing[0] * dist_lines[0]) * (self.spacing[1] * dist_lines[1])

        francisfunc.grid.printImage(img_grid_pre, lines, showplot, savefig)

        self.res_Grid_size = np.round(squaresize,2)
        self.res_Grid_angle = np.round(np.rad2deg(angle_cross),2)

    def size(self, showplot=False, savefig=False):
        img = self.imagedata[2]

        # Initial location of centerpoint
        thld = int(0.8 * utilfunc.getThreshold.findPeak(img,10))
        thld_img = utilfunc.createThresholdImage(img,thld)
        centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)
        diameter = np.mean([utilfunc.measureDistance(thld_img,centerpoint,x) for x in [45,135]])

        # Finding exact center using small mask over centerdot
        centermask = utilfunc.circularROI(thld_img,centerpoint,int(0.04*diameter))
        tmp_center_thldImg = thld_img.copy()
        tmp_center_thldImg[centermask] = 0
        centerpoint = utilfunc.findCenter.centerOfMass(tmp_center_thldImg)


        measureResults1 = francisfunc.ga.measureDistance(thld_img, centerpoint,  45, self.spacing)
        measureResults2 = francisfunc.ga.measureDistance(thld_img, centerpoint, -45, self.spacing)

        if showplot or print:
            plt.imshow(img, cmap='gray')
            for len, coord in [measureResults1, measureResults2]:
                plt.scatter(coord[:, 1], coord[:, 0], c='red', marker='x')
                plt.plot(coord[:, 1], coord[:, 0], label=f"Length = {len}")
            plt.scatter(centerpoint[1], centerpoint[0], c='blue', marker='x')
            plt.legend()
            if showplot:
                plt.show()
            if savefig:
                plt.savefig("francis_size.png")
                plt.close()
        
        self.res_GA = np.round(np.mean([measureResults1[0], measureResults2[0]]),2)
        self.res_GA_SD = np.round(np.std([measureResults1[0], measureResults2[0]]),2)
        return

    def ghosting(self, showplot=False, print=False):
        thld = utilfunc.getThreshold.findPeak(self.imagedata[8],10)
        thldimg = utilfunc.createThresholdImage(self.imagedata[8], thld/2)
        center = utilfunc.findCenter.centerOfMassFilled(thldimg)

        result, imagemasks, data = francisfunc.psg.calcPSG(self.imagedata[8],thldimg, center)

        resultText = f"center:{data[0]}\ntop:{data[1]}\nbot:{data[3]}\nright:{data[2]}\nleft:{data[4]}"

        if showplot or print:
            plt.imshow(self.imagedata[8], cmap="bone")
            for i in imagemasks:
                plt.imshow(np.ma.masked_array(self.imagedata[8], i))
            plt.gcf().text(0.81, 0.5, resultText, fontsize=10)
            if showplot:
                plt.show()
            if print:
                plt.savefig("francis_ghosting.png")
                plt.close()

        self.res_Ghosting = np.round(100 * result, 2)
        return

    def _organize_data(self):
        header = ["Date of measurement",
                  "Time of Measurement",
                  "Resolution",
                  "ResolutionSD",
                  "Geometric Accuracy",
                  "Geometric Accuracy SD",
                  "Low Contrast",
                  "Image Uniformity",
                  "Slice Thickness",
                  "Slice Position",
                  "Grid Angle",
                  "Grid Size",
                  "Ghosting"
                  ]
        self.results.append(header)
        
        data = [f"{self.metadata[0x00080020].value}",
                f"{self.metadata[0x00080031].value}",
                f"{self.res_RES}",
                f"{self.res_RES_SD}",
                f"{self.res_GA}",
                f"{self.res_GA_SD}",
                f"{self.res_LCOD}",
                f"{self.res_IIU}",
                f"{self.res_STA}",
                f"{self.res_SPA}",
                f"{self.res_Grid_angle}",
                f"{self.res_Grid_size}",
                f"{self.res_Ghosting}"
                ]

        self.results.append(data)

    def add2csv(self):

        self._organize_data()
        
        # Create a CSV file
        csv_filename = f'{self.scannername}.csv'
        write_header = not os.path.isfile(csv_filename)
        
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            if write_header:
                header = self.results[0]  # assuming self.results[0] is a list of headers
                csv_writer.writerow(header)
            writedata = self.results[1]  # assuming self.results[1] is a list of data
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

    def create_report(self):
        # Initialize the PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add a title page
        pdf.add_page()
        pdf.set_font("Arial", size=24)
        pdf.cell(200, 10, "Francis Analyzer Report", ln=True, align='C')

        # Add summary table
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

        results_mapping = {
            "Resolution": (self.res_RES, "0.8 - 1.2 mm", "mm"),
            "Resolution SD": (self.res_RES_SD, "< 2 mm", "mm"),
            "Diameter": (self.res_GA, "142 - 148 mm", "mm"),
            "Diameter SD": (self.res_GA_SD, "< 2 mm", "mm"),
            "Low Contrast": (self.res_LCOD, "> 6 spokes", "spokes"),
            "Image Uniformity": (self.res_IIU, "> 80%", "%"),
            "Slice Thickness": (self.res_STA, "4 - 6 mm", "mm"),
            "Slice Position": (self.res_SPA, "-4 to 4 mm", "mm"),
            "Grid Angle": (self.res_Grid_angle, "87 - 93 degrees", "degrees"),
            "Grid Size": (self.res_Grid_size, "28 - 44 mm2", "mm2"),
            "Ghosting": (self.res_Ghosting, "< 5%", "%")
        }

        for key, value in results_mapping.items():
            result, criteria, unit = value
            pass_fail = "Pass" if self._check_criteria(key, result) else "Fail"
            pdf.cell(40, 10, key, 1)
            pdf.cell(40, 10, f"{result} {unit}", 1)
            pdf.cell(40, 10, criteria, 1)
            pdf.cell(40, 10, pass_fail, 1)
            pdf.ln()

        # Add each result with its corresponding image
        pdf.set_font("Arial", size=12)

        results_mapping_with_images = {
            "Resolution": ("francis_res.png", [("Resolution", self.res_RES, "mm"), ("Resolution SD", self.res_RES_SD, "mm")]),
            "Geometric Accuracy": ("francis_size.png", [("Diameter", self.res_GA, "mm"), ("Diameter SD", self.res_GA_SD, "mm")]),
            "Low Contrast": ("francis_contrast.png", [("Low Contrast", self.res_LCOD, "spokes")]),
            "Image Uniformity": ("francis_uniformity.png", [("Image Uniformity", self.res_IIU, "%")]),
            "Slice Thickness": ("francis_thickness.png", [("Slice Thickness", self.res_STA, "mm")]),
            "Slice Position": ("francis_position.png", [("Slice Position", self.res_SPA, "mm")]),
            "Grid": ("francis_grid.png", [("Grid Size", self.res_Grid_size, "mm2"), ("Grid Angle", self.res_Grid_angle, "degrees")]),
            "Ghosting": ("francis_ghosting.png", [("Ghosting", self.res_Ghosting, "%")])
        }

        for key, value in results_mapping_with_images.items():
            image, metrics = value
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(200, 10, key, ln=True, align='L')
            pdf.image(image, x=80, y=20, w=120)
            # pdf.set_xy(20, 30)
            pdf.ln(20)
            pdf.set_font("Arial", size=12)
            for metric_name, metric_value, unit in metrics:
                pdf.cell(200, 10, f"{metric_name}: {metric_value} {unit}", ln=True)

        # Save the PDF
        pdf.output(f"{self.scannername}_{self.creationdate}_QAreport.pdf")

    def _readcsv(self):
        csv_filename = f'{self.scannername}.csv'
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

        tests = {   "Resolution":           [0.3,2  ],
                    "Geometric Accuracy":   [140,150],
                    "Low Contrast":         [0  ,9  ],
                    "Image Uniformity":     [0  ,100],
                    "Slice Thickness":      [0  ,10 ],
                    "Slice Position":       [-5 ,5  ],
                    "Grid Size":            [28 ,44 ],
                    "Ghosting":             [0  ,100]
        }

        
        for testname in tests.keys():
            xdata_raw = self.longtermdata["Date of measurement"]
            dates = [datetime.strptime(date, '%Y%m%d') for date in xdata_raw]
            ydata = [float(i) for i in self.longtermdata[testname]]

            # Plot the data
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            plt.gcf().autofmt_xdate()  # Auto format date labels
            plt.plot(dates, ydata, marker='o')
            plt.ylim(tests[testname])
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

        for i, testname in enumerate(tests.keys()):
            if i % 3 == 0:
                pdf.add_page()
                y_offset = 30
            pdf.image(f"Longterm_{testname}.png", x=x_offset, y=y_offset, w=width, h=height)
            pdf.set_xy(x_offset, y_offset + height + 5)
            # pdf.cell(200, 100, testname, ln=True)
            y_offset += height + 20
        
        pdf.output(f'{self.scannername}_longterm_report.pdf')
        pass
