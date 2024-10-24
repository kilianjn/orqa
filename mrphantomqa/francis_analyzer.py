from .utils.methods import functions as utilfunc
from .francis.methods import functions as francisfunc

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os

from datetime import datetime
from fpdf import FPDF

# Debug
from .utils import viewer as viw

class francisAnalyzer:
    def __init__(self, data, workdir) -> None:
        self.imagedata          = data.imagedata[0] if hasattr(data, 'imagedata') else None
        self.metadata           = data.metadata if hasattr(data, 'metadata') else None

        self.spacing            = [1,1]
        # Get correct pixel spacing
        if self.metadata.get(0x52009230) is not None: # Case Enhanced save
            self.spacing        = self.metadata[0x52009230][0][0x00289110][0][0x00280030].value
        if self.metadata.get(0x00280030) is not None: # Case INteroperability
            self.spacing        = [float(i) for i in self.metadata[0x00280030].value]

        self.res_RES                = None  # Resolution
        self.res_RES_SD             = None  # Resolution SD
        self.res_GA                 = None  # Geometric lenghts
        self.res_GA_SD              = None  # Geometric lengths SD
        self.res_LCOD               = None  # Low contrast
        self.res_IIU                = None  # Image uniformity
        self.res_STA                = None  # Slice thickness
        self.res_SPA                = None  # Slice position
        self.res_Grid_size          = None  # Grid size
        self.res_Grid_angle         = None  # Grid angle
        self.res_Grid_lines_hori    = None  # Detected grid lines horizontal
        self.res_Grid_lines_vert    = None  # Detected grid lines vertical
        self.res_Ghosting           = None  # Percent Ghosting Ratio

        # # Evaluator Part - put in own class at some point
        self.scannername        = self.metadata[0x00080080].value
        self.creationdate       = self.metadata[0x00080012].value

        self.workdir            = workdir
        while True:
            if os.path.exists(self.workdir):
                break
            else:
                print(f"Path to given working directory does not exist ({self.workdir}). Try anew.")
                self.workdir = str(input("Type the path of the desired working directory: \n"))

        self._data_organized    = None  # All important data and metadata is stored here to create the corresponding reports.

        self.longtermdata       = {}    # Data from CSV files is stored here.

        # Make folders
        self.dirs               = {
            "png"   : os.path.join(workdir, "francis_reports", self.scannername, "imgs", ""),
            "csv"   : os.path.join(workdir, "francis_reports", self.scannername, ""),
            "srp"   : os.path.join(workdir, "francis_reports", self.scannername, "single_reports", ""),
            "lrp"   : os.path.join(workdir, "francis_reports", self.scannername, "")
        }

        for filetype, dir_to_save_to in self.dirs.items():
            if not os.path.exists(dir_to_save_to):
                os.makedirs(dir_to_save_to, exist_ok=True)

    def resolution(self, showplot=False, savefig=False, offsetx=0, offsety=0):
        
        # Interpolation for smoother lines
        img = utilfunc.interpolateImage(self.imagedata[2])

        # Initial location of centerpoint
        thld = int(utilfunc.getThreshold.otsuMethod(img))
        thld_img = utilfunc.createThresholdImage(img,thld)
        centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)

        diameter = np.mean([utilfunc.measureDistance(thld_img,centerpoint,x) for x in [45,135]])

        # Finding exact center using small mask over centerdot
        centermask = utilfunc.circularROI(thld_img,centerpoint,int(0.04*diameter))
        tmp_center_thldImg = thld_img.copy()
        tmp_center_thldImg[centermask] = 0
        centerpoint = utilfunc.findCenter.centerOfMass(tmp_center_thldImg)

        # OffsetDebug
        centerpoint = (centerpoint[0]+offsety,centerpoint[1]-offsetx)

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
            plt.xlabel("Left to right")
            plt.ylabel("Posterior to anterior")

            for i in range(len(llCords)):
                plt.plot(llCords[i][:,1],llCords[i][:,0], color="red")
            if showplot:
                plt.show()
            elif savefig:
                plt.savefig(self.dirs["png"]+"francis_res.png")
                plt.close()
        
        self.res_RES = np.round((5 - (np.median(longestLength)/10))*2, 1)
        self.res_RES_SD = np.round(np.std([(5 - (i/10))*2 for i in longestLength]),1)
        # self.res_RES_SD = np.round((5 - (np.std(longestLength)/50 * 5))*2, 1)
        return

    def low_contrast(self, showplot=False, savefig=False, offsety=0,offsetx=0):

        img = utilfunc.interpolateImage(self.imagedata[4])

        thld = int(utilfunc.getThreshold.otsuMethod(img)) #90 percent to include little more of edges into thresholding
        thld_img = utilfunc.createThresholdImage(img,thld)
        centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)
        # OffsetDebug
        centerpoint = (centerpoint[0]+offsety,centerpoint[1]-offsetx)

        images = []

        # Global centerpoint approach
        diameter = np.mean([utilfunc.measureDistance(thld_img,centerpoint,x) for x in [45,135]])
        radius = diameter/2 * 0.65

        # # Masked approcah
        # ## Cutout center and get appropiate masks
        # mask = ~utilfunc.cutoutStructureMask(thld_img, centerpoint)
        # mask = utilfunc.removeHoles(mask)
        # centerpoint = utilfunc.findCenter.centerOfMass(mask)
        # diameter = utilfunc.measureDistance(mask,centerpoint,0,[1,1])
        # radius = diameter/2 * 0.9


        circMask = utilfunc.circularROI(img, centerpoint,radius)

        # Actual Computation
        cutoutImage = np.ma.masked_array(img, circMask)
        lineArraysByAngle, _ = francisfunc.lcod.radialTrafo_LCOD(cutoutImage, centerpoint, -105)

        # Generate edgeimage
        edgeImage = []
        for angle in range(lineArraysByAngle.shape[0]):
            edgeImage.append(np.convolve(lineArraysByAngle[angle], [1,1,1,0,-1,-1,-1], "valid"))
        edgeImage = np.array(edgeImage)
        lineArraysByAngle = edgeImage[:,:int(edgeImage.shape[1]*0.95)]

        images.append(lineArraysByAngle)

        countedSpokes, spokePosition = francisfunc.lcod.countEdges(lineArraysByAngle)
        distances = []
        for j in range(4): # Divide in 3 areas for peaks
            distances.append(int((0.08 + 0.23*j)*lineArraysByAngle.shape[1]))
        distances.append(lineArraysByAngle.shape[1]-1)

        if showplot or savefig:
            plt.subplot(121)
            plt.imshow(img, cmap="gray")
            plt.imshow(cutoutImage)
            plt.xlabel("Left to right")
            plt.ylabel("Posterior to anterior")

            plt.subplot(122)
            plt.imshow(lineArraysByAngle)
            plt.vlines(distances,0,lineArraysByAngle.shape[0]-1, colors="orange", alpha=0.3)
            plt.title(f"{countedSpokes} spokes counted")
            plt.xlabel("Distance from center [pixel]")
            plt.ylabel("Angle [deg]")
            for i in spokePosition:
                plt.hlines(i, 0, int(lineArraysByAngle.shape[1]-1), colors="red", alpha=0.3)
            if showplot:
                plt.show()
            if savefig:
                plt.savefig(self.dirs["png"]+"francis_contrast.png")
                plt.close()
        
        self.res_LCOD = countedSpokes

    def uniformity(self, showplot=False, savefig=False):
        img = self.imagedata[8]
        thld = int(0.8 * utilfunc.getThreshold.otsuMethod(img))
        thld_img = utilfunc.createThresholdImage(img,thld)
        centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)

        diameter = np.mean([utilfunc.measureDistance(thld_img,centerpoint,x) for x in [45,135]])
        roiRadius = int(0.7*(diameter/2))
        centermask = utilfunc.circularROI(img, centerpoint, roiRadius)


        kernelsize = int(np.round(10/self.spacing[0],0)) # 1cm2 area of rect kernel
        kernel = np.ones((kernelsize,kernelsize))/(kernelsize**2)
        convolvedImg = utilfunc.convolveImage(img, kernel)

        convolvedMaskedImg = np.ma.masked_array(convolvedImg, centermask)

        maxValue, minValue, maxCoord, minCoord = francisfunc.iiu.searchForCircularSpots(convolvedMaskedImg)
        if showplot or savefig:
            plt.imshow(img, cmap="gray")
            plt.imshow(convolvedMaskedImg)
            plt.scatter(maxCoord[1],maxCoord[0],facecolors='none',edgecolors='r', s=10**2, label=f"max: {maxValue}")
            plt.scatter(minCoord[1],minCoord[0],facecolors='none',edgecolors='b', s=10**2, label=f"min: {minValue}")
            plt.xlabel("Left to right")
            plt.ylabel("Posterior to anterior")
            plt.legend()
            if showplot:
                plt.show()
            if savefig:
                plt.savefig(self.dirs["png"]+"francis_uniformity.png")
                plt.close()

        self.res_IIU = np.round(100 * (1-(maxValue-minValue)/(maxValue+minValue)),1)

    def thickness(self, showplot=False, savefig=False):
        img = self.imagedata[6]
        rect_img = francisfunc.sta.cutoutRect(img)

        length, coords, border = francisfunc.sta.measureLength(rect_img, self.spacing[1]) 
        self.res_STA = np.round(length,1)

        if showplot or savefig:
            y_half = rect_img.shape[0]/2
            plt.vlines(coords[0],ymin=0,ymax=y_half)
            plt.vlines(coords[1],ymin=0,ymax=y_half)
            plt.vlines(coords[2],ymin=y_half,ymax=y_half*2, color="red")
            plt.vlines(coords[3],ymin=y_half,ymax=y_half*2, color="red")
            plt.imshow(rect_img, cmap="bone")
            plt.xlabel("Left to right")
            plt.ylabel("Posterior to anterior")
            if showplot:
                plt.show()
            if savefig:
                plt.savefig(self.dirs["png"]+"francis_thickness.png")
                plt.close()
        pass

    def position(self, showplot=False, savefig=False):
        img = self.imagedata[6]
        rectimg = francisfunc.spa.cutoutRect(img)
        length_diff, lengths = francisfunc.spa.getPositionDifference(~rectimg)

        if showplot or savefig:
            plt.imshow(rectimg)
            plt.hlines(lengths[0]-0.5 ,0,int(rectimg.shape[1]/2))
            plt.hlines(lengths[1]-0.5,int(rectimg.shape[1]/2), rectimg.shape[1]-1)
            plt.vlines(int(rectimg.shape[1]/2), lengths[0]-0.5, lengths[1]-0.5, colors="red")
            plt.xlabel("Left to right")
            plt.ylabel("Posterior to anterior")
            if showplot:
                plt.show()
            elif savefig:
                plt.savefig(self.dirs["png"]+"francis_position.png")
                plt.close()

        self.res_SPA = np.round((length_diff/2) * self.spacing[0],1)
        
        pass

    def grid(self, showplot=False, savefig=False):
        img = self.imagedata[0]
        img_grid_pre = francisfunc.grid.cutoutSquare(img)

        img_grid = francisfunc.grid.imagePreProcessing(img_grid_pre, False)

        ## Grid Detection using the Hough Transform
        dist_lines, angle_cross, lines = francisfunc.grid.gridDetect(img_grid)

        squaresize = (self.spacing[0] * dist_lines[0]) * (self.spacing[1] * dist_lines[1])

        francisfunc.grid.printImage(img_grid_pre, lines, showplot, savefig, self.dirs["png"])

        self.res_Grid_size = np.round(squaresize,1)
        self.res_Grid_angle = np.round(np.rad2deg(angle_cross),2)
        self.res_Grid_lines_hori = np.unique(lines[:,:,1], return_counts=True)[1][0]
        self.res_Grid_lines_vert = np.unique(lines[:,:,1], return_counts=True)[1][1]

    def size(self, showplot=False, savefig=False):
        img = self.imagedata[2]

        # Initial location of centerpoint
        thld = int(utilfunc.getThreshold.otsuMethod(img))
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

        if showplot or savefig:
            plt.imshow(img, cmap='gray')
            for len, coord in [measureResults1, measureResults2]:
                plt.scatter(coord[:, 1], coord[:, 0], c='red', marker='x')
                plt.plot(coord[:, 1], coord[:, 0], label=f"Length = {np.round(len,1)}mm")
            plt.scatter(centerpoint[1], centerpoint[0], c='blue', marker='x')
            plt.legend()
            plt.xlabel("Left to right")
            plt.ylabel("Posterior to anterior")
            if showplot:
                plt.show()
            if savefig:
                plt.savefig(self.dirs["png"]+"francis_size.png")
                plt.close()
        
        self.res_GA = np.round(np.mean([measureResults1[0], measureResults2[0]]),1)
        self.res_GA_SD = np.round(np.std([measureResults1[0], measureResults2[0]]),1)
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
            plt.xlabel("Left to right")
            plt.ylabel("Posterior to anterior")
            if showplot:
                plt.show()
            if print:
                plt.savefig(self.dirs["png"]+"francis_ghosting.png")
                plt.close()

        self.res_Ghosting = np.round(result, 1)
        return


    @property
    def data_organized(self):
        # Organize data. All tests have to have ran otherwise you get an error later on.
        if self._data_organized is None:
            self._data_organized = {
                "Resolution": {
                    "result": self.res_RES,
                    "deviation": self.res_RES_SD,
                    "criteria": {"min": np.round(self.spacing[0]*0.6,1), "max": np.round(self.spacing[0]*1.4,1)},
                    "unit": "mm",
                    "image": self.dirs["png"]+"francis_res.png",
                    "display_range": [0.3,2],
                    "description": "The module consists of 12 triangles arranged around a center point. The triangles work as a continuous measure of resolution by measuring how close thetwo triangle sides can be to each other before there is no sufficient contrast anymore to distinguish triangle from background."
                },
                "Diameter": {
                    "result": self.res_GA,
                    "deviation": self.res_GA_SD,
                    "criteria": {"min": 144,"max": 152},
                    "unit": "mm",
                    "image": self.dirs["png"]+"francis_size.png",
                    "display_range": [140,156],
                    "description": "The phantoms inner diameter is measured to determine correct spatial resolution."
                },
                "Low Contrast": {
                    "result": self.res_LCOD,
                    "criteria": {"min": 6, "max": 8},
                    "unit": "spokes",
                    "image": self.dirs["png"]+"francis_contrast.png",
                    "display_range": [0  ,9  ],
                    "description": "The module consists of a disk where 12 spokes go from the centerpoint outwards. Each spoke consists of four circular cutouts with increasing size form the center. he number of visible spokes gives an indication of the scanner's capability to resolve contrasts. "
                },
                "Image Uniformity": {
                    "result": self.res_IIU,
                    "criteria": {"min": 80, "max": 100},
                    "unit": "%",
                    "image": self.dirs["png"]+"francis_uniformity.png",
                    "display_range": [50 ,110],
                    "description": "The image is smoothed by convolving it with a rectangle kernel. Next the highest and lowest point of the convolved image is determined and the values are used for the calculation of the intensity uniformity via the equation from the ACR manual."
                },
                "Slice Thickness": {
                    "result": self.res_STA,
                    "criteria": {"min": 4, "max": 6},
                    "unit": "mm",
                    "image": self.dirs["png"]+"francis_thickness.png",
                    "display_range": [0  ,10 ],
                    "description": "It is analog to the ACR tests in which the slice position is measured by having two opposite 45 degree ramps meeting in the middle of their incling, resulting in two rectangles in an image which, if equally long, indicates to correct slice positioning."
                },
                "Slice Position": {
                    "result": self.res_SPA,
                    "criteria": {"min":-3,"max": 3},
                    "unit": "mm",
                    "image": self.dirs["png"]+"francis_position.png",
                    "display_range": [-5 ,5  ],
                    "description": "It is analog to the ACR tests in which the slice position is measured by having two opposite tunnels meeting in the middle of their incling, resulting in two rectangles in an image whose length indicate how thick the slice is."
                },
                "Grid Size": {
                    "result": self.res_Grid_size,
                    "criteria": {"min":30, "max": 44},
                    "unit": "mm2",
                    "image": self.dirs["png"]+"francis_grid.png",
                    "display_range": [28 ,46 ],
                    "description": "This module checks for warping of straight lines as well as the correct size representation in the transversal plane. The slice shows a grid of squares which are separated by grid lines from one another. The inner squares have a length of 6mm and the grid lines are 1mm thick.",
                },
                "Grid Lines horizontal": {
                    "result": self.res_Grid_lines_hori,
                    "criteria": {"min":7, "max": 13},
                    "unit": "Lines",
                    "display_range": [0,15],
                    "description": "",
                },
                "Grid Lines vertical": {
                    "result": self.res_Grid_lines_vert,
                    "criteria": {"min":7, "max": 13},
                    "unit": "Lines",
                    "display_range": [0,15],
                    "description": "",
                },
                "Ghosting": {
                    "result": self.res_Ghosting,
                    "criteria": {"min": 0, "max": 5},
                    "unit": "%",
                    "image": self.dirs["png"]+"francis_ghosting.png",
                    "display_range": [-1  ,10],
                    "description": " The same area as in the Image Intensity Uniformity test is masked out in addition to four ellipses top, bottom, left and right of the phantom. The PSG is calculated according to the ACR manual."
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

        csv_filename = self.dirs["csv"] + f'{self.scannername}_francis.csv'
        write_header = not os.path.isfile(csv_filename)
        
        # Write CSV
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            if write_header:
                header = savedata.keys()
                csv_writer.writerow(header)
            writedata = [savedata[key] for key in savedata.keys()]
            csv_writer.writerow(writedata)

        # Remove duplicates
        df = pd.read_csv(csv_filename, sep=",")
        df.drop_duplicates(subset=df.columns.difference(['Time of evaluation']), inplace=True) # exclude TOE in exclusion
        df.sort_values(["Date of measurement"])
        df.to_csv(csv_filename, index=False)
        pass

    def _check_criteria(self, test_name, value):
        criteria = self.data_organized.get(test_name).get("criteria")
        if not criteria:
            return False

        if "min" in criteria and value < criteria["min"]:
            return False
        if "max" in criteria and value > criteria["max"]:
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
        pdf.cell(200, 10, "Francis Analyzer Report", ln=True, align='C')

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
            if value.get("image") is None:
                continue
            image_filename = value["image"]
            metric_name = key
            metric_value = value["result"]
            metric_unit = value["unit"]
            metric_desc = value["description"]

            metric_deviation = f'+-{value["deviation"]}' if value.get("deviation") else ""

            pdf.add_page()
            pdf.set_font("Arial", size=16)
            pdf.cell(200, 10, key, ln=True, align='L')
            pdf.image(image_filename, x=80, y=20, w=120)
            # pdf.set_xy(20, 30)
            pdf.ln(20)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"{metric_name}: {metric_value}{metric_deviation} {metric_unit}", ln=True)
            pdf.ln(80)
            pdf.set_font("Arial", size=14)
            pdf.cell(100,10, "Description")
            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            pdf.multi_cell(0,5,metric_desc, border=True)

        # Save the PDF
        pdf.output(self.dirs["srp"] + f"{self.scannername}_{self.creationdate}_QAreport.pdf")

    def _readcsv(self):
        csv_filename = self.dirs["csv"] + f'{self.scannername}_francis.csv'
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
            plt.axhline(value["criteria"]["min"], color = "g")
            plt.axhline(value["criteria"]["max"], color = "g")
            plt.plot(dates, ydata, marker='o')
            plt.ylim(value["display_range"])
            # plt.ylim([0,10])

            # Adding titles and labels
            plt.title(f'Longitudinal plot for {testname}')
            plt.ylabel(f'Result [{value["unit"]}]')
            plt.xlabel("Date of test")

            # Show the plot
            # plt.show()
            # Save figure
            plt.savefig(self.dirs["png"]+f"Longterm_{testname}_francis.png")
            plt.close()

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.set_font("Arial", size=11)
        x_offset = 10
        y_offset = 0
        width = 90
        height = 70

        for i, testname in enumerate(self.data_organized.keys()):
            if i % 3 == 0:
                pdf.add_page()
                y_offset = 30
            pdf.image(self.dirs["png"]+f"Longterm_{testname}_francis.png", x=x_offset+100, y=y_offset, w=width, h=height)
            pdf.set_xy(x_offset, y_offset+10)
            pdf.multi_cell(95,5,f"{self.data_organized[testname]['description']}")
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

        # if os.path.exists(self.dirs["png"]):
        #     shutil.rmtree(self.dirs["png"])
            
