import csv
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

from .acr.methods import functions as acrfunc
from .utils.methods import functions as utilfunc

class acrAnalyzer:
    def __init__(self, data) -> None:
        self.imagedata_loc = data.imagedata[0]
        self.metadata = data.metadata if hasattr(data, 'metadata') else None
        self.pixelSpacing = [float(i) for i in self.metadata[0x00280030].value]

        self.res_GA = None
        self.res_HCSR = None
        self.res_STA = None
        self.res_SPA = None
        self.res_IIU = None
        self.res_PGA = None
        self.res_LCOD = None

        self.tableData = None
        pass

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

    def get_results(self):
        table = [["Test", "Pass value", "Measured value", "Pass/Fail"]]
        if self.res_GA is not None:
            table.append(["GA", "190mm +- 5mm", f"{np.round(self.res_GA,1)}mm", "Pass" if np.abs(self.res_GA - 190) < 10 else "Fail"])
        if self.res_SPA is not None:
            table.append(["SPA", "Delta < 5mm", str(np.round(self.res_SPA,2)), "Pass" if np.abs(self.res_SPA) < 5 else "Fail"])
        if self.res_IIU is not None:
            table.append(["IIU", str(80), str(np.round(self.res_IIU,1)), "Pass" if np.abs(self.res_IIU) >80 else "Fail"])
        if self.res_PGA is not None:
            table.append(["PSG", "< 5.0%", f"{np.round(self.res_PGA*100,1)}%", "Pass" if np.abs(self.res_PGA *100) < 5 else "Fail"])
        if self.res_LCOD is not None:
            table.append(["LCOD", "37 Spokes", f"{self.res_LCOD} Spokes", "Pass" if np.abs(self.res_LCOD) > 35 else "Fail"])
        if self.res_STA is not None:
            table.append(["STA", "5.0mm +- 0.7mm", f"{np.round(self.res_STA,2)}mm", "Pass" if np.abs(self.res_STA-5) < 0.7 else "Fail"])

        self.tableData = table

    def add2csv(self):
        if self.tableData is None:
            print("No resultdata generated!")
            return
        
        # Create a CSV file
        csv_filename = f'{self.metadata[0x00080080].value}.csv'
        header = ["DateOfMeasurement", "TimeOfMeasurement", "GA", "SPA", "IIU", "PGA", "LCOD", "STA"]
        writedata = [f"{self.metadata[0x00080022].value}",f"{self.metadata[0x00080032].value}"] + [self.tableData[i][2] for i in range(1,len(self.tableData))]

        with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            csv_writer.writerow(writedata)
        return

    def createReport(self):
        WIDTH = 210
        HEIGHT = 297

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 24)  
        pdf.ln(60)
        pdf.write(5, f"ACR QA")
        pdf.ln(10)
        pdf.set_font('Arial', '', 16)
        pdf.write(4, f'Automated report')
        pdf.ln(30)

        # Header hinzufügen
        pdf.set_font_size(14)
        for item in self.tableData[0]:
            pdf.cell(40, 10, item, 1)
        pdf.ln()
        pdf.set_font_size(10)
        # Daten hinzufügen
        for row in self.tableData[1:]:
            for item in row:
                pdf.cell(40, 10, item, 1)
            pdf.ln()

        pdf.add_page()
        pdf.image("test1.png",80,5,WIDTH-80)
        pdf.set_font('Arial', '', 10)
        pdf.text(5,15, f'Geometric Accuracy')
        
        
        pdf.image("test2.png",80,100,WIDTH-80)
        pdf.text(5,110, f'Slice Position Accuracy')

        pdf.image("test3.png",80,195,WIDTH-80)
        pdf.text(5,205, f'Image Intensity Uniformity')

        pdf.add_page()

        pdf.image("test4.png",80,5,WIDTH-80)
        pdf.set_font('Arial', '', 10)
        pdf.text(5,15, f'Percent Signal Ghosting')
        
        
        pdf.image("test5.png",80,100,WIDTH-80)
        pdf.text(5,110, f'Low Contrast Object Detectability')

        pdf.image("test6.png",80,195,WIDTH-80)
        pdf.text(5,205, f'Slice Thickness Accuracy')


        pdf.output("../test.pdf", 'F')
        print("done")