import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.morphology import flood_fill
from scipy import interpolate
import tkinter as tk
from tkinter import ttk

#Debugging
import mrphantomqa.utils.viewer as viw

class functions:
    def __init__(self):
        self.value = None

    class findCenter:
        def centerOfMass(thldimage, showplot = False):
            assert len(np.unique(thldimage)) == 2,"Image values must be binary"
            com_x = 0
            com_y = 0
            y_ax, x_ax = thldimage.shape
            totalsignal = np.sum(thldimage)

            for x in range(x_ax):
                com_x += (x * np.ma.sum(thldimage[:,x]))/totalsignal
            for y in range(y_ax):
                com_y += (y * np.sum(thldimage[y,:]))/totalsignal
            # print(f"{com_x} und {com_y}")
            if showplot:
                plt.imshow(thldimage)
                plt.scatter(com_x,com_y, marker="o")
                plt.show()
            return np.round(com_y).astype(int), np.round(com_x).astype(int)
        
        def centerOfMassFilled(thldimage, showplot=False):
            assert len(np.unique(thldimage)) == 2,"Image values must be binary"
            com_x = 0
            com_y = 0
            y_ax, x_ax = thldimage.shape
            tempimage = np.zeros(thldimage.shape)

            for x in range(thldimage.shape[1]):
                indicesOfOne = np.where(thldimage[:,x] == 1)[0]
                if len(indicesOfOne) == 0:
                    continue
                minIndex = min(indicesOfOne)
                maxIndex = max(indicesOfOne)
                tempimage[minIndex:maxIndex,x] = 1

            for y in range(thldimage.shape[0]):
                indicesOfOne = np.where(thldimage[y,:] == 1)[0]
                if len(indicesOfOne) == 0:
                    continue
                minIndex = min(indicesOfOne)
                maxIndex = max(indicesOfOne)
                tempimage[y,minIndex:maxIndex] = 1

            totalsignal = np.sum(tempimage)

            for x in range(x_ax):
                com_x += (x * np.sum(tempimage[:,x]))/totalsignal
            for y in range(y_ax):
                com_y += (y * np.sum(tempimage[y,:]))/totalsignal

            if showplot:
                plt.imshow(tempimage)
                plt.scatter(com_x,com_y, marker="o")
                plt.show()
            return (round(com_y), round(com_x))
 
    class getThreshold:
        def findPeak(imagedata, skip_perc=10, showplot=False):
            x, bins = functions.getHistogram(imagedata)
            startpoint = int(len(x) * (skip_perc/100))

            peak_value_count = np.max(x[startpoint:])
            peak_value = np.mean(np.where(x[startpoint:] == peak_value_count)) + startpoint
            if showplot:
                plt.plot(bins,x)
                plt.ylim(0,np.max(x[startpoint:]*1.1))
                plt.axvline(peak_value, color="red", linestyle="dashed")
                plt.axvline(startpoint, color="green", linestyle="dashed")
                plt.show()
 
            return peak_value
        
        def findMinAlongRows(imagedata, showplot=False):
            """Note: img is cut to only show a small cutout from the middle to avoid zeroes from left/right and noise from top/bottom"""
            mins = []
            height, width = imagedata.shape
            cutout = imagedata[2:height-2, int(width/2)-5: int(width/2)+5]
            for i in range(cutout.shape[1]):
                mins.append(np.min(cutout[:,i]))
            thld_value = np.max(mins).astype(int) + 1

            return thld_value
        
        def otsuMethod(imagedata, skip_perc=0, showplot=False):
            x, bins = functions.getHistogram(imagedata)
            startpoint = int(len(x) * (skip_perc/100))
            x = x[startpoint:]
            bins = bins[startpoint:]
            totalsignal = np.sum(x)

            w_0 = np.empty(x.shape)
            w_1 = np.empty(x.shape)
            mu_0 = np.empty(x.shape)
            mu_1 = np.empty(x.shape)

            for i in range(len(x)):
                w_0[i] = np.sum(x[:i+1]) / totalsignal
                w_1[i] = 1 - w_0[i]

                mu_0[i] = np.sum(x[:i+1] / totalsignal * bins[:i+1]) / w_0[i]
                mu_1[i] = np.sum(x[i+1:] / totalsignal * bins[i+1:]) / w_1[i] if i < len(x)-1 else mu_1[-2]

            sig_b = np.round(w_0 * w_1 * (mu_0 - mu_1)**2).astype(int)
            threshold = np.mean(np.where(sig_b == np.max(sig_b))).astype(int) + startpoint
            return threshold

    class ga:
        """Geometric Accuracy"""
        def measureDistance(imagedata, startpoint, angle_in_deg, spacing=[1,1], showplot=False):
            """Finds length of straight line at an angle. Image has to be thresholded and Length is measured
            as length between the first and last 1 along the line in the picture."""
            assert len(np.unique(imagedata)) == 2,"Image values must be binary"
            spacing_y, spacing_x = spacing
            y_startpoint, x_startpoint = startpoint

            line = functions.draw_line(imagedata, startpoint, angle_in_deg)
            line_image_overlap = imagedata & line
            nonzero_coords = np.argwhere(line_image_overlap)

            if len(nonzero_coords) > 0:
                # Find the two most outer points
                outer_points = np.array([nonzero_coords[0], nonzero_coords[-1]])

                length = np.round(np.sqrt(((outer_points[0,1] - outer_points[1,1])*spacing_x)**2 + ((outer_points[0,0] - outer_points[1,0])*spacing_y)**2),3)

                if showplot:
                    # Plot imagedata
                    plt.imshow(imagedata, cmap='gray')

                    # Overlay line_image_overlap with transparency (alpha=0.5)
                    # plt.imshow(line_image_overlap, origin='lower', cmap='jet', alpha=0.5)

                    # Plot the two most outer points
                    plt.scatter(outer_points[:, 1], outer_points[:, 0], c='red', marker='x')
                    plt.plot(outer_points[:, 1], outer_points[:, 0], label=f"Length = {length}")
                    plt.scatter(x_startpoint, y_startpoint, c='blue', marker='x')
                    plt.legend()
                    plt.show()
            else:
                print("No non-zero elements found in line_image_overlap.")

            return length, outer_points

    class hcsr:
        """High-Contrast-Spatial-Resolution"""
        def cutoutRelevantSquare(imagedata, showplot=False):
            peakValue = functions.getThreshold.findPeak(imagedata,10)
            thldimg = functions.createThresholdImage(imagedata, peakValue/2)
            com_x, com_y = functions.findCenter.centerOfMassFilled(thldimg)
            coord = functions.getAreaBoundaries(thldimg, (np.round(com_y + 0.3 * com_y).astype(int) ,com_x))
            width = coord[3] - coord[2]
            height = coord[1] - coord[0]
            coord = (coord[0] + int(0.2 * height), coord[1] - int(0.2 * height), coord[2] + int(0.25*width), coord[3]-int(0.1*width))
            test = functions.createSubarray(imagedata, coord).astype(int)

            if showplot:
                # plt.imshow(test)
                # plt.show()
                functions.questionWindow(test, "All resolved?")
            
            return test

    class sta:
        """Slice-Thickness-Accuracy"""
        def getrows(imagedata):
            row_sum = np.sum(imagedata,axis=1)
            rowthld = np.median(row_sum)*2
            sel_rows = np.where(row_sum < rowthld)[0]
            minrow, maxrow = (np.min(sel_rows), np.max(sel_rows))
            return minrow,maxrow+1

        def measureLength(imagedata, spacing=1):
            thld = functions.getThreshold.otsuMethod(imagedata) 

            rowsum = np.sum(functions.createThresholdImage(imagedata, thld),axis=1)
            rowsum_conv = np.abs(np.convolve(rowsum,[1,-1],"valid"))
            a = rowsum_conv.shape[0]
            b = int(a*0.25)
            border = np.where(rowsum_conv == np.max(rowsum_conv[b:-b]))[0][0] +1  # Edgecase fixen mit 2 minima/was wenn beide gleich lang. Idee: bei fehlschlag einfach mitte nehmen.

            upperhalf = imagedata[:border]
            lowerhalf = imagedata[border+1:]

            upperthldimg = functions.createThresholdImage(upperhalf,thld)
            lowerthldimg = functions.createThresholdImage(lowerhalf,thld)

            upperBorder = np.where(np.sum(upperthldimg, axis=0) > (max(np.sum(upperthldimg, axis=0)))/2)
            lowerBorder = np.where(np.sum(lowerthldimg, axis=0) > (max(np.sum(lowerthldimg, axis=0)))/2)

            upperLength = np.max(upperBorder) - np.min(upperBorder)
            lowerLength = np.max(lowerBorder) - np.min(lowerBorder)
            meanLength = 0.2 * (upperLength * lowerLength) / (upperLength + lowerLength) * spacing



            return meanLength, (np.max(upperBorder),np.min(upperBorder),np.max(lowerBorder),np.min(lowerBorder)), border

    class spa:
        """slice-Position-Accuracy"""
        def getPositionDifference(thldimage, centerCoord, showplot=False):
            
            diameter, _ = functions.ga.measureDistance(thldimage,centerCoord,0)
            x1_offset = centerCoord[1] - int(diameter*0.02)
            x2_offset = centerCoord[1] + int(diameter*0.02)
            # y_offset = centerCoord[0] - int(diameter*0.25)

            imgROI = thldimage[centerCoord[0] - int(diameter*0.4):centerCoord[0] - int(diameter*0.25),:]
            line1 = np.sum(imgROI[:,x1_offset])
            line2 = np.sum(imgROI[:,x2_offset])
            lengthDifference = line1 - line2

            print(lengthDifference)
            if showplot:
                plt.axvline(centerCoord[1])
                plt.axvline(x1_offset, linestyle="dotted")
                plt.axvline(x2_offset, linestyle="dotted")
                plt.imshow(imgROI)
                plt.show()

            return lengthDifference, imgROI, (x1_offset,x2_offset)
        
    class iiu:
        """Image-Intensity-Uniformity"""
        
        def searchForCircularSpots(imagedata, center, roiDiameter, kernelsize, showplot=False):
            convolvedImg = cv.filter2D(imagedata, -1, np.ones((kernelsize,kernelsize))/(kernelsize**2))
            roiMask = functions.circularROI(imagedata, center, roiDiameter)
            convImgROI = np.ma.masked_array(convolvedImg, roiMask)
            maxValue = np.max(convImgROI)
            minValue = np.min(convImgROI)
            maxCoord = np.where(convImgROI == np.max(convImgROI))
            minCoord = np.where(convImgROI == np.min(convImgROI))

            if showplot:
                plt.imshow(convImgROI)
                plt.scatter(maxCoord[1],maxCoord[0],facecolors='none',edgecolors='r', s=kernelsize**2, label=f"max: {maxValue}")
                plt.scatter(minCoord[1],minCoord[0],facecolors='none',edgecolors='b', s=kernelsize**2, label=f"min: {minValue}")
                plt.legend()
                plt.show()
            return maxValue, minValue, maxCoord, minCoord, convImgROI

    class psg:
        """Percent-Signal-Ghosting"""
        def calcPSG(imagedata, thldimg, center, showplot=False):
            radius,_ = functions.ga.measureDistance(thldimg,center,0)
            radius /= 2
            centermask = functions.circularROI(imagedata, center, 0.8*radius)

            spaceTop = imagedata.shape[0] - center[0] - radius
            spaceBot = center[0] - radius
            spaceRight = imagedata.shape[1] - center[1] - radius
            spaceLeft = center[1] - radius
            ellipseRadius = np.min([spaceBot,spaceLeft,spaceRight,spaceBot])

            topmask = functions.circularROI(imagedata, center, ellipseRadius/2,False,0,radius + 1.3 * spaceTop/2,3.2,0.7)
            botmask = functions.circularROI(imagedata, center, ellipseRadius/2,False,0,-(radius + 1.3 * spaceBot/2),3.2,0.7)
            rightmask = functions.circularROI(imagedata, center, ellipseRadius/2,False,radius + 1.3 * spaceRight/2,0,0.7,3.2)
            leftmask = functions.circularROI(imagedata, center, ellipseRadius/2,False,-(radius + 1.3 * spaceLeft/2),0,0.7,3.2)

            meantop = np.round(np.mean(np.ma.masked_array(imagedata,topmask)),2)
            meanbot = np.round(np.mean(np.ma.masked_array(imagedata,botmask)),2)
            meanleft = np.round(np.mean(np.ma.masked_array(imagedata,leftmask)),2)
            meanright = np.round(np.mean(np.ma.masked_array(imagedata,rightmask)),2)
            meancenter = np.round(np.mean(np.ma.masked_array(imagedata,centermask)),2)

            ghostingPercentRatio = 100 * np.abs(((meantop+meanbot)-(meanleft+meanright))/(2*meancenter))

            test= np.ma.mask_or(~topmask,~botmask)
            test1= np.ma.mask_or(~leftmask,~rightmask)
            test= np.ma.mask_or(test,test1)

            if showplot:
                plt.imshow(np.ma.masked_array(imagedata), cmap="bone")
                plt.imshow(np.ma.masked_array(imagedata,~test))
                plt.show()
            # print(ghostingPercentRatio)

            return ghostingPercentRatio, [centermask,~test], (meancenter, meantop, meanright, meanbot, meanleft)
        
    class lcod:
        """Low-Contrast-Object-Detectability"""

        def calcLCOD(imagedata, method="peaks", showplot=False):
            count = []
            images = []
            spokes = []

            # Interpolation
            interpolatedArray = np.zeros((11,imagedata.shape[1]*2,imagedata.shape[2]*2))
            for slice in [7,8,9,10]:
                interpolatedArray[slice] = functions.interpolateImage(imagedata[slice],2)
            imagedata = interpolatedArray

            for slice in [7,8,9,10]:
                ## Get general data
                thld = functions.getThreshold.findPeak(imagedata[slice],10)
                thldimg = functions.createThresholdImage(imagedata[slice], thld*0.8)
                center = functions.findCenter.centerOfMass(thldimg)

                ## Cutout center and get appropiate masks
                mask = functions.cutoutStructureMask(thldimg, center)
                thldMaskedImg = np.ma.masked_array(thldimg, mask)

                ## Data for mask
                center_new = functions.findCenter.centerOfMassFilled(thldMaskedImg)
                diameter,_ = functions.ga.measureDistance(thldMaskedImg,center_new,0)
                radius = diameter/2

                circMask = functions.circularROI(imagedata[slice], center_new,radius)

                # Actual Computation
                ## Normalize Image

                if method == "peaks":

                    # Generate normalized image
                    kernel = int(7)
                    blurredImage = cv.filter2D(imagedata[slice], -1, np.ones((kernel,kernel)) / (kernel ** 2))
                    normalizedImage = np.abs(imagedata[slice]) / (np.abs(blurredImage) + 1)
                    normalizedCutoutImage = np.ma.masked_array(normalizedImage, circMask)

                    lineArraysByAngle, _ = functions.lcod.radialTrafo_LCOD(normalizedCutoutImage, center_new, slice)
                    images.append(lineArraysByAngle)

                    countedSpokes, spokePosition = functions.lcod.countPeaks(lineArraysByAngle, showplot)
                    count.append(countedSpokes)
                    spokes.append(spokePosition)

                elif method == "edges":
                    cutoutImage = np.ma.masked_array(imagedata[slice], circMask)
                    lineArraysByAngle, _ = functions.lcod.radialTrafo_LCOD(cutoutImage, center_new, slice)
                    
                    # Generate edgeimage
                    edgeImage = []
                    for angle in range(lineArraysByAngle.shape[0]):
                        edgeImage.append(np.convolve(lineArraysByAngle[angle], [1,0,-1], "valid"))
                    edgeImage = np.array(edgeImage)
                    lineArraysByAngle = edgeImage[:,:int(edgeImage.shape[1]*0.95)]

                    images.append(lineArraysByAngle)

                    countedSpokes, spokePosition = functions.lcod.countEdges(lineArraysByAngle, showplot)
                    count.append(countedSpokes)
                    spokes.append(spokePosition)

                elif method == "edges1":
                    cutoutImage = np.ma.masked_array(imagedata[slice], circMask)
                    lineArraysByAngle, _ = functions.lcod.radialTrafo_LCOD(cutoutImage, center_new, slice)
                    
                    # Generate edgeimage
                    edgeImage = []
                    for angle in range(lineArraysByAngle.shape[0]):
                        edgeImage.append(np.convolve(lineArraysByAngle[angle], [1,0,-1], "valid"))
                    edgeImage = np.array(edgeImage)
                    lineArraysByAngle = edgeImage[:,:int(edgeImage.shape[1]*0.95)]

                    images.append(lineArraysByAngle)

                    countedSpokes, spokePosition = functions.lcod.countEdges1(lineArraysByAngle, showplot)
                    count.append(countedSpokes)
                    spokes.append(spokePosition)
                    return

            print(sum(count))
            
            return tuple(zip(count, images, spokes))

        def radialTrafo_LCOD(imagedata, centerpoint, slice=0, showplot=False):
            lineArraysByAngle = []
            masks = []
            for i in range(-110, 250):
                line, valuesacrosLine = functions.draw_line1(imagedata, centerpoint, i)
                lineArraysByAngle.append(valuesacrosLine)
                masks.append(line)
            
            cutoff = min([np.ma.count(lineArraysByAngle[i]) for i in range(len(lineArraysByAngle))])-2
            lineArraysByAngle = np.array([lineArraysByAngle[i][:cutoff] for i in range(360)])
            # lineArraysByAngle = np.transpose(lineArraysByAngle)

            match slice:
                case 8:
                    lineArraysByAngle = np.roll(lineArraysByAngle, -9, 0)
                case 9:
                    lineArraysByAngle = np.roll(lineArraysByAngle, -18, 0)
                case 10:
                    lineArraysByAngle = np.roll(lineArraysByAngle, -27, 0)


            if showplot:
                plt.imshow(lineArraysByAngle)
                plt.show()

            return lineArraysByAngle, masks

        def countPeaks(imagedataNormed, showplot=False):
            countedSpokes = 0
            allAngles, length = imagedataNormed.shape
            countThld = np.mean(imagedataNormed[:,:int(length)]) + 2.5 * np.std(imagedataNormed[:,:int(length)])

            angles = [0]
            for i in range(10): # Divide in 10 angles
                angles.append(45 + i*34 if i != 9 else 359)
            distances = []
            for j in range(4): # Divide in 3 areas for peaks
                distances.append(int((0.14 + 0.31*j)*length) if j != 3 else length-1)

            allFoundPeaks = []
            for i in range(allAngles):
                singlepeak = []
                for j in range(3):
                    if any(imagedataNormed[i,distances[j]:distances[j+1]] > countThld):
                        singlepeak.append(1)
                    else:
                        singlepeak.append(0)
                allFoundPeaks.append(singlepeak)

            allFoundPeaks = np.array(allFoundPeaks)

            # Windowing
            foundSpokes = []
            for i in range(allAngles):
                windowarray = allFoundPeaks[i-1] + allFoundPeaks[i] + allFoundPeaks[(i+1)%359]
                if all(windowarray > 0):
                    foundSpokes.append(i)

            foundSpokes = np.array(foundSpokes)
            for angleArea in range(10):
                if any(foundSpokes < angles[angleArea+1]) and any(angles[angleArea] <= foundSpokes):
                    countedSpokes += 1

            print(countedSpokes)

            if showplot:
                plt.subplot(121)
                plt.imshow(imagedataNormed)
                plt.hlines(angles, 0,length, colors="yellow",linestyles="--", label="Borders")
                plt.hlines(foundSpokes, 0, length, colors="red", alpha=0.3, label="Detections")
                plt.subplot(122)
                plt.imshow(imagedataNormed, cmap="jet")
                plt.show()

            return countedSpokes, foundSpokes

        def countEdges(edgeImage, showplot=False):
            countedSpokes = 0
            allAngles, length = edgeImage.shape
            allAngles = range(allAngles)
            allAngles = list(range(14,31)) + \
                        list(range(50,66)) + \
                        list(range(88,101)) + \
                        list(range(124,136)) + \
                        list(range(159,168)) + \
                        list(range(194,205)) + \
                        list(range(230,238)) + \
                        list(range(268,275)) + \
                        list(range(304,311)) + \
                        list(range(342,348))
            # 17, 16, 13, 12,  9, 11, 8, 7, 7, 6
            # 17, 15, 13, 11, 11, 11, 9, 9, 9, 7
            countThld = np.mean(edgeImage) + 1.8 * np.std(edgeImage)

            # for i in range(360):
            #     plt.subplot(121)
            #     plt.plot(edgeImage[i])
            #     plt.ylim(bottom=-40,top=40)
            #     plt.hlines(countThld,0,40)
            #     plt.hlines(-countThld,0,40)
            #     plt.subplot(122)
            #     plt.imshow(edgeImage)
            #     plt.hlines(i,0,40)
            #     plt.show()

            # 10 Regions
            angles = [0]
            for i in range(10): # Divide in 10 angles
                angles.append(45 + i*34 if i != 9 else 359)
            distances = []
            for j in range(4): # Divide in 3 areas for peaks
                distances.append(int((0.14 + 0.31*j)*length) if j != 3 else length-1)


            allFoundPeaks = []
            for i in allAngles:
                singlepeak = []
                for j in range(3):
                    if any(edgeImage[i,distances[j]:distances[j+1]] > countThld): #positive thld
                        singlepeak.append(1)
                    else:
                        singlepeak.append(0)

                    if any(edgeImage[i,distances[j]:distances[j+1]] < -countThld): # negative thld
                        singlepeak.append(1)
                    else:
                        singlepeak.append(0)
                allFoundPeaks.append(singlepeak)

            allFoundPeaks = np.array(allFoundPeaks)

            # Windowing
            requiredPeaks = 3
            foundSpokes = []
            for i in range(allFoundPeaks.shape[0]):
                windowarray = allFoundPeaks[i-1] + allFoundPeaks[i] + allFoundPeaks[(i+1)%allFoundPeaks.shape[0]-1]
                if all(windowarray[:requiredPeaks] > 0):
                    foundSpokes.append(allAngles[i])

            # Find hits in intervals
            foundSpokes = np.array(foundSpokes)
            for angleArea in range(10):
                if any(angles[angleArea] <= value <=angles[angleArea+1] for value in foundSpokes):
                    # print(f"founf at {angleArea}")
                    countedSpokes += 1

            print(countedSpokes)

            if showplot:
                plt.subplot(121)
                plt.imshow(edgeImage)
                plt.hlines(angles, 0,length, colors="yellow",linestyles="--", label="Borders")
                plt.hlines(foundSpokes, 0, length, colors="red", alpha=0.5, label="Detections")
                plt.subplot(122)
                plt.imshow(edgeImage, cmap="jet")
                plt.show()

            return countedSpokes, foundSpokes

        def countEdges1(edgeimage, showplot=False):
            countedSpokes = 0
            allAngles = list(range(14,31)) + \
                        list(range(50,66)) + \
                        list(range(88,101)) + \
                        list(range(124,136)) + \
                        list(range(159,168)) + \
                        list(range(194,205)) + \
                        list(range(230,238)) + \
                        list(range(268,275)) + \
                        list(range(304,311)) + \
                        list(range(342,348))
            pass


    def getHistogram(imagedata, showplot=False):
        # all_bins = np.linspace(int(np.min(imagedata)), int(np.max(imagedata)+1), int(np.max(imagedata)-np.min(imagedata)+2))
        all_bins = np.arange(np.max(imagedata)+2)
        x,bins = np.histogram(imagedata,bins=all_bins)
        if showplot:
            plt.plot(bins[0:-1],x)
            plt.show()

        return x,bins[0:-1]

    def createThresholdImage(imagedata, threshold, showplot=False):
        thld_image = np.zeros(np.shape(imagedata))
        thld_image[np.where(imagedata > threshold)] = 1
        thld_image = thld_image.astype(int)
        if showplot:
            plt.imshow(thld_image)
            plt.show()
        return thld_image

    def draw_line(imagedata, startpoint, angle):
        angle = np.deg2rad(angle)
        y_startpoint, x_startpoint = startpoint

        height, width = imagedata.shape
        lineArray = np.zeros((height, width), dtype=int)
        max_length = int(2 * np.ceil(np.sqrt(height**2 + width**2)))

        timepoints = np.arange(stop = np.ceil(max_length/2)+1, start=np.floor(-max_length/2))

        coordinates = np.empty((timepoints.shape[0],2))
        coordinates[:,1] = (y_startpoint + (np.sin(angle) * timepoints)).astype(int)
        coordinates[:,0] = (x_startpoint + (np.cos(angle) * timepoints)).astype(int)

        coordinates = coordinates[(coordinates[:,0] >= 0) & (coordinates[:,0] < width) & (coordinates[:,1] >= 0) & (coordinates[:,1] < height)]
        coordinates = np.round(coordinates).astype(int)
        lineArray[coordinates[:,1], coordinates[:,0]] = 1

        return lineArray

    def draw_line1(imagedata, startpoint, angle):
        angle = np.deg2rad(angle)
        y_startpoint, x_startpoint = startpoint

        height, width = imagedata.shape
        lineArray = np.zeros((height, width), dtype=int)
        max_length = int(2 * np.ceil(np.sqrt(height**2 + width**2)))

        timepoints = np.arange(stop = max_length, start=0)

        coordinates = np.empty((timepoints.shape[0],2))
        coordinates[:,1] = (y_startpoint + (np.sin(angle) * timepoints)).astype(int)
        coordinates[:,0] = (x_startpoint + (np.cos(angle) * timepoints)).astype(int)

        coordinates = coordinates[(coordinates[:,0] >= 0) & (coordinates[:,0] < width) & (coordinates[:,1] >= 0) & (coordinates[:,1] < height)]
        coordinates = np.round(coordinates).astype(int)
        lineArray[coordinates[:,1], coordinates[:,0]] = 1
        mask = lineArray == 1
        return ~mask, imagedata[coordinates[:,1], coordinates[:,0]]

    def getAreaBoundaries(imagedata, startpoint,showplot=False):
        """
        get rectangle boundariers of given thld image: min_row, max_row, min_col, max_col 
        """
        assert len(np.unique(imagedata)) == 2,"Image values must be binary"
        coordinates = None

        temp = flood_fill(imagedata, startpoint, 2)
        coordinates = np.where(temp == 2)

        assert np.sum(temp[coordinates[0], coordinates[1]]) == np.size(coordinates),"Sanity Check failed"

        if showplot:
            plt.scatter(startpoint[1], startpoint[0])
            plt.imshow(temp)
            plt.show()

        boundaries = (min(coordinates[0]), max(coordinates[0]), min(coordinates[1]), max(coordinates[1]))

        return boundaries
    
    def cutoutStructureMask(thldimg, startpoint,showplot=False):
        assert len(np.unique(thldimg)) == 2,"Image values must be binary"
        coordinates = None

        temp = flood_fill(thldimg, startpoint, 2)
        mask = temp == 2

        if showplot:
            plt.scatter(startpoint[1], startpoint[0])
            plt.imshow(np.ma.masked_array(thldimg, ~mask))
            plt.show()

        return ~mask
    
    def createSubarray(imagedata, boundaries):
        min_row, max_row, min_col, max_col = boundaries
        subarray = imagedata[min_row:max_row+1, min_col:max_col+1]
        return subarray
    
    def interpolateImage(imagedata, resMultiple:int=2):
        """Interpolator, which doubles the array size in both x and y direction. Expects 2D array"""
        if len(imagedata.shape) != 2:
            print("Dimensions are wrong for interpolation. Insert only 2D arrays.")
            return imagedata

        X_original = np.linspace(0, imagedata.shape[1] - 1, imagedata.shape[1])
        Y_original = np.linspace(0, imagedata.shape[0] - 1, imagedata.shape[0])

        ## Double the points in every direction
        X = np.linspace(0, imagedata.shape[1] - 1, resMultiple * imagedata.shape[1])
        Y = np.linspace(0, imagedata.shape[0] - 1, resMultiple * imagedata.shape[0])

        ## Interpolate here
        spline = interpolate.interp2d(X_original, Y_original, imagedata,'cubic')

        ## Generate new Values
        interpolated_image = spline(X, Y)

        return interpolated_image

    def circularROI(imagedata, cenrterpoint:tuple=(1,1), radius:int=1, showplot=False, shiftCenX=0, shiftCenY=0, morphX=1, morphY=1):
        morphX = (1/morphX)**2
        morphY = (1/morphY)**2
        imagemask = morphX * (np.arange(imagedata.shape[1])[np.newaxis,:] - (cenrterpoint[1]+shiftCenX))**2 + morphY*(np.arange(imagedata.shape[0])[:,np.newaxis] - (cenrterpoint[0]+shiftCenY))**2 <= radius**2
        if showplot:
            plt.imshow(np.ma.masked_array(imagedata, ~imagemask))
            plt.show()
        return ~imagemask

    def radialTrafo(imagedata, centerpoint, showplot=False):
        lineArraysByAngle = []
        masks = []
        for i in range(-110, 250):
            line, valuesacrosLine = functions.draw_line1(imagedata, centerpoint, i)
            lineArraysByAngle.append(valuesacrosLine)
            masks.append(line)
        
        cutoff = min([np.ma.count(lineArraysByAngle[i]) for i in range(len(lineArraysByAngle))])-2
        lineArraysByAngle = np.array([lineArraysByAngle[i][:cutoff] for i in range(360)])
        # lineArraysByAngle = np.transpose(lineArraysByAngle)

        if showplot:
            plt.imshow(lineArraysByAngle)
            plt.show()

        return lineArraysByAngle, masks

    def questionWindow(imagedata, question:str=None, **kwargs):
        def valueInput():
            window.user_input = int(textWidget.get("1.0",'end-1c'))
            window.destroy()

        window = tk.Tk()
        window.user_input = None
        window.title('Plotting in Tkinter') 
        window.geometry("750x500") 

        lhs = ttk.Frame(window,width=500)
        lhs.pack(expand=True,fill="both", side="left")
        lhs.pack_propagate(0)
        rhs = ttk.Frame(window,width=250)
        rhs.pack(expand=True,fill="both", side="right")
        rhs.pack_propagate(0)
   
        fig = Figure(figsize = (5, 5), dpi = 100) 
        plot1 = fig.add_subplot(111) 
        plot1.imshow(imagedata, cmap="jet", **kwargs)
        canvas = FigureCanvasTkAgg(fig, master = lhs)
        canvas.draw() 
        canvas.get_tk_widget().pack() 

        if question:   
            headline = tk.Label(rhs,text=f"{question}")
            headline.pack(fill="both")

            textWidget = tk.Text(rhs, height=3, width=15)
            textWidget.pack()

            plot_button = tk.Button(master = rhs,  
                                command = valueInput, 
                                height = 1,  
                                width = 6, 
                                text = "Confirm") 
            plot_button.pack()

        window.mainloop()
        return window.user_input

