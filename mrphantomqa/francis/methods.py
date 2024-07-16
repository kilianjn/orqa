import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#Debugging
import mrphantomqa.utils.viewer as viw
from ..utils.methods import functions as utilFunc

class functions:
    def __init__(self):
        pass

    class ga:
        """Geometric Accuracy"""
        def measureDistance(imagedata, startpoint, angle_in_deg, spacing=[1,1], showplot=False):
            """Finds length of straight line at an angle. Image has to be thresholded and Length is measured
            as length between the first and last 1 along the line in the picture."""
            assert len(np.unique(imagedata)) == 2 or len(np.unique(imagedata)) == 3,"Image values must be binary"
            spacing_y, spacing_x = spacing
            y_startpoint, x_startpoint = startpoint

            line = utilFunc.draw_line(imagedata, startpoint, angle_in_deg)
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
        
    class res:
        """Geometric Accuracy"""
        def measureDistance(imagedata, startpoint, angle_in_deg, spacing=[1,1], showplot=False):
            """Finds length of straight line at an angle. Image has to be thresholded and Length is measured
            as length between the first and last 1 along the line in the picture."""
            assert len(np.unique(imagedata)) == 2 or len(np.unique(imagedata)) == 3,"Image values must be binary" #3 for the case of masked arrays
            spacing_y, spacing_x = spacing
            y_startpoint, x_startpoint = startpoint

            line, _ = utilFunc.draw_line1(imagedata, startpoint, angle_in_deg)
            line_image_overlap = np.ma.masked_array(imagedata,line)
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
                return length, outer_points
            
            else:
                # print("No non-zero elements found in line_image_overlap.")
                return 0,0


       
    class lcod:
        """Low-Contrast-Object-Detectability"""

        def calcLCOD(imagedata, showplot=False):
            count = []
            images = []
            spokes = []

            # Interpolation
            interpolatedArray = np.zeros((11,imagedata.shape[1]*2,imagedata.shape[2]*2))
            interpolatedArray = utilFunc.interpolateImage(imagedata[slice],2)
            imagedata = interpolatedArray

            ## Get general data
            thld = utilFunc.getThreshold.findPeak(imagedata,10)
            thldimg = utilFunc.createThresholdImage(imagedata, thld*0.8)
            center = utilFunc.findCenter.centerOfMass(thldimg)

            ## Cutout center and get appropiate masks
            mask = utilFunc.cutoutStructureMask(thldimg, center)
            thldMaskedImg = np.ma.masked_array(thldimg, mask)

            ## Data for mask
            center_new = utilFunc.findCenter.centerOfMass(~mask)
            diameter,_ = functions.ga.measureDistance(~mask,center_new,0)
            radius = diameter/2

            circMask = utilFunc.circularROI(imagedata, center_new,radius)

            # Actual Computation
            cutoutImage = np.ma.masked_array(imagedata, circMask)
            lineArraysByAngle, _ = functions.lcod.radialTrafo_LCOD(cutoutImage, center_new, -105)

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

            print(sum(count))

            return tuple(zip(count, images, spokes))

        def radialTrafo_LCOD(imagedata, centerpoint, startangle, showplot=False):
            lineArraysByAngle = []
            masks = []
            for i in range(startangle, startangle + 360):
                line, valuesacrosLine = utilFunc.draw_line1(imagedata, centerpoint, i)
                lineArraysByAngle.append(valuesacrosLine)
                masks.append(line)
            
            cutoff = min([np.ma.count(lineArraysByAngle[i]) for i in range(len(lineArraysByAngle))])-2
            lineArraysByAngle = np.array([lineArraysByAngle[i][:cutoff] for i in range(360)])
            # lineArraysByAngle = np.transpose(lineArraysByAngle)


            if showplot:
                plt.imshow(lineArraysByAngle)
                plt.show()

            return lineArraysByAngle, masks

        def countEdges(edgeImage):
            allAngles, length = edgeImage.shape
            countThld = np.mean(edgeImage) + 2.5 * np.std(edgeImage)

            peakcount = np.zeros((allAngles,8)) # per angle 1 positive and 1 negative threshold
            countedSpokes = 0
            foundSpokes = []

            # allAngles = range(allAngles)
            relevantAngles = []
            relevantAngles.append([9,21])
            relevantAngles.append([56,68])
            relevantAngles.append([101,111])
            relevantAngles.append([145,156])
            relevantAngles.append([189,203])
            relevantAngles.append([235,246])
            relevantAngles.append([280,291])
            relevantAngles.append([330,341])

            distances = []
            for j in range(5): # Divide in 3 areas for peaks
                distances.append(int((0.08 + 0.2*j)*length))

            # Peakdetection
            for i in range(allAngles): #iterate through all angles
                for j in range(4): #iterate through all four peak areas as defined in var distances
                    peakcount[i,2*j] = 1 if any(edgeImage[i,distances[j]:distances[j+1]] > countThld) else 0
                    peakcount[i,2*j+1] = 1 if any(edgeImage[i,distances[j]:distances[j+1]] < -countThld) else 0

            # Evaluation of peakdetection array over all angles
            for i in range(len(relevantAngles)):
                columnSum = np.sum(peakcount[relevantAngles[i][0]:relevantAngles[i][1]], axis=0)
                #At least one detection in an area is needed to make it count as detected.
                if (columnSum[0:2] > 0).sum() > 0 and (columnSum[2:4] > 0).sum() > 0 and (columnSum[4:6] > 0).sum() > 0 and (columnSum[6:8] > 0).sum():
                    countedSpokes += 1
                    foundSpokes.append(relevantAngles[i][0])
                    foundSpokes.append(relevantAngles[i][1])
            foundSpokes = np.array(foundSpokes)

            return countedSpokes, foundSpokes

    class iiu:
        def searchForCircularSpots(convImgROI, showplot=False):
            maxValue = np.max(convImgROI)
            minValue = np.min(convImgROI)
            maxCoord = np.where(convImgROI == np.max(convImgROI))
            minCoord = np.where(convImgROI == np.min(convImgROI))

            if showplot:
                plt.imshow(convImgROI)
                plt.scatter(maxCoord[1],maxCoord[0],facecolors='none',edgecolors='r', s=10**2, label=f"max: {maxValue}")
                plt.scatter(minCoord[1],minCoord[0],facecolors='none',edgecolors='b', s=10**2, label=f"min: {minValue}")
                plt.legend()
                plt.show()
            return maxValue, minValue, maxCoord, minCoord