import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

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
        """Resolution"""
        def measureDistance(imagedata, startpoint, angle_in_deg, spacing=[1,1], showplot=False):
            """Finds length of straight line at an angle. Image has to be thresholded and Length is measured
            as length between the first and last 1 along the line in the picture."""
            assert len(np.unique(imagedata)) == 2 or len(np.unique(imagedata.data)) == 2,"Image values must be binary" #2nd for the case of masked arrays
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
            countThld = np.mean(edgeImage) + 2.2 * np.std(edgeImage)

            peakcount = np.zeros((allAngles,8)) # per angle 1 positive and 1 negative threshold
            countedSpokes = 0
            foundSpokes = []

            # allAngles = range(allAngles)
            relevantAngles = []
            relevantAngles.append([  5, 24])
            relevantAngles.append([ 53, 71])
            relevantAngles.append([ 98,114])
            relevantAngles.append([142,159])
            relevantAngles.append([186,206])
            relevantAngles.append([234,249])
            relevantAngles.append([277,294])
            relevantAngles.append([322,339])

            distances = []
            for j in range(4): # Divide in 3 areas for peaks
                distances.append(int((0.08 + 0.23*j)*length))
            distances.append(length-1)

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
        
    class grid:
        def cutoutSquare(imagedata, showplot=False):
            thld = utilFunc.getThreshold.otsuMethod(imagedata)
            thld_img = utilFunc.createThresholdImage(imagedata,thld)
            centerpoint = utilFunc.findCenter.centerOfMassFilled(thld_img)

            diameter = utilFunc.measureDistance(thld_img, centerpoint, 45)

            center_offset = int(diameter * 0.25)
            coord1 = centerpoint[1] - center_offset
            coord2 = centerpoint[1] + center_offset
            coord3 = centerpoint[0] - center_offset
            coord4 = centerpoint[0] + center_offset


            center_cutout = imagedata[coord1:coord2,coord3:coord4]

            if showplot:
                plt.subplot(121)
                plt.imshow(imagedata)
                plt.subplot(122)
                plt.imshow(center_cutout)
                plt.show()

            return center_cutout

        def imagePreProcessing(imagedata, showplot=False):
            ## Thresholding
            thld = utilFunc.getThreshold.otsuMethod(imagedata)
            thld_img = utilFunc.createThresholdImage(imagedata,thld)

            ## Preparing for cv lib
            processedImage = thld_img * 255
            processedImage = processedImage.astype(np.uint8)
            processedImage = ~processedImage

            ## Dilatation and erosion
            kernel = np.ones((3,3),np.uint8)
            processedImage = cv.dilate(processedImage,kernel,iterations = 1)
            processedImage = cv.erode(processedImage,kernel,iterations = 1)
            
            return processedImage
        
        def gridDetect(imagedata):
            """
            Takes in binarized imagedata
            blavblalba
            """
            # Line Detection
            ## Hough Transform
            houghThld = imagedata.shape[1]-1
            lines = cv.HoughLines(imagedata, 1, np.pi / 180, houghThld)

            # Line processing
            if lines is None:
                print("No lines found at all.")
                return [0,0], False, []

            ## Angletolerance for horizontal and vertica lines
            d_theta = np.deg2rad(20)
            ## Extract lines in certain angle region      
            lines_vert_pre = [line for line in lines if line[0][1] < d_theta or line[0][1] > np.pi - d_theta]
            lines_hori_pre = [line for line in lines if np.pi/2-d_theta <= line[0][1] <= np.pi/2+d_theta]

            if lines_hori_pre == [] or lines_vert_pre == []:
                print("No gridlines found.")
                return [0,0], False, lines
            
            ## Discrimination of angle minorities

            ### Horizontal lines
            thetas = [line[0][1] for line in lines_hori_pre]
            unique_thetas, counts = np.unique(thetas, return_counts=True)
            most_common_theta = unique_thetas[np.argmax(counts)]
            lines_hori = np.sort(np.array([line for line in lines_hori_pre if line[0][1] == most_common_theta]), axis=0)


            ### Vertical lines
            thetas = [line[0][1] for line in lines_vert_pre]
            unique_thetas, counts = np.unique(thetas, return_counts=True)
            most_common_theta = unique_thetas[np.argmax(counts)]
            lines_vert = np.sort(np.array([line for line in lines_vert_pre if line[0][1] == most_common_theta]), axis=0)



            ## Combination of all lines to one array for output.
            lines_main = np.append(lines_hori, lines_vert, axis=0)

            ## Calculation of median square size, check for perpendicularity
            distance_hori = np.median([lines_hori[i+1][0][0] - lines_hori[i][0][0] for i in range(len(lines_hori)-1)]) - 1
            distance_vert = np.median([lines_vert[i+1][0][0] - lines_vert[i][0][0] for i in range(len(lines_vert)-1)]) - 1

            squaresize_in_pixels = [np.abs(distance_vert), np.abs(distance_hori)]

            angle_grid = np.abs(lines_vert[0][0][1] - lines_hori[0][0][1])

            return squaresize_in_pixels, angle_grid, lines_main
        
        def printImage(imagedata, linedata, showplot, savefig, path):

            plt.imshow(imagedata, cmap='gray')

            if linedata is not None and (showplot or savefig):
                for line in linedata:
                    for rho, theta in line:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        plt.plot([x1, x2], [y1, y2], color='red')

                plt.title("Detected Lines")
                plt.xlim(0, imagedata.shape[1]-1)
                plt.ylim(imagedata.shape[0]-1,0)
                # plt.axis('off')  # Turn off axis numbers and ticks
                plt.xlabel("Left to right")
                plt.ylabel("Posterior to anterior")
                if showplot:
                    plt.show()
                if savefig:
                    plt.savefig(path + "francis_grid.png")
                    plt.close()
            
            return None
        
    class sta:
        def cutoutRect(imagedata):
            thld = utilFunc.getThreshold.otsuMethod(imagedata)
            thld_img = utilFunc.createThresholdImage(imagedata,thld)
            centerpoint = utilFunc.findCenter.centerOfMassFilled(thld_img)

            diameter = utilFunc.measureDistance(thld_img, centerpoint, 45)

            center_offset_y = int(diameter * 0.05)
            center_offset_x = int(diameter * 0.40)

            coord1 = centerpoint[0] - center_offset_y
            coord2 = centerpoint[0] + center_offset_y
            coord3 = centerpoint[1] - center_offset_x
            coord4 = centerpoint[1] + center_offset_x


            center_cutout = imagedata[coord1:coord2,coord3:coord4]
            return center_cutout
        
        def measureLength(imagedata, spacing=1):
            thld = utilFunc.getThreshold.otsuMethod(imagedata) 

            rowsum = np.sum(utilFunc.createThresholdImage(imagedata, thld),axis=1)
            rowsum_conv = np.abs(np.convolve(rowsum,[1,-1],"valid"))
            a = rowsum_conv.shape[0]
            b = int(a*0.25)
            border = np.where(rowsum_conv[b:-b] == np.max(rowsum_conv[b:-b]))[0][0] +1  # Edgecase fixen mit 2 minima/was wenn beide gleich lang. Idee: bei fehlschlag einfach mitte nehmen.

            upperhalf = imagedata[:border+b]
            lowerhalf = imagedata[border+b+1:]

            upperthldimg = utilFunc.createThresholdImage(upperhalf,thld)
            lowerthldimg = utilFunc.createThresholdImage(lowerhalf,thld)

            upperBorder = np.where(np.sum(upperthldimg, axis=0) > (max(np.sum(upperthldimg, axis=0)))/2)
            lowerBorder = np.where(np.sum(lowerthldimg, axis=0) > (max(np.sum(lowerthldimg, axis=0)))/2)

            upperLength = np.max(upperBorder) - np.min(upperBorder)
            lowerLength = np.max(lowerBorder) - np.min(lowerBorder)
            # meanLength = 0.2 * (upperLength * lowerLength) / (upperLength + lowerLength) * spacing # ACR Spec formula
            meanLength = np.mean([0.125 * i * spacing - 1 for i in [upperLength, lowerLength]]) # Mathematically reasonable formula



            return meanLength, (np.max(upperBorder),np.min(upperBorder),np.max(lowerBorder),np.min(lowerBorder)), border+b


    class spa:
        def cutoutRect(imagedata):
            thld = utilFunc.getThreshold.otsuMethod(imagedata)
            thld_img = utilFunc.createThresholdImage(imagedata, thld)
            centerpoint = utilFunc.findCenter.centerOfMassFilled(thld_img)

            diameter = np.mean([utilFunc.measureDistance(thld_img, centerpoint, i) for i in [45, 135]])
            coord_divider = (centerpoint[0] - int(0.4*diameter),centerpoint[1])

            center_offset_y = int(diameter * 0.03)
            center_offset_x = int(diameter * 0.07)

            coord1 = coord_divider[0] - (1 * center_offset_y)
            coord2 = coord_divider[0] + (5 * center_offset_y)
            coord3 = coord_divider[1] - center_offset_x
            coord4 = coord_divider[1] + center_offset_x


            center_cutout = imagedata[coord1:coord2,coord3:coord4]
            thld_rect = utilFunc.getThreshold.otsuMethod(center_cutout)
            cutoutrect = utilFunc.createThresholdImage(center_cutout,thld_rect)
            return cutoutrect
        

        def getPositionDifference(thldimage, showplot=False):
            column_sum = np.sum(thldimage, axis=0)

            midpoint = int(column_sum.shape[0] / 2)
            midpoint_offset = int(column_sum.shape[0] * 0.1)

            length_left  = np.median(column_sum[:midpoint-midpoint_offset])
            length_right = np.median(column_sum[midpoint+midpoint_offset:])

            diff_in_pix = length_left - length_right
            if showplot:
                plt.imshow(thldimage)
                plt.hlines(length_left ,0,midpoint-midpoint_offset)
                plt.hlines(length_right,midpoint+midpoint_offset, column_sum.shape[0]-1)
                plt.vlines(midpoint, length_left, length_right, colors="red")
                plt.show()

            return diff_in_pix, [length_left,length_right]
        
    class psg:
        """Percent-Signal-Ghosting"""
        def calcPSG(imagedata, thldimg, center, showplot=False):
            diameter = np.mean([utilFunc.measureDistance(thldimg, center, i) for i in [45, 135]])
            radius = diameter/2
            centermask = utilFunc.circularROI(imagedata, center, 0.7*radius)

            spaceTop = imagedata.shape[0] - center[0] - radius
            spaceBot = center[0] - radius
            spaceRight = imagedata.shape[1] - center[1] - radius
            spaceLeft = center[1] - radius
            ellipseRadius = np.min([spaceBot,spaceLeft,spaceRight,spaceBot])

            topmask = utilFunc.circularROI(imagedata, center, ellipseRadius/2,False,0,radius + 1.3 * spaceTop/2,2,0.5)
            botmask = utilFunc.circularROI(imagedata, center, ellipseRadius/2,False,0,-(radius + 1.3 * spaceBot/2),2,0.5)
            rightmask = utilFunc.circularROI(imagedata, center, ellipseRadius/2,False,radius + 1.3 * spaceRight/2,0,0.5,2)
            leftmask = utilFunc.circularROI(imagedata, center, ellipseRadius/2,False,-(radius + 1.3 * spaceLeft/2),0,0.5,2)

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
  