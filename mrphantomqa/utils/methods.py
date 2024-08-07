import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import interpolate, ndimage
from skimage.morphology import flood_fill
import tkinter as tk
from tkinter import ttk
from tqdm import tqdm

#Debugging
import mrphantomqa.utils.viewer as viw

class functions:
    class findCenter:
        def centerOfMass(thldimage, showplot = False):
            # assert len(np.unique(thldimage)) == 2,"Image values must be binary"
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
                plt.imshow(thldimage)
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

    def getHistogram(imagedata, showplot=False):
        # all_bins = np.linspace(int(np.min(imagedata)), int(np.max(imagedata)+1), int(np.max(imagedata)-np.min(imagedata)+2))
        all_bins = np.arange(np.max(imagedata)+2)
        x,bins = np.histogram(imagedata,bins=all_bins)
        if showplot:
            plt.plot(bins[0:-1],x)
            plt.show()

        return x,bins[0:-1]

    def createThresholdImage(imagedata, threshold, showplot=False):
        thld_image = np.zeros(imagedata.shape)
        mask = np.ma.getmask(imagedata)
        thld_image[np.where(imagedata >= threshold)] = 1
        thld_image = thld_image.astype(int)
        
        if mask.any():
            thld_image = np.ma.masked_array(thld_image, mask)

        if showplot:
            plt.imshow(thld_image)
            plt.show()
        return thld_image

    def draw_line(imagedata, startpoint, angle):
        angle = np.deg2rad(angle)
        y_startpoint, x_startpoint = startpoint

        height, width = imagedata.shape
        lineArray = np.zeros((height, width), dtype=bool)
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
        """Draws Line in one direction from startpoint. Returns mask and values across the line"""
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
        """creates mask in binary image using floodfill and filles holes in mask"""
        assert len(np.unique(thldimg)) == 2,"Image values must be binary"
        coordinates = None

        temp = flood_fill(thldimg, startpoint, 2)
        mask = temp == 2
        # mask = ndimage.binary_fill_holes(mask)
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

    def removeHoles(thldimage, showplot=False):
        """Fills shape and removes holes"""
        assert len(np.unique(thldimage)) == 2,"Image values must be binary"

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


        if showplot:
            plt.imshow(tempimage)
            plt.show()

        return tempimage.astype(bool)
    
    def convolveImage(imagedata, kernel):
        """Returns convolved image with """
        return cv.filter2D(imagedata, -1, kernel)
    
    def measureDistance(imagedata, startpoint, angle_in_deg, spacing=[1,1], showplot=False):
        """Finds length of straight line at an angle. Image has to be thresholded and Length is measured
        as length between the first and last 1 along the line in the picture."""
        assert len(np.unique(imagedata)) == 2 or len(np.unique(imagedata)) == 3,"Image values must be binary" #3 for the case of masked arrays
        spacing_y, spacing_x = spacing
        y_startpoint, x_startpoint = startpoint

        line = functions.draw_line(imagedata, startpoint, angle_in_deg)
        line_image_overlap = np.ma.masked_array(imagedata,~line)
        nonzero_coords = np.argwhere(line_image_overlap)

        if len(nonzero_coords) > 0:
            # Find the two most outer points
            outer_points = np.array([nonzero_coords[0], nonzero_coords[-1]])
            length = np.round(np.sqrt(((outer_points[0,1] - outer_points[1,1])*spacing_x)**2 + ((outer_points[0,0] - outer_points[1,0])*spacing_y)**2),3)

            if showplot:
                # Plot imagedata
                plt.imshow(imagedata, cmap='gray')
                # Plot the two most outer points
                plt.scatter(outer_points[:, 1], outer_points[:, 0], c='red', marker='x')
                plt.plot(outer_points[:, 1], outer_points[:, 0], label=f"Length = {length}")
                # Plot startpoint
                plt.scatter(x_startpoint, y_startpoint, c='blue', marker='x')
                plt.legend()
                plt.show()
            return length
        
        else:
            print("No non-zero elements found in line_image_overlap.")
            return 0