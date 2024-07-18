from .utils.methods import functions as utilfunc
from .francis.methods import functions as francisfunc

import matplotlib.pyplot as plt
import numpy as np

class francisAnalyzer:
    def __init__(self, data) -> None:
        # self.imagedata_loc = data.imagedata[0]
        self.imagedata = data.imagedata[0]
        self.metadata = data.metadata if hasattr(data, 'metadata') else None
        self.spacing = self.metadata[0x52009230][0][0x00289110][0][0x00280030].value

        self.res_RES = None
        self.res_RES_SD = None
        self.res_LCOD = None
        self.res_IIU = None
        self.res_STA = None

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
        
        self.res_RES = (5 - (np.mean(longestLength)/50 * 5))*2
        self.res_RES_SD = (5 - (np.std(longestLength)/50 * 5))*2
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

        circMask = utilfunc.circularROI(img, center_new,int(0.65*radius), True)

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
            plt.imshow(img)

            plt.subplot(122)
            plt.imshow(lineArraysByAngle)
            plt.title(f"{countedSpokes} spokes counted")
            for i in spokePosition:
                plt.hlines(i, 0, int(lineArraysByAngle.shape[1]-1), colors="red", alpha=0.3)
            if showplot:
                plt.show()
            if savefig:
                plt.savefig("test5.png")
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
            plt.imshow(convolvedMaskedImg)
            plt.scatter(maxCoord[1],maxCoord[0],facecolors='none',edgecolors='r', s=10**2, label=f"max: {maxValue}")
            plt.scatter(minCoord[1],minCoord[0],facecolors='none',edgecolors='b', s=10**2, label=f"min: {minValue}")
            plt.legend()
            if showplot:
                plt.show()
            if savefig:
                plt.savefig("francis_uniformity.png")

        self.res_IIU = 100 * (1-(maxValue-minValue)/(maxValue+minValue))

    def thickness(self, showplot=False, savefig=False):
        img = self.imagedata[6]
        rectimg = francisfunc.sta.cutoutRect(img)
        test = francisfunc.sta.measureLength(rectimg)
        pass

    def position(self, showplot=False, savefig=False):

        pass

    def grid(self, showplot=False, savefig=False):
        img = self.imagedata[0]
        img_grid_pre = francisfunc.grid.cutoutSquare(img)

        img_grid = francisfunc.grid.imagePreProcessing(img_grid_pre, False)

        ## Grid Detection using the Hough Transform
        dist_lines, angle_cross, lines = francisfunc.grid.gridDetect(img_grid)

        squaresize = (self.spacing[0] * dist_lines[0]) * (self.spacing[1] * dist_lines[1])

        francisfunc.grid.printImage(img_grid_pre, lines, showplot, savefig)

        self.res_Grid_size = squaresize
        self.res_Grid_angle = np.rad2deg(angle_cross)

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