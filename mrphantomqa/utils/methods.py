import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import flood_fill
from tqdm import tqdm

from ..acr.methods import functions as acrFunctions
from ..glover.methods import functions as gloverFunctions

class functions(acrFunctions, gloverFunctions):
    def __init__(self):
        super().__init__()

    def fillShape(thldimage, showplot=False):
        """Fills shape"""
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

        return tempimage
    
    def potatoness(thldimg, centerpoint, showplot=False):
        circMask = functions.circularROI(thldimg, centerpoint)
        np.ma.count()
        return