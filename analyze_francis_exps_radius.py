import mrphantomqa
import cv2 as cv
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from mrphantomqa.utils.methods import functions as utl
paths = [
    "/Volumes/KINGSTON/Francis/Kamera1/06242024",
    "/Volumes/KINGSTON/Francis/Kamera1/07122024",
    "/Volumes/KINGSTON/Francis/Kamera1/07222024",
    "/Volumes/KINGSTON/Francis/Kamera1/07302024",
    "/Volumes/KINGSTON/Francis/Kamera1/08052024",
    "/Volumes/KINGSTON/Francis/Kamera1/08122024",
    "/Volumes/KINGSTON/Francis/Kamera1/08192024",
    "/Volumes/KINGSTON/Francis/Kamera1/08272024",
    "/Volumes/KINGSTON/Francis/Kamera1/09092024",

    "/Volumes/KINGSTON/Francis/Kamera2/06242024",
    "/Volumes/KINGSTON/Francis/Kamera2/07152024",
    "/Volumes/KINGSTON/Francis/Kamera2/07222024",
    "/Volumes/KINGSTON/Francis/Kamera2/07302024",
    "/Volumes/KINGSTON/Francis/Kamera2/08052024",
    "/Volumes/KINGSTON/Francis/Kamera2/08132024",
    "/Volumes/KINGSTON/Francis/Kamera2/08192024",
    "/Volumes/KINGSTON/Francis/Kamera2/08282024",
    "/Volumes/KINGSTON/Francis/Kamera2/09092024",

    "/Volumes/KINGSTON/Francis/Kamera13/06242024",
    "/Volumes/KINGSTON/Francis/Kamera13/07122024",
    "/Volumes/KINGSTON/Francis/Kamera13/07222024",
    "/Volumes/KINGSTON/Francis/Kamera13/07302024",
    "/Volumes/KINGSTON/Francis/Kamera13/08052024",
    "/Volumes/KINGSTON/Francis/Kamera13/08122024",
    "/Volumes/KINGSTON/Francis/Kamera13/08192024",
    "/Volumes/KINGSTON/Francis/Kamera13/08272024",
    "/Volumes/KINGSTON/Francis/Kamera13/09092024"
]
workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code"



dfs = mrphantomqa.dicomFolderScanner(paths[11])
dfs.choose_scan_via_menu(True)
dfs.get_data()

Analyzer = mrphantomqa.francisAnalyzer(dfs, workdir)
test = Analyzer.imagedata[0]
thld = utl.getThreshold.findPeak(test,5)
thldimg = utl.createThresholdImage(test, thld/2)
thldimg = utl.removeHoles(thldimg)
thldimg = thldimg.astype(np.uint8) * 255


tst = cv.HoughCircles(thldimg, cv.HOUGH_GRADIENT, 0.01, thldimg.shape[0], param1=1, param2=1,minRadius=50, maxRadius=100)
print(len(np.where(tst[0,:,2]==np.max(tst[0,:,2]))[0]))
plt.imshow(thldimg,cmap="grey")
if tst is not None:
    tst = np.uint16(np.around(tst))
    for i in np.where(tst[0,:,2]==np.max(tst[0,:,2]))[0]:
        center = (tst[0,i,0], tst[0,i,1])
        radius = tst[0,i,2]
        # circle center
        
plt.imshow(~utl.circularROI(thldimg,center,radius),alpha=0.2,cmap="jet")

plt.show()