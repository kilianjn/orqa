import mrphantomqa
import mrphantomqa.utils.viewer as viw
from mrphantomqa.utils.methods import functions as func

import numpy as np
import matplotlib.pyplot as plt

filepath = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/ownTest/T1_ACR_0011"
# filepath = "../mntvol"
# filepath = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/albin1.5tesla"
# filepath = "/Volumes/KINGSTON/3dprint1_5T/DICOM/24032212/42080000"
filepath = "/Volumes/KINGSTON/Francis/Kamera1/04192024"

dfs = mrphantomqa.dicomFolderScanner(filepath,True)
dfs.list_scans()
dfs.choose_scan("125152.121000 FP_T2")
dfs.sequence_properties()
dfs.get_data()
dfs.view_raw_image()


# analyzer = mrphantomqa.acrAnalyzer(dfs)
img = dfs.imagedata[0]

interpolatedArray = np.zeros((11,img.shape[1]*2,img.shape[2]*2))
interpolatedArray[8] = func.interpolateImage(img[8],2)
img = interpolatedArray
# viw.plot2D(img)
thld = func.getThreshold.findPeak(img[8],10)
thldimg = func.createThresholdImage(img[8], thld*0.8)
center = func.findCenter.centerOfMass(thldimg)
print(center)

center = (258,264)
circMask = func.circularROI(img[8], center,80)

cutout = np.ma.masked_array(img[8], circMask)
viw.plot2D(cutout)
lineArraysByAngle, _ = func.radialTrafo(cutout, center)

edgeImage = []
for angle in range(lineArraysByAngle.shape[0]):
    edgeImage.append(np.convolve(lineArraysByAngle[angle], [1,0,-1], "valid"))
edgeImage = np.array(edgeImage)
lineArraysByAngle = edgeImage[:,:int(edgeImage.shape[1]*0.95)]

viw.plot2D(lineArraysByAngle)
