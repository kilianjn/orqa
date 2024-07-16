import mrphantomqa
from mrphantomqa.utils.methods import functions as utilfunc
from mrphantomqa.francis.methods import functions as francisfunc
from mrphantomqa.francis_analyzer import francisAnalyzer

import mrphantomqa.utils.viewer as viw
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

filepath = "/Volumes/KINGSTON/Francis/Kamera3/Goldstandard/Phantom^Francis_2024.06.11-16_36_52-DST-1.3.12.2.1107.5.99.3_20000101"
filepath = "/Volumes/SANDISK/Francis/FrancisFantome_2024.06.13-10_10_19-DST-1.3.12.2.1107.5.99.3_19000101/Neuro_Vuxen_20240613_101047.700000"
filepath = "/Volumes/KINGSTON/Francis/Kamera2/06172024/Phantom^Francis_NaCl_2024.06.14-15_59_53-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240614_160012.700000"

dfs = mrphantomqa.dicomFolderScanner(filepath)
dfs.list_scans()
dfs.choose_scan("160615.436000 t1_tse_5slice")
dfs.sequence_properties()
dfs.get_data()
# dfs.view_raw_image()

img = dfs.imagedata[0]

# Analyzer = francisAnalyzer(dfs)
# Analyzer.resolution(True)
# Analyzer.low_contrast(True)
# Analyzer.uniformity(True)
# Analyzer.size(True)











### Grid Code
img = img[2]
img = img[110:210,100:200]

thld = int(utilfunc.getThreshold.otsuMethod(img))
thld_img = utilfunc.createThresholdImage(img,thld)
centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img)

testimg = img.astype(np.uint8)

test = thld_img * 255
test = test.astype(np.uint8)
test = ~test
kernel = np.ones((3,3),np.uint8)
test = cv.dilate(test,kernel,iterations = 1)
kernel = np.ones((3,3),np.uint8)
test = cv.erode(test,kernel,iterations = 1)

edges = cv.Canny(testimg,70,110,apertureSize = 3)
kernel = np.ones((3,3),np.uint8)
edges = cv.dilate(edges,kernel,iterations = 1)
kernel = np.ones((3,3),np.uint8)
edges = cv.erode(edges,kernel,iterations = 1)

lines = cv.HoughLines(test, 1, np.pi / 180, 99)
plt.imshow(testimg, cmap='gray')

if lines is not None:
    for line in lines:
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
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()
# print()

# viw.plot2D(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)))))
