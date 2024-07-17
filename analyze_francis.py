import mrphantomqa
from mrphantomqa import askforpath
from mrphantomqa.francis_analyzer import francisAnalyzer

### DEBUG ###
# from mrphantomqa.utils.methods import functions as utilfunc
# from mrphantomqa.francis.methods import functions as francisfunc
# import mrphantomqa.utils.viewer as viw
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt

filepath = "/Volumes/KINGSTON/Francis/Kamera3/Goldstandard/Phantom^Francis_2024.06.11-16_36_52-DST-1.3.12.2.1107.5.99.3_20000101"
# filepath = "/Volumes/SANDISK/Francis/FrancisFantome_2024.06.13-10_10_19-DST-1.3.12.2.1107.5.99.3_19000101/Neuro_Vuxen_20240613_101047.700000"
filepath = "/Volumes/KINGSTON/Francis/Kamera2/06172024/Phantom^Francis_NaCl_2024.06.14-15_59_53-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240614_160012.700000"
# askforpath()
dfs = mrphantomqa.dicomFolderScanner(filepath)
# dfs.menuGUI()
dfs.list_scans()
dfs.choose_scan("160615.436000 t1_tse_5slice")
# dfs.choose_scan("164725.181000 T1 acr")
dfs.sequence_properties()
dfs.get_data()
# dfs.view_raw_image()

# Analyzer = francisAnalyzer(dfs)
# Analyzer.resolution(True)
# Analyzer.low_contrast(True)
# Analyzer.uniformity(True)
# Analyzer.size(True)
# Analyzer.grid(True)

# img = dfs.imagedata[0]
# img=img[2]
# thld = utilfunc.getThreshold.otsuMethod(img,10)
# thldimg = utilfunc.createThresholdImage(img,thld)
# cp = utilfunc.findCenter.centerOfMassFilled(thldimg)
