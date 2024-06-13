import mrphantomqa
from mrphantomqa.utils.methods import functions as utilfunc
from mrphantomqa.francis.methods import functions as francisfunc

import numpy as np
import matplotlib.pyplot as plt

filepath = "/Volumes/KINGSTON/Francis/Kamera3/Goldstandard/Phantom^Francis_2024.06.11-16_36_52-DST-1.3.12.2.1107.5.99.3_20000101"
filepath = "/Volumes/SANDISK/Francis/FrancisFantome_2024.06.13-10_10_19-DST-1.3.12.2.1107.5.99.3_19000101/Neuro_Vuxen_20240613_101047.700000"

dfs = mrphantomqa.dicomFolderScanner(filepath,True)
dfs.list_scans()
dfs.choose_scan("122434.613000 5slcs_t1_b0std_dRxE_elfilt_vspec_2avg_tr200_te9_plane")
dfs.sequence_properties()
dfs.get_data()
# dfs.view_raw_image()
spacing = dfs.metadata[0x52009230][0][0x00289110][0][0x00280030].value

# analyzer = mrphantomqa.acrAnalyzer(dfs)
img = dfs.imagedata[0]

thld = int(0.8 * utilfunc.getThreshold.findPeak(img[2],11, True))
thld_img = utilfunc.createThresholdImage(img[2],thld, True)
centerpoint = utilfunc.findCenter.centerOfMassFilled(thld_img,True)

utilfunc.radialTrafo(thld_img,centerpoint,True)
francisfunc.ga.measureDistance(thld_img,centerpoint,30,spacing, True)
# francisfunc.ga.measureDistance(thld_img,centerpoint,-30,[1,1])

