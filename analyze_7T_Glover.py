import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import mrphantomqa
from mrphantomqa.utils.methods import functions as utfc
import mrphantomqa.utils.viewer as viw

CYLINDER = False

folder_path = "/Volumes/KINGSTON/glover7T/qa2/qa_test_2_2023.11.30-16_05_23-STD-1.3.12.2.1107.5.99.3_19900101/Neuro_Vuxen_20231130_160526.300000"

scanner = mrphantomqa.dicomFolderScanner(folder_path)
scanner.list_scans()
scanner.choose_scan("160241.148000 ep2d_128_3mm_p2_bw3004_AP_fullABox_absolute_accFac4_ref36_dyn")
scanner.sequence_properties()
scanner.get_data()
# scanner.view_raw_image()

gm = mrphantomqa.gloverAnalyzer(scanner.imagedata[:,15])

# gm.viewer.view_image(gm.signalImage,"SignalImage")
gm.viewer.view_image(gm.tempFlucImage,"temporalFlucNoise")
gm.viewer.view_image(gm.sfnrImage,"SFNR Image")
gm.viewer.view_image(gm.staticSpatialNoiseImage,"SNR Image")

# print(f"SNR {gm.snrSV}")
# print(f"SFNR {gm.sfnrSV}")

# print(f"Percent Fluctuation {gm.percentFluc}")
# print(f"Drift {gm.drift}")

# gm.viewer.plot_pixel_over_time(gm.residualsSVs,title="Residual SVs over time")
# gm.viewer.plot_pixel_over_time(gm.residualsSVsFT, title="FT'd SVs over Frequency")

# gm.weisskoffAnalysis(21,True)


if CYLINDER:

    ana = mrphantomqa.cylinderAnalyzer(scanner)
    ana.sequence_name = "154759.038000 ep2d_128_3mm_p2_bw3004_AP_fullABox_absolute_accFac2_ref30_dyn"
    ana.loadMetrics()
    test = ana.all_metrics.copy()

    ana1 = mrphantomqa.cylinderAnalyzer(scanner)
    ana1.sequence_name = "155518.979000 ep2d_128_3mm_p2_bw3004_AP_fullABox_absolute_accFac3_ref27_dyn"
    ana1.loadMetrics()
    test1 = ana1.all_metrics.copy()

    diff0 = test[0] - test1[0]
    diff1 = test[1] - test1[1]
    diff2 = test[2] - test1[2]
    diff3 = test[3] - test1[3]

    # viw.plot3D(diff2)
    plt.subplot(221)
    viw.differenceViewer(diff2, "")
    plt.subplot(222)
    viw.differenceViewer(diff3, "")
    plt.subplot(223)
    plt.imshow(ana.imagedata[0,0])
    plt.scatter(test[1], test[0])
    plt.title("COM 2mm")
    plt.subplot(224)
    plt.imshow(ana.imagedata[0,0])
    plt.scatter(test1[1], test1[0])
    plt.title("COM 3mm")
    plt.show()
