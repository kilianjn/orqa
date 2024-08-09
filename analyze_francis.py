import mrphantomqa
from mrphantomqa import askForPath
from mrphantomqa.francis_analyzer import francisAnalyzer

### DEBUG ###
# from mrphantomqa.utils.methods import functions as utilfunc
# from mrphantomqa.francis.methods import functions as francisfunc
# import mrphantomqa.utils.viewer as viw
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt

filepath = "/Volumes/KINGSTON/Francis/Kamera1/07122024"
filepath = "/Volumes/KINGSTON/Francis/Kamera2/06242024"
filepath = askForPath()
dfs = mrphantomqa.dicomFolderScanner(filepath)
dfs.menu_gui()
# dfs.list_scans()
# dfs.choose_scan("124630.342000 FP_T1")
# dfs.choose_scan("123939.620000 t1_tse_tf1soSE_9sl_medFilter")
dfs.sequence_properties()
dfs.get_data()
# dfs.view_raw_image()

Analyzer = francisAnalyzer(dfs)
Analyzer.resolution(False, True)
Analyzer.low_contrast(False, True)
Analyzer.uniformity(False, True)
Analyzer.size(False, True)
Analyzer.grid(False, True)
Analyzer.thickness(False, True)
Analyzer.position(False, True)
Analyzer.ghosting(False ,True)

Analyzer.add2csv()
Analyzer.create_report()
Analyzer.create_longterm_report()
