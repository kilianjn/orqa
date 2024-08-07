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

# filepath = "/Volumes/KINGSTON/Francis/Kamera3/Goldstandard/Phantom^Francis_2024.06.11-16_36_52-DST-1.3.12.2.1107.5.99.3_20000101"
# filepath = "/Volumes/SANDISK/Francis/FrancisFantome_2024.06.13-10_10_19-DST-1.3.12.2.1107.5.99.3_19000101/Neuro_Vuxen_20240613_101047.700000"
# filepath = "/Volumes/KINGSTON/Francis/Kamera2/06172024/Phantom^Francis_NaCl_2024.06.14-15_59_53-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240614_160012.700000"
filepath = "/Volumes/KINGSTON/Francis/Kamera2/07152024/Phantom^Francis_2024.07.15-12_36_55-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240715_123712.000000"
filepath = "/Volumes/KINGSTON/Francis/Kamera13/07122024/Phantom^Francis_2024.07.12-12_07_52-DST-1.3.12.2.1107.5.99.3_20000101/NEURO_Hjärna_20240712_120814.200000"
filepath = "/Volumes/KINGSTON/Francis/Kamera1/07122024"
filepath = "/Volumes/KINGSTON/Francis/Kamera2/08052024"
filepath = askForPath()
dfs = mrphantomqa.dicomFolderScanner(filepath)
dfs.menu_gui()
# dfs.list_scans()
# dfs.choose_scan("124630.342000 FP_T1")
# dfs.choose_scan("124630.342000 FP_T1")
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