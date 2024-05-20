import mrphantomqa
import mrphantomqa.utils.viewer as viw

#IMPORTANT: Export MRI data in Interoperability Mode!!!

filepath = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/ownTest/T1_ACR_0011"
# filepath = "../mntvol"
# filepath = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/albin1.5tesla"
# filepath = "/Volumes/KINGSTON/ACR/Kamera4/05142024"

dfs = mrphantomqa.dicomFolderScanner(filepath, True)
dfs.list_scans()
dfs.choose_scan("141246.280000 T1 acr")
dfs.sequence_properties()
dfs.get_data()

analyzer = mrphantomqa.acrAnalyzer(dfs)
# analyzer.high_contrast_spatial_resolution(True) # WIP...
analyzer.geometric_accuracy(False, True) # Done.
analyzer.slice_position_accuracy(False, True) # Done.
analyzer.image_intensity_uniformity(False, True) # Done.
analyzer.percent_ghosting_ratio(False, True) # Done.
analyzer.low_contrast_object_detectibility("edges1", False, True) # Done.
analyzer.slice_thickness_accuracy(False, True) # Done.

analyzer.get_results()
analyzer.createReport()
# analyzer.add2csv()
