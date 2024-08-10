import mrphantomqa
import mrphantomqa.utils.viewer as viw

# filepath = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/ownTest/T1_ACR_0011"
# filepath = "../mntvol"
# filepath = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/albin1.5tesla"
# filepath = "/Volumes/KINGSTON/ACR/Kamera1/05282024"
# filepath = "/Volumes/KINGSTON/Best_case/ACR"

filepaths = [
    "/Volumes/KINGSTON/ACR/Kamera1/05132024",
    "/Volumes/KINGSTON/ACR/Kamera1/05222024",
    "/Volumes/KINGSTON/ACR/Kamera1/05282024",
    "/Volumes/KINGSTON/ACR/Kamera1/05312024",
    "/Volumes/KINGSTON/ACR/Kamera1/06042024",
    "/Volumes/KINGSTON/ACR/Kamera1/06142024",
    "/Volumes/KINGSTON/ACR/Kamera1/06172024",
    "/Volumes/KINGSTON/ACR/Kamera1/06242024",
    "/Volumes/KINGSTON/ACR/Kamera1/07122024",
    "/Volumes/KINGSTON/ACR/Kamera1/07222024",
    "/Volumes/KINGSTON/ACR/Kamera1/07302024",
    "/Volumes/KINGSTON/ACR/Kamera1/08052024"
]

for filepath in filepaths:
    dfs = mrphantomqa.dicomFolderScanner(filepath, True)
    dfs.menu_gui(True)
    # dfs.list_scans()
    # dfs.choose_scan("123532.535000 ACR_T1")
    # dfs.sequence_properties()
    dfs.get_data()

    analyzer = mrphantomqa.acrAnalyzer(None)
    analyzer.runall()
    del dfs, analyzer























# analyzer.high_contrast_spatial_resolution(True) # WIP...
# analyzer.geometric_accuracy(False, True) # Done.
# analyzer.slice_position_accuracy(False, True) # Done.
# analyzer.image_intensity_uniformity(False, True) # Done.
# analyzer.percent_ghosting_ratio(False, True) # Done.
# analyzer.low_contrast_object_detectibility("edges1", False, True) # Done.
# analyzer.slice_thickness_accuracy(False, True) # Done.

# analyzer.add2csv()
# analyzer.create_report()
# analyzer.create_longterm_report