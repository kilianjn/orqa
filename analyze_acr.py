import mrphantomqa

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
filepaths = [
    "/Volumes/KINGSTON/ACR/Kamera13/05242024",
    "/Volumes/KINGSTON/ACR/Kamera13/05292024",
    "/Volumes/KINGSTON/ACR/Kamera13/06052024",
    "/Volumes/KINGSTON/ACR/Kamera13/06122024",
    "/Volumes/KINGSTON/ACR/Kamera13/06172024",
    "/Volumes/KINGSTON/ACR/Kamera13/06242024",
    "/Volumes/KINGSTON/ACR/Kamera13/07122024",
    "/Volumes/KINGSTON/ACR/Kamera13/07222024",
    "/Volumes/KINGSTON/ACR/Kamera13/07302024",
    "/Volumes/KINGSTON/ACR/Kamera13/08052024"
]
workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code"

for filepath in filepaths:
    dfs = mrphantomqa.dicomFolderScanner(filepath, True)
    dfs.list_scans()
    dfs.choose_scan_via_menu(True)
    # dfs.choose_scan("123532.535000 ACR_T1")
    # dfs.sequence_properties()
    dfs.get_data()

    analyzer = mrphantomqa.acrAnalyzer(dfs, workdir)
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