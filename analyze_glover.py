import mrphantomqa

paths = [
    "/Volumes/KINGSTON/glover7T/05082024",
    "/Volumes/KINGSTON/glover7T/13082024",
    "/Volumes/KINGSTON/glover7T/22082024",
    "/Volumes/KINGSTON/glover7T/23082024",
    "/Volumes/KINGSTON/glover7T/08282024",
    "/Volumes/KINGSTON/glover7T/09092024",
]

for path in paths:

    dfs = mrphantomqa.dicomFolderScanner(path)
    dfs.choose_scan_via_menu(True)
    dfs.list_scans()
    # dfs.choose_scan("123939.620000 t1_tse_tf1soSE_9sl_medFilter")
    # dfs.sequence_properties()
    dfs.get_data()
    # dfs.view_raw_image()


    gm = mrphantomqa.gloverAnalyzer(dfs,8)

    gm.add2csv()
    gm.create_longterm_report()

    del dfs, gm




# folder_path = "/Volumes/KINGSTON/glover7T/GloverPhantom24112023_2023.11.24-14_18_42-STD-1.3.12.2.1107.5.99.3_20000101/head_library_20231124_141954.600000"
# folder_path = "/Volumes/KINGSTON/glover7T/qa_test_2_2023.11.30-16_05_23-STD-1.3.12.2.1107.5.99.3_19900101/Neuro_Vuxen_20231130_160526.300000"
# folder_path = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/23102614/55050000"
# folder_path = "/Volumes/SANDISK/Francis/cmrr_testscan200slices/Phantom^Francis_test_final_2024.07.25-12_31_30-DST~Patient1/Neuro_Vuxen_20240725_123146.500000"
# # folder_path = "/Volumes/KINGSTON/glover7T/cylinder/20240813"
# scanner = mrphantomqa.dicomFolderScanner(folder_path)
# scanner.list_scans()
# scanner.choose_scan("124546.095000 cmrr_mbep2d_se_11sl")
# # scanner.sequence_properties()
# scanner.get_data()
# # scanner.view_raw_image()

# # scanner.view_raw_image()
# gm = mrphantomqa.gloverAnalyzer(scanner,7)

# gm.add2csv()
# gm.create_longterm_report()

# gm.viewer.view_image(gm.signalImage,"SignalImage")

# # gm.viewer.view_image(gm.tempFlucImage,"temporalFlucNoise")
# # gm.viewer.view_image(gm.sfnrImage,"SFNR Image")
# # gm.viewer.view_image(gm.staticSpatialNoiseImage,"SNR Image")

# print(f"SNR {gm.snrSV}")
# print(f"SFNR {gm.sfnrSV}")

# print(f"Percent Fluctuation {gm.percentFluc}")
# print(f"Drift {gm.drift}")

# gm.viewer.plot_pixel_over_time(gm.residualsSVs,title="Residual SVs over time")
# gm.viewer.plot_pixel_over_time(gm.residualsSVsFT, title="FT'd SVs over Frequency")

# gm.weisskoffAnalysis(21,True)

