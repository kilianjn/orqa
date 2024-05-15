import mrphantomqa

# folder_path = "/Volumes/KINGSTON/glover7T/GloverPhantom24112023_2023.11.24-14_18_42-STD-1.3.12.2.1107.5.99.3_20000101/head_library_20231124_141954.600000"
# folder_path = "/Volumes/KINGSTON/glover7T/qa_test_2_2023.11.30-16_05_23-STD-1.3.12.2.1107.5.99.3_19900101/Neuro_Vuxen_20231130_160526.300000"
# folder_path = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/23102614/55050000"
folder_path = "/Volumes/KINGSTON/glover7T/qa2/qa_test_2_2023.11.30-16_05_23-STD-1.3.12.2.1107.5.99.3_19900101/Neuro_Vuxen_20231130_160526.300000"

scanner = mrphantomqa.dicomFolderScanner(folder_path)
scanner.list_scans()
scanner.choose_scan("160241.148000 ep2d_128_3mm_p2_bw3004_AP_fullABox_absolute_accFac4_ref36_dyn")
# scanner.sequence_properties()
scanner.get_data()

# scanner.view_raw_image()
gm = mrphantomqa.gloverAnalyzer(scanner.imagedata[:,10,:,:])

# gm.viewer.view_image(gm.signalImage,"SignalImage")



# gm.viewer.view_image(gm.tempFlucImage,"temporalFlucNoise")
# gm.viewer.view_image(gm.sfnrImage,"SFNR Image")
# gm.viewer.view_image(gm.staticSpatialNoiseImage,"SNR Image")

print(f"SNR {gm.snrSV}")
print(f"SFNR {gm.sfnrSV}")

print(f"Percent Fluctuation {gm.percentFluc}")
print(f"Drift {gm.drift}")

# gm.viewer.plot_pixel_over_time(gm.residualsSVs,title="Residual SVs over time")
# gm.viewer.plot_pixel_over_time(gm.residualsSVsFT, title="FT'd SVs over Frequency")

# gm.weisskoffAnalysis(21,True)


