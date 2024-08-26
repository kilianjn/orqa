import mrphantomqa

paths = [
    # "/Volumes/KINGSTON/angle_test/Phantom^Francis_2024.08.13-12_01_25-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240813_120145.700000",
    "/Volumes/KINGSTON/Francis/Kamera1/06242024",
    "/Volumes/KINGSTON/Francis/Kamera1/07122024",
    "/Volumes/KINGSTON/Francis/Kamera1/07222024",
    "/Volumes/KINGSTON/Francis/Kamera1/07302024",
    "/Volumes/KINGSTON/Francis/Kamera1/08052024"
]
workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code"

for path in paths:
    
    dfs = mrphantomqa.dicomFolderScanner(path)
    dfs.choose_scan_via_menu(True)
    # dfs.list_scans()
    # dfs.choose_scan("123939.620000 t1_tse_tf1soSE_9sl_medFilter")
    # dfs.sequence_properties()
    dfs.get_data()
    # dfs.view_raw_image()

    Analyzer = mrphantomqa.francisAnalyzer(dfs, workdir)
    Analyzer.runall()
    # for i in range(-5,6):
    #     Analyzer.resolution(offset=i)
    #     print(Analyzer.res_RES)

    # Analyzer.resolution(False)
    # Analyzer.low_contrast(False)
    # Analyzer.uniformity(False)
    # Analyzer.size(False)
    # Analyzer.grid(False)
    # Analyzer.thickness(False)
    # Analyzer.position(False)
    # Analyzer.ghosting(False)

    del dfs, Analyzer


































# Analyzer.add2csv()
# Analyzer.create_report()
# Analyzer.create_longterm_report()