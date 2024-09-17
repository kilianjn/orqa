import mrphantomqa

paths = [
    "/Volumes/KINGSTON/francis t1_t2/Phantom^Francis_2024.08.30-12_04_54-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240830_120510.000000/t2_tse_tf1soSE_9sl_medFilter_5_MR",
]
workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/thesiswork/t2"

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