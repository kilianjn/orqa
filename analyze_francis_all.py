import mrphantomqa

paths = [
    "/Volumes/KINGSTON/Francis/Kamera1/06242024",
    "/Volumes/KINGSTON/Francis/Kamera1/07122024",
    "/Volumes/KINGSTON/Francis/Kamera1/07222024",
    "/Volumes/KINGSTON/Francis/Kamera1/07302024",
    "/Volumes/KINGSTON/Francis/Kamera1/08052024",
    "/Volumes/KINGSTON/Francis/Kamera1/08122024",
    "/Volumes/KINGSTON/Francis/Kamera1/08192024",
    "/Volumes/KINGSTON/Francis/Kamera1/08272024",
    "/Volumes/KINGSTON/Francis/Kamera1/09092024",

    "/Volumes/KINGSTON/Francis/Kamera2/06242024",
    "/Volumes/KINGSTON/Francis/Kamera2/07152024",
    "/Volumes/KINGSTON/Francis/Kamera2/07222024",
    "/Volumes/KINGSTON/Francis/Kamera2/07302024",
    "/Volumes/KINGSTON/Francis/Kamera2/08052024",
    "/Volumes/KINGSTON/Francis/Kamera2/08132024",
    "/Volumes/KINGSTON/Francis/Kamera2/08192024",
    "/Volumes/KINGSTON/Francis/Kamera2/08282024",
    "/Volumes/KINGSTON/Francis/Kamera2/09092024",

    "/Volumes/KINGSTON/Francis/Kamera13/06242024",
    "/Volumes/KINGSTON/Francis/Kamera13/07122024",
    "/Volumes/KINGSTON/Francis/Kamera13/07222024",
    "/Volumes/KINGSTON/Francis/Kamera13/07302024",
    "/Volumes/KINGSTON/Francis/Kamera13/08052024",
    "/Volumes/KINGSTON/Francis/Kamera13/08122024",
    "/Volumes/KINGSTON/Francis/Kamera13/08192024",
    "/Volumes/KINGSTON/Francis/Kamera13/08272024",
    "/Volumes/KINGSTON/Francis/Kamera13/09092024"
]
workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code"

for path in paths:
    
    dfs = mrphantomqa.dicomFolderScanner(path)
    dfs.choose_scan_via_menu(True)
    dfs.list_scans()
    # dfs.choose_scan("123939.620000 t1_tse_tf1soSE_9sl_medFilter")
    # dfs.sequence_properties()
    dfs.get_data()
    # dfs.view_raw_image()

    Analyzer = mrphantomqa.francisAnalyzer(dfs, workdir)
    Analyzer.runall()
    # Analyzer.resolution(True)

    del dfs, Analyzer


































# Analyzer.add2csv()
# Analyzer.create_report()
# Analyzer.create_longterm_report()