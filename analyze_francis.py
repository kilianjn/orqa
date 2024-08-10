import mrphantomqa
from mrphantomqa.francis_analyzer import francisAnalyzer



paths = [
    "/Volumes/KINGSTON/Francis/Kamera1/06242024",
    "/Volumes/KINGSTON/Francis/Kamera1/07122024",
    "/Volumes/KINGSTON/Francis/Kamera1/07222024",
    "/Volumes/KINGSTON/Francis/Kamera1/07302024",
    "/Volumes/KINGSTON/Francis/Kamera1/08052024"
]

for path in paths:

    dfs = mrphantomqa.dicomFolderScanner(path)
    dfs.menu_gui(True)
    # dfs.list_scans()
    # dfs.choose_scan("123939.620000 t1_tse_tf1soSE_9sl_medFilter")
    # dfs.sequence_properties()
    dfs.get_data()
    # dfs.view_raw_image()

    Analyzer = francisAnalyzer(dfs)
    Analyzer.runall()
    # Analyzer.resolution(False, True)
    # Analyzer.low_contrast(False, True)
    # Analyzer.uniformity(False, True)
    # Analyzer.size(False, True)
    # Analyzer.grid(False, True)
    # Analyzer.thickness(False, True)
    # Analyzer.position(False, True)
    # Analyzer.ghosting(False ,True)

    # Analyzer.add2csv()
    # Analyzer.create_report()
    # Analyzer.create_longterm_report()

    del dfs, Analyzer





















