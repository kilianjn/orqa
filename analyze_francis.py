import mrphantomqa

path_to_dicom = ""
savedir = ""


dfs = mrphantomqa.dicomFolderScanner(path_to_dicom)
dfs.choose_scan_via_menu()
dfs.get_data()

Analyzer = mrphantomqa.francisAnalyzer(dfs, savedir)
Analyzer.runall()



