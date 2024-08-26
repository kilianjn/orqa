import mrphantomqa

filepath = "/Volumes/KINGSTON/francis_angle_offsets/Phantom^Francis_tolerances_2024.08.23-12_15_23-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240823_121540.000000"
workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/thesiswork"

sequences = [
    "122211.627000 optimal",
    "122428.077000 optimal_deg+1",
    "122644.798000 optimal_deg+2",
    "122901.323000 optimal_deg+3",
    "123117.765000 optimal_deg+4",
    "123334.266000 optimal_deg+5",
    "123550.786000 optimal_deg-1",
    "123807.400000 optimal_deg-2",
    "124023.882000 optimal_deg-3",
    "124240.591000 optimal_deg-4",
    "124457.265000 optimal_deg-5"
]

dfs = mrphantomqa.dicomFolderScanner(filepath)
# dfs.choose_scan_via_menu(True)
# dfs.list_scans()
dfs.choose_scan("122211.627000 optimal")
# dfs.sequence_properties()
dfs.get_data()
# dfs.view_raw_image()

Analyzer = mrphantomqa.francisAnalyzer(dfs, workdir)
# Analyzer.runall()
test = []
for y in range(-5,6):
    test1 = []
    for x in range(-5,6):
        Analyzer.low_contrast(offsety=y, offsetx=x, showplot=True)
        print(f"{Analyzer.res_LCOD} for y={y} x={x}")
        test1.append(Analyzer.res_LCOD)
    test.append(test1)
print(test)

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