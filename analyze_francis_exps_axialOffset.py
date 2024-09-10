import mrphantomqa
import os, shutil

filepath = "/Volumes/KINGSTON/francis_axial_offset/Phantom^Francis_2024.08.28-12_07_26-DST-1.3.12.2.1107.5.99.3_20000101/Neuro_Vuxen_20240828_120743.800000"

# DO NOT CHANGE
WORKDIR = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/thesiswork/axialOffset"
if os.path.exists(os.path.join(WORKDIR,"francis_reports")):
    shutil.rmtree(os.path.join(WORKDIR,"francis_reports"))
# DANGERRRRRRRRRRRR

sequences = [
    "124427.981000 t1_offset_F_-5",
    "124212.162000 t1_offset_F_-4",
    "123956.280000 t1_offset_F_-3",
    "123740.486000 t1_offset_F_-2",
    "123524.600000 t1_offset_F_-1",
    "122149.721000 t1_tse_tf1soSE_9sl_medFilter",
    "122405.479000 t1_offset_F_+1",
    "122621.358000 t1_offset_F_+2",
    "122837.199000 t1_offset_F_+3",
    "123052.971000 t1_offset_F_+4",
    "123308.858000 t1_offset_F_+5",
]
for sequence in sequences:
    
    dfs = mrphantomqa.dicomFolderScanner(filepath)
    # dfs.choose_scan_via_menu(True)
    # dfs.list_scans()
    dfs.choose_scan(sequence)
    # dfs.sequence_properties()
    dfs.get_data()
    # dfs.view_raw_image()

    Analyzer = mrphantomqa.francisAnalyzer(dfs, WORKDIR)
    Analyzer.runall()

    del dfs, Analyzer


