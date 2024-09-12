import mrphantomqa
import os, shutil

paths = [
    "/Volumes/KINGSTON/glover7T/05082024",
    "/Volumes/KINGSTON/glover7T/13082024",
    "/Volumes/KINGSTON/glover7T/22082024",
    "/Volumes/KINGSTON/glover7T/23082024",
    "/Volumes/KINGSTON/glover7T/08282024",
    "/Volumes/KINGSTON/glover7T/09092024",
]

workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code"

#################DANGER!!!!!!!
deletion_path = os.path.join(workdir,"glover")
if os.path.exists(deletion_path):
    shutil.rmtree(deletion_path)
##################DANGER!!!!!!

for path in paths:

    dfs = mrphantomqa.dicomFolderScanner(path)
    dfs.choose_scan_via_menu(True)
    dfs.list_scans()
    # dfs.choose_scan("123939.620000 t1_tse_tf1soSE_9sl_medFilter")
    # dfs.sequence_properties()
    dfs.get_data()
    # dfs.view_raw_image()


    gm = mrphantomqa.gloverAnalyzer(dfs,workdir,5,0.15)
    gm.runall()

    del dfs, gm
