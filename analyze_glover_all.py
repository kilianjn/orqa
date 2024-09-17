import mrphantomqa
import os, shutil

paths = [
    "/Volumes/KINGSTON/glover7T/05082024",
    "/Volumes/KINGSTON/glover7T/13082024", # TR 1000
    "/Volumes/KINGSTON/glover7T/22082024",
    "/Volumes/KINGSTON/glover7T/23082024",
    "/Volumes/KINGSTON/glover7T/08282024",
    "/Volumes/KINGSTON/glover7T/09092024",
    "/Volumes/KINGSTON/glover7T/12092024",
]
# paths = ["/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code/Data/23102614"] # old 3T data

workdir = "/Users/rameshjain/Documents/Studium/M. Sc. Masteruppsats/Code"

#################DANGER!!!!!!!
deletion_path = os.path.join(workdir,"glover")
if os.path.exists(deletion_path):
    shutil.rmtree(deletion_path)
##################DANGER!!!!!!

for path in paths:

    dfs = mrphantomqa.dicomFolderScanner(path)
    dfs.choose_scan_via_menu(True)
    # dfs.list_scans()
    # dfs.choose_scan("155202.620000 ep2d_se")
    # dfs.sequence_properties()
    dfs.get_data()
    # dfs.view_raw_image()


    gm = mrphantomqa.gloverAnalyzer(dfs,workdir,5,0.15)
    gm.runall()

    del dfs, gm
