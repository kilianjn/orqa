import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pydicom
from tqdm import tqdm

import json
import os

class dicomFolderScanner:
    """
    A class for scanning and processing DICOM files in a folder.

    Main Output:
        imagedata (numpy.ndarray): 4D Array (timestep, slice, imgY, imgX)

    Parameters:
        folder_path (str): The path to the folder containing DICOM files.
        rescan (bool): Whether to force a rescan of the folder, even if a previously saved scan exists.

    Attributes:
        folder_path (str): The path to the folder containing DICOM files.
        all_sequences (list): A list of unique sequence names found in the DICOM files.
        all_dcmfiles (dict): A dictionary where keys are sequence names and values are lists of corresponding DICOM file paths.
        sel_sequence (str): The selected sequence for analysis.
        sel_dcmfiles (dict): A dictionary where keys are acquisition numbers and values are lists of DICOM file paths for the selected sequence.
        metadata (dict): A dictionary containing metadata for the selected sequence.
        imagedata (numpy.ndarray): A 4D numpy array containing the image data for the selected sequence.

    Methods:
        list_scans(): Prints a list of different scans (sequences) available in the folder.
        choose_scan(listkey:str): Chooses a scan based on the provided key from the list of sequences.
        image_properties(): Displays information about the selected sequence, such as the number of slices, images in the time series, and dimensions.
        view_raw_image(): Views a raw DICOM image from the selected sequence.
        choose_scan_via_menu(autoMode:bool): Launches menu where the user can interactively choose a given scan. If automode is True the latest scan is automatically chosen.

    Private Methods:
        _askForPath(): Prompts the user to input a valid folder path if the provided path is invalid or not supplied.
        _scan_folder(): Scans the folder and organizes DICOM files based on sequence names. If a saved scan exists, it will load it unless rescan is set to True.
    """

    def __init__(self, folder_path:str=None, rescan = False, **kwargs):
        print("")
        self.folder_path = self._askForPath(folder_path)
        self.rescan = rescan

        self.all_sequences = None # sequence name keys
        self.all_dcmfiles = None # filenames corresponding to sequences
        self._scan_folder()

        self.sel_sequence = None # key to chosen sequence
        self.sel_dcmfiles = None # list with files corresponding to chosen sequence
        self.metadata = None

        self.imagedata = None

    def _askForPath(self, path):
        while True:
            if os.path.exists(str(path)):
                return path
            path = str(input(f"The given path does not exist({path})\nType the path of the desired DICOM directory: \n"))


    def _scan_folder(self):
        dicom_files = {}
        if not self.rescan and os.path.isfile(os.path.join(self.folder_path, "dicomDirectory.txt")): # Look for dirFile
            dicom_files = json.load(open(os.path.join(self.folder_path, "dicomDirectory.txt")))
            self.all_sequences = list(dicom_files.keys())
            self.all_dcmfiles = dicom_files
            print("Dicom Directory found and loaded in...")
            return

        for root, _, files in tqdm(os.walk(self.folder_path),"Scanning folders..."):
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    dcm = pydicom.dcmread(filepath)
                    if hasattr(dcm, "SeriesDescription"):
                        # series_uid = f"{dcm.SeriesDescription}"
                        series_uid = f"{dcm.SeriesTime} {dcm.SeriesDescription}"
                        if series_uid not in dicom_files:
                            dicom_files[series_uid] = []
                        dicom_files[series_uid].append(filepath)
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        print("Writing dirFile...")
        json.dump(dicom_files, open(os.path.join(self.folder_path, "dicomDirectory.txt"),'w'))
        
        self.all_sequences = list(dicom_files.keys())
        self.all_dcmfiles = dicom_files
        
    def choose_scan_via_menu(self, autoMode:bool=False):
        while True:
            sequences = [x for x in self.all_sequences if not "PhoenixZIPReport" in x]
            if autoMode:
                self.choose_scan(sequences[len(sequences)-1])
                break
            
            print("\nPlease select a sequence by entering the corresponding number:")
            for idx, sequence in enumerate(sequences):
                print(f"{idx + 1}. {sequence}")

            try:
                choice = int(input("Enter the number of the sequence you want to choose: ")) - 1
                if 0 <= choice < len(sequences):
                    self.choose_scan(sequences[choice])
                    break
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def list_scans(self):
        print("\nList of different scans in the folder:")
        for scan in [x for x in self.all_sequences if not "PhoenixZIPReport" in x]:
            print(scan)
        print("")

    def choose_scan(self, listkey:str):
        assert listkey in self.all_dcmfiles,"Key is not in Dictionary. Choose one from list_scans."
        self.sel_sequence = listkey

        if not self.rescan and os.path.isfile(os.path.join(self.folder_path, f"{listkey}.txt")): # Look for sortedDirfile
            print("Sorted Sequence Directory found. Loading in...")
            self.sel_dcmfiles = json.load(open(os.path.join(self.folder_path, f"{listkey}.txt")))

            timesteps_list = list(self.sel_dcmfiles.keys())
            timesteps_list.sort(key=int)
            self.metadata = pydicom.dcmread(self.sel_dcmfiles[timesteps_list[0]][0])
            return

        dicom_files = {}
        for filename in tqdm(self.all_dcmfiles[listkey],"Creating timesteps..."):
            try:
                dcm = pydicom.dcmread(filename)
                if hasattr(dcm, "AcquisitionNumber"):
                    acquNum = f"{dcm.AcquisitionNumber}"
                    # series_uid = f"{dcm.SeriesDescription} {dcm.SeriesInstanceUID}"
                    if acquNum not in dicom_files:
                        dicom_files[acquNum] = []
                    dicom_files[acquNum].append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")

        for key, value in tqdm(dicom_files.items(),"Sorting timesteps..."):
            value.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
        self.sel_dcmfiles = dicom_files

        timesteps_list = list(self.sel_dcmfiles.keys())
        timesteps_list.sort(key=int)
        self.metadata = pydicom.dcmread(self.sel_dcmfiles[timesteps_list[0]][0])
        
        print("Writing dir for current chosen scan...")
        json.dump(dicom_files, open(os.path.join(self.folder_path, f"{listkey}.txt"),'w'))

    def get_dtag(self, tag, default="N/A"):
        if self.sel_dcmfiles is None:
            print("No Dicom files selected.")
            return
        ds = pydicom.dcmread(self.sel_dcmfiles[sorted(list(self.sel_dcmfiles.keys()))[0]][0]) # Sort keys and use the smallest acquisition number
        if tag in ds:
            tag = ds[tag].value
        else:
            tag = default
        return tag

    def sequence_properties(self):
        print(f"Image Information for {self.sel_sequence}:\n"
              f"Slices: {self.get_dtag(0x00280008, 'N/A')} \n"
              f"Images in timeseries: {self.get_dtag(0x00200105, 'N/A')}\n"
              f"Dimensions: x={self.get_dtag(0x00280010, 'N/A')}, y={self.get_dtag(0x00280011, 'N/A')}\n\n")

    def _create_volume(self, aquNum="1"):
        volume = []

        for file in self.sel_dcmfiles[str(aquNum)]:
            ds = pydicom.dcmread(file)
            volume.append(ds.pixel_array)
        
        volume = np.array(volume)
        if len(volume.shape) == 4 and volume.shape[0] == 1:
            return volume[0]
        elif len(volume.shape) == 3:
            return volume

        return volume

    def get_data(self):
        timesteps_list = list(self.sel_dcmfiles.keys())
        timesteps_list.sort(key=int)
        time_series_images = []
        for timestep in tqdm(timesteps_list, "Generating dataarray..."):
            time_series_images.append(self._create_volume(timestep))
        # self.timeseries = np.stack(time_series_images, axis=0)
        self.imagedata = np.array(time_series_images)
        print(f"Dimensions of Array: {self.imagedata.shape}")

    def view_raw_image(self):
        if self.imagedata is None:
            print("No volume available to show.")
            return 
        else:
            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.25)

            img = ax.imshow(self.imagedata[0, 0], cmap='viridis')

            ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            time_slider = Slider(ax_time_slider, 'Timestep', 0, self.imagedata.shape[0] - 1, valinit=0, valstep=1)

            ax_slice_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
            slice_slider = Slider(ax_slice_slider, 'Slice', 0, self.imagedata.shape[1] - 1, valinit=0, valstep=1)

            def update(val):
                time_step = int(time_slider.val)
                slice_step = int(slice_slider.val)
                img.set_data(self.imagedata[time_step, slice_step])
                fig.canvas.draw_idle()

            time_slider.on_changed(update)
            slice_slider.on_changed(update)

            plt.show()
            return