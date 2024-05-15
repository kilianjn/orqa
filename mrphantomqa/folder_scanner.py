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
        imagedata(numpy.ndarray): 4D Array (timestep, slice, imgY, imgX)

    Parameters:
        folder_path (str): The path to the folder containing DICOM files.

    Attributes:
        folder_path (str): The path to the folder containing DICOM files.
        all_sequences (list): A list of unique sequence names found in the DICOM files.
        all_dcmfiles (dict): A dictionary where keys are sequence names and values are lists of corresponding DICOM file paths.
        sel_sequence (str): The selected sequence for analysis.
        sel_dcmfiles (dict): A dictionary where keys are acquisition numbers and values are lists of DICOM file paths for the selected sequence.
        timeseries (numpy.ndarray): A 4D numpy array representing a time series of 3D volumes.
        volume (numpy.ndarray): A 3D numpy array representing a volume for a specific acquisition number in the selected sequence.

    Methods:
        list_scans(): Print a list of different scans in the folder.
        choose_scan(listkey): Choose a scan based on the provided key.
        image_properties(): Display information about the selected sequence, such as the number of slices, images in the time series, and dimensions.
        view_raw_image: View a raw DICOM image.

    Private Methods:
        _scan_folder(): Private method to scan the folder and organize DICOM files based on series and acquisition numbers.
        _create_volume(aquNum): Private method to create a 3D volume for a specific acquisition number.
    """

    def __init__(self, folder_path, rescan = False, **kwargs):
        print("")
        self.folder_path = folder_path
        self.rescan = rescan

        self.all_sequences = None # sequence name keys
        self.all_dcmfiles = None # filenames corresponding to sequences
        self._scan_folder()

        self.sel_sequence = None # key to chosen sequence
        self.sel_dcmfiles = None # list with files corresponding to chosen sequence
        self.metadata = None

        self.imagedata = None

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
        
    def list_scans(self):
        print("\nList of different scans in the folder:")
        for scan in [x for x in self.all_sequences if not "PhoenixZIPReport" in x]:
            print(scan)
        print("")

    def choose_scan(self, listkey):
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

        # self.pixeldims = [float(i) for i in self.get_dtag(0x00280030, [1,1,1])]


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
        print(f"Image Information for {self.sel_sequence}:\nSlices: {self.get_dtag(0x00280008, 'N/A')} \nImages in timeseries: {self.get_dtag(0x00200105, 'N/A')} \nDimensions: x={self.get_dtag(0x00280010, 'N/A')}, y={self.get_dtag(0x00280011, 'N/A')}\n\n")

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