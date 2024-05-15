from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit

from .utils.methods import functions  as utfc

class cylinderAnalyzer:
    def __init__(self, folderscanner) -> None:
        self.imagedata = folderscanner.imagedata
        self.all_metrics = None

        self.folder_path = folderscanner.folder_path
        self.sequence_name = folderscanner.sel_sequence

        # self.backupfileName = os.path.join(folderscanner.folder_path, f"metrics_{folderscanner.sel_sequence}.npy")

    @property
    def backupfileName(self):
        return os.path.join(self.folder_path, f"metrics_{self.sequence_name}.npy")

    def saveMetrics(self):
        if self.all_metrics is None:
            print("No Metrics have been calculated.")
            return
        with open(self.backupfileName, 'wb') as f:
            np.save(f, self.all_metrics)
            print("Mertrics have been saved.")

    def loadMetrics(self):
        if not os.path.isfile(self.backupfileName):
            print("No backupfile found. continuing to create metrics...")
            self.getMetrics()
            self.saveMetrics()
            self.loadMetrics()
            return
        
        with open(self.backupfileName, 'rb') as f:
            self.all_metrics = np.load(f)

    def getMetrics(self):
        metrics = np.empty((4, self.imagedata.shape[0], self.imagedata.shape[1])) 
        """
        (metric, time, slice)
        metric: centerpointY, centerpointX, size, circleness
        """

        for j in tqdm(range(self.imagedata.shape[0]),"Getting Metrics"):
        # for j in [0,1,2,3,4,5,6,7,8,9]:
            for i in range(self.imagedata.shape[1]):
                thld = utfc.getThreshold.otsuMethod(self.imagedata[j,i,:,:])
                thldimg = utfc.createThresholdImage(self.imagedata[j,i,:,:], thld)
                filledthldImg = utfc.fillShape(thldimg)
                metrics[0,j,i], metrics[1,j,i] =  utfc.findCenter.centerOfMass(filledthldImg)
                metrics[2,j,i] = np.sum(filledthldImg)

                radius = np.sqrt(metrics[2,j,i]/np.pi)
                circCutout = np.ma.masked_array(filledthldImg, utfc.circularROI(filledthldImg, (metrics[0,j,i], metrics[1,j,i]), radius))
                metrics[3,j,i] = np.sum(circCutout) / metrics[2,j,i]
        self.all_metrics = metrics

