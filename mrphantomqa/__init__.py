"""
MRPhantomQA - A package for handling DICOM file scanning and fMRI quality assurance metrics calculation.

This package provides the following main modules:
- dicomFolderScanner: Module for scanning and processing DICOM files in a folder.
- gloverAnalyzer: Module for calculating quality assurance metrics on fMRI data based on the "Report on a Multicenter fMRI Quality Assurance Protocol" paper by Friedman and Glover.
- acrAnalyzer: Module for evaliuating QA scans performed on the ACR Large MRI Phantom using standard settings.

For more information, refer to the readme.md file.

Author: Kilian Jain
"""

import os

from .folder_scanner import dicomFolderScanner
from .glover_analyzer import gloverAnalyzer
from .acr_analyzer import acrAnalyzer
from .cylinderPhantom7t import cylinderAnalyzer


def askForPath():
    while True:
        path = str(input("Type the path of the desired DICOM directory: \n"))
        if os.path.exists(path):
            return path
        else:
            print("Path does not exist. Try anew.")
