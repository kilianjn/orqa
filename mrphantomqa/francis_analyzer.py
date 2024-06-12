from .utils.methods import functions

import matplotlib.pyplot as plt
import numpy as np

class francisAnalyzer:
    def __init__(self, data) -> None:
        # self.imagedata_loc = data.imagedata[0]
        self.imagedata_loc = data.imagedata
        self.metadata = data.metadata if hasattr(data, 'metadata') else None

        self.res_GA = None

    def geometric_accuracy(self, showplot=False, print=False):
        
        pass