from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from ..utils.methods import functions as utilFunc

class functions:
    @staticmethod
    def roi(imagedata, size=21, slicenumber = None):
        assert imagedata is not None,"Imagedata provided does not exist"
        assert len(imagedata.shape) == 3 or 2,"Wrong dimensions."
        if len(imagedata.shape) == 3:
            num_slices, height, width = imagedata.shape
        elif len(imagedata.shape) == 2:
            height, width = imagedata.shape
            num_slices = 1  # Set number of slices to 1 for 2D data

        center_x, center_y = width // 2, height // 2
        half_size = size // 2

        if size % 2 == 0:
            start_x = center_x - half_size
            end_x = center_x + half_size
            start_y = center_y - half_size
            end_y = center_y + half_size
        else:
            start_x = center_x - half_size
            end_x = center_x + half_size + 1
            start_y = center_y - half_size
            end_y = center_y + half_size + 1

        start_x = max(0, start_x)
        end_x = min(width, end_x)
        start_y = max(0, start_y)
        end_y = min(height, end_y)

        # if len(imagedata.shape) == 3:
        #     cropped_data = imagedata[:, start_y:end_y, start_x:end_x]
        # else:  # 2D data
        #     cropped_data = imagedata[start_y:end_y, start_x:end_x]

        mask = np.zeros_like(imagedata, dtype=bool)
        if len(imagedata.shape) == 3:
            mask[:, start_y:end_y, start_x:end_x] = True
        else:  # 2D data
            mask[start_y:end_y, start_x:end_x] = True

        return np.ma.masked_array(imagedata,~mask)

    @staticmethod
    def detrend_image(imagedata):
        assert len(imagedata.shape) == 3,"3D array needed for detrending of timeseries."

        num_slices, height, width = imagedata.shape

        # Initialize an array to store the detrended data
        detrended_data = np.empty(imagedata.shape)

        # Create a tqdm progress bar
        progress_bar = tqdm(total=width * height, desc="Detrending Pixels")

        # Iterate over all pixels
        for x in range(width):
            for y in range(height):
                pixel_timeseries = imagedata[:, x, y]

                # Call the quad_fit method to fit the quadratic function
                fitted_curve = functions.quad_fit(pixel_timeseries)

                # Subtract the fitted curve from the original data
                detrended_data[:, x, y] = pixel_timeseries - fitted_curve

                # Update the progress bar
                progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        return detrended_data

    @staticmethod
    def quad_fit(y):
        assert len(y.shape) == 1,"Input must be 1D"
        num_slices = len(y)
        x = np.arange(num_slices)  # Create an array of time indices
        def quadratic_function(x, a, b, c):
            return a * x**2 + b * x + c

        popt, _ = curve_fit(quadratic_function, x, y)
        fitted_curve = quadratic_function(x, *popt)

        return fitted_curve

    @staticmethod
    def addPixelwiseSums(imagedata):
        if imagedata is None:
            print("No time series data available")
            return
        
        if imagedata.shape[0] %2 == 0: # Check for uneven amount of timesteps and pop the first if so.
            imagedata = imagedata[1:]

        timesteps, height, width = imagedata.shape

        # Initialize arrays to store the sums for even and odd slices
        sum_even_images = np.zeros((height, width))
        sum_odd_images = np.zeros((height, width))

        # Iterate through the time series data
        for i in range(timesteps):
            if i % 2 == 0:  # Even numbered slice
                sum_even_images += imagedata[i, :, :]
            else:  # Odd numbered slice
                sum_odd_images += imagedata[i, :, :]

        return sum_even_images, sum_odd_images

    @staticmethod
    def summaryValue(imagedata, size=21):
        assert len(imagedata.shape) ==  2,"summaryValue only takes 2D input"
        imgROI = functions.roi(imagedata, size)
        sv = np.mean(imgROI)
        return sv
    
    @staticmethod
    def summaryValue_over_time(imagedata, size=21):
        assert len(imagedata.shape) == 3,"summaryValue_over_time needs 3D input"
        testimage = np.empty(imagedata.shape[0])
        for i in range(imagedata.shape[0]):
            testimage[i] = functions.summaryValue(imagedata[i,:,:], size)
        return testimage

class MiscFunctions(functions):
    def debug_skewResDataSinus(self, resData, amplitude=0.5, freqOverROI=10):
        sinus = np.empty(resData.shape[0])
        skewedData = np.empty(resData.shape[0])
        totaltime = resData.shape[0]
        print(totaltime)
        for timestep in range(totaltime):
            sinus[timestep] = amplitude * np.sin((2 * np.pi)*(freqOverROI/totaltime) * timestep)
            print(f"{np.round(sinus[timestep],2)} with {freqOverROI} at time {timestep}")
        skewedData = resData + sinus
        return skewedData

    def debug_detrend_image(self, imagedata, x, y):
        if imagedata is None:
            print("No time series data available")
            return
        
        if len(imagedata.shape) == 1:
            imagedata = imagedata[:,np.newaxis,np.newaxis]

        pixel_timeseries = imagedata[:, x, y]

        # Call the quad_fit method to fit the quadratic function
        fitted_curve = self.quad_fit(pixel_timeseries)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 1)
        
        # Plot the original data in the first subplot
        axes.plot(pixel_timeseries, label='Original Data')
        axes.plot(fitted_curve, label='Fitted Quadratic Curve', linestyle='--')
        axes.set_xlabel('Time')
        axes.set_ylabel('Pixel Value')
        axes.set_title(f'Debug Plot of Pixel ({x}, {y})')
        axes.legend()
        axes.grid(True)

        plt.show()

        return
