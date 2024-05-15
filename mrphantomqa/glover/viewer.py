import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def view_image(imagedata, plottitle= "CalculatedImage"):
    
    if imagedata is not None:
        plt.imshow(imagedata, cmap=plt.cm.bone)
        plt.title(f"{plottitle}")
        plt.colorbar()
        plt.show()

    else:
        print("Imagedata empty")
        return

def plot_pixel_over_time(imagedata, x=0, y=0, title=None):
    if imagedata is not None and len(imagedata.shape) >= 3:
        # pixel_values = [image[x, y] for image in imagedata]
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(imagedata[:,y,x])
        plt.xlabel('Time')
        plt.ylabel('Pixel Value')
        plt.title(f'Pixel Value Over Time at ({x}, {y})' if title is None else f"{title}")
        plt.legend()
        plt.grid(True)
        plt.subplot(122)
        plt.imshow(imagedata[0])
        plt.scatter(x,y)
        plt.colorbar()
        plt.show()
    elif imagedata is not None and len(imagedata.shape) == 1:
        # pixel_values = [image[x, y] for image in imagedata]
        plt.figure(figsize=(10, 5))
        plt.plot(imagedata)
        plt.xlabel('Time')
        plt.ylabel('Pixel Value')
        plt.title(f'Pixel Value Over Time' if title is None else f"{title}")
        plt.legend()
        plt.grid(True)

        plt.show()
    else:
        print("No time series data available")

def quadTrend_pixel(imagedata, x, y):
    # wrk_timeseries = self.timeseries.copy()
    if imagedata is None:
        print("No time series data available")
        return

    # Extract the time series data for the specified pixel
    pixel_timeseries = imagedata[:, x, y]

    # Create an array of time indices
    time_indices = np.arange(len(pixel_timeseries))
    
    def quadratic_function(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Fit the quadratic function to the time series data
    popt, _ = curve_fit(quadratic_function, time_indices, pixel_timeseries)

    # Generate the fitted quadratic curve
    fitted_curve = quadratic_function(time_indices, *popt)
    detrended_pixel = pixel_timeseries - fitted_curve

    # Plot the original data and the fitted curve
    plt.figure(figsize=(10, 5))
    plt.plot(time_indices, pixel_timeseries, label='Original Data')
    plt.plot(time_indices, fitted_curve, label='Fitted Quadratic Curve', linestyle='--')
    plt.plot(time_indices, detrended_pixel, label='det', linestyle='solid')
    plt.xlabel('Time')
    plt.ylabel('Pixel Value')
    plt.title(f'Quadratic Fitting for Pixel ({x}, {y})')
    plt.legend()
    plt.grid(True)
    plt.show()

    return
