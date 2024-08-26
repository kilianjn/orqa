import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np

def differenceViewer(differenceImage, title=""):
    """Gives graph of difference. Requires the substract of two datasets. Colormap is blue for positive values and red for negative values.
    
    Returns plt.imshow
    """
    # Farbverlauf definieren
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Rot, Weiß, Grün
    n_bins = 100  # Anzahl der Farbstufen
    cmap_name = 'custom_colormap'

    # Colormap erstellen
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Normalisierung erstellen (weißer Punkt bei Null)
    vmin, vmax = differenceImage.min(), differenceImage.max()

    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    plt.imshow(differenceImage, cmap=custom_cmap, norm=norm)
    plt.title(f"{title}")
    plt.colorbar() 
    plt.gca().set_aspect('auto')


def plot3D(data, title="3D Surface Plot", cmap='viridis'):
    """3D viewer for 2D array"""
    if len(data.shape) != 2:
        print(f"Data has {len(data.shape)} dimensions instead of the required 2")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
    surf = ax.plot_surface(x, y, data, cmap=cmap,rstride=1,cstride=1)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    ax.set_xlabel('X-values')
    ax.set_ylabel('Y-values')
    ax.set_zlabel('Z-values')
    ax.set_title(title)

    plt.show()

def plot2D(data, title="3D Surface Plot", cmap='gray'):
    """2D viewer for 2D array"""
    if len(data.shape) != 2:
        print(f"Data has {len(data.shape)} dimensions instead of the required 2")
        return
    
    plt.imshow(data, cmap=cmap)
    plt.xlabel("Left to right")
    plt.ylabel("Posterior to anterior")
    # plt.axis("off")
    plt.show()