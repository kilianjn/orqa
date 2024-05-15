import matplotlib.pyplot as plt

def histviewer(x, bins):
    plt.plot(bins[0:-1],x)
    plt.show()
    return