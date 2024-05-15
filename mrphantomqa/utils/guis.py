import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def questionWindow(imagedata, question=None, **kwargs):
    window = tk.Tk()
    window.title('Plotting in Tkinter')
    window.geometry("500x500")

    lhs = ttk.Frame(window, width=500)
    lhs.pack(expand=True, fill="both", side="left")
    lhs.pack_propagate(0)

    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.imshow(imagedata, cmap="jet", **kwargs)
    canvas = FigureCanvasTkAgg(fig, master=lhs)
    canvas.draw()
    canvas.get_tk_widget().pack()

    window.mainloop()
    del window
    return 

if __name__ == "__main__":
    # Beispielaufruf:
    imagedata_example = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # plt.imshow(imagedata_example)
    questionWindow(imagedata_example, "Choose a number and colormap:")