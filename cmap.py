from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Greens', 128)

newcolors = np.vstack((top(np.linspace(0.5, 1.0, 128)),
                       bottom(np.linspace(0, 0.5, 128))))
OrGr = ListedColormap(newcolors, name='OrGr')