import fitsio
from astropy.io import fits
import numpy as np
from minisom import MiniSom
import matplotlib as mpl
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import RegularPolygon, Ellipse
from matplotlib import cm, colorbar

mpl.rcParams['figure.dpi'] = 200
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_som(ax, som_heatmap, topology='rectangular', colormap=cm.viridis, cbar_name=None):
    if topology == 'rectangular':
        ax.matshow(som_heatmap.T, cmap=colormap)
    else:
        cscale = som_heatmap / som_heatmap[~np.isnan(som_heatmap)].max()
        som_dim = cscale.shape[0]
        yy, xx= np.meshgrid(np.arange(som_dim), np.arange(som_dim))
        shift = np.zeros(som_dim)
        shift[::2]=-0.5
        xx = xx + shift
        for i in range(cscale.shape[0]):
            for j in range(cscale.shape[1]):
                wy = yy[(i, j)] * np.sqrt(3) / 2
                if (cscale[i,j] == 0 or (np.isnan(cscale[i,j]))):
                    color = 'k'
                else:
                    color = colormap(cscale[i,j])
            
                hex = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius= 1 / np.sqrt(3),
                                 facecolor=color, 
                                 edgecolor=color,
                                 #alpha=.4, 
                                 lw=0,)
                ax.add_patch(hex)

        scmap = plt.scatter([0,0],[0,0], s=0, c=[som_heatmap[~np.isnan(som_heatmap)].min(),
                                                 som_heatmap[~np.isnan(som_heatmap)].max()], 
                            cmap=colormap)
        ax.set_xlim(-1,som_dim-.5)
        ax.set_ylim(-0.5,som_dim * np.sqrt(3) / 2)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        plt.colorbar(scmap, cax=cax, label=cbar_name)