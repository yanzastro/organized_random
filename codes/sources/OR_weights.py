import fitsio
from astropy.io import fits
import numpy as np
from minisom import MiniSom
import matplotlib as mpl
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 200


def calculate_or_weights(Ns, testing_radec, testing_som_ind):

    hp_ind = hp.ang2pix(Ns, testing_radec.T[0], testing_radec.T[1], lonlat=True)

    hp_ind_unique, n_p = np.unique(hp_ind, return_counts=True)  # this line gives the unique healpix pixel indices that contains at least one galaxy, as well as the number of galaxies in each pixel 
    A_p = hp.nside2pixarea(Ns) * hp_ind_unique.size  # the total area of the footprint

    som_1dind_unique = np.unique(testing_som_ind)  # the unique SOM pixel indices (in 1D)
    delta_i_sum = 0
    weight_map = np.zeros(hp.nside2npix(Ns))
    for som_ind in som_1dind_unique:
        source_ind = np.where(testing_som_ind==som_ind)[0]  # pick out the catalog indices of sources that are in the som_ind'th SOM pixel
        n_i = source_ind.size  # and the number of sources in that SOM pixel
        hp_ind_ = hp_ind[source_ind]  # and the Healpix pixel indices
        hp_ind_insomi_unique, n_p_i = np.unique(hp_ind_, return_counts=True)  # n_p_i is the number of sources that are in the i (som_ind here)-th SOM cell in each Healpix pixel
        A_p_i = n_p_i/n_p[np.intersect1d(hp_ind_unique, hp_ind_insomi_unique, return_indices=True)[1]] * A_p  # the effective pixel area for the i-th SOM cell
        A_i = np.sum(A_p_i)  # the total effective area for the i-th SOM cell
        delta_i = n_i / A_i  # effective number density 
        delta_i_sum += delta_i
        weight_map[hp_ind_insomi_unique] += delta_i * n_p_i  # weight of each Healpix pixel
    
    weight_map/=np.mean(delta_i_sum)
    weight_map[weight_map==0] = hp.UNSEEN
    return weight_map