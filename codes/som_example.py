import fitsio
from astropy.io import fits
import numpy as np
from minisom import MiniSom

kids_goldsample = fitsio.FITS('/net/home/fohlen13/yanza21/DATA/KiDS_data/KiDS_gold_smallsample.fits')

train_data = np.array(kids_goldsample[1]['MU_THRESHOLD',
                                         'FWHM_WORLD', 
                                         'PSF_e1',
                                         'PSF_e2', 
                                         'MAG_LIM_u', 
                                         'MAG_LIM_g',
                                         'MAG_LIM_r',
                                         'MAG_LIM_i',
                                         'MAG_LIM_Z',
                                         'MAG_LIM_Y',
                                         'MAG_LIM_J',
                                         'MAG_LIM_H',
                                         'MAG_LIM_Ks', 
                                         'EXTINCTION_r'][:].tolist())
q = train_data.T[2] / train_data.T[3]
train_data.T[2] = 1-q
train_data = np.delete(train_data.T, 3, axis=0).T
invalid_ind = np.where(np.logical_or(np.abs(train_data)>100, np.isnan(train_data)))[0]

train_data = np.delete(train_data, invalid_ind, axis=0)
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)

som_dim = int((5*(train_data.shape[0])**0.5)**0.5)
som = MiniSom(som_dim, som_dim, train_data.shape[1], sigma=1, learning_rate=1, 
              neighborhood_function='gaussian', topology='hexagonal', random_seed=0) # initialization of 6x6 SOM
#som.random_weights_init(train_data)
som.pca_weights_init(train_data)
som.train(train_data, 100000000, random_order=True, verbose=True)

import pickle
with open('som_sample.p', 'wb') as outfile:
    pickle.dump(som, outfile)