import fitsio
from astropy.io import fits
import numpy as np
from minisom import MiniSom
import matplotlib as mpl
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 200

import numba
from numba import int32, float32    # import the types

from numba import jit
from numba.experimental import jitclass


class cat2som:
    
    '''
    This is a class that train and summarize a self-organized map from data.
    Input:
    training_data: an NxM dimensional ndarray of the training data vector. N is the 
        number of sources; M is the number of columns to be concerned.
    colnames: a list of strings specifying the names corresponding to each dimension 
        of the training vector.
    **kwargs: arguments that goes into a MiniSom class.
    '''
    def __init__(self, training_data, colnames, **kwargs):
        self.training_data = training_data
        self.colnames = colnames
        self.som_dim = int((5*(training_data.shape[0])**0.5)**0.5)
        self.som = MiniSom(self.som_dim, self.som_dim, training_data.shape[1], 
                           **kwargs) # initialization of 6x6 SOM
        #som.random_weights_init(training_data)
        self.som.pca_weights_init(training_data)
        
    def train_som(self, n_iter):
        '''
        This function calls the 'train' function in a MiniSom object.
        Input:
        n_iter: an integer. The number of training epochs.
        '''
        self.som.train(self.training_data, n_iter)
        
    def get_activation_map(self, testing_data):
        '''
        This function calls the 'activation_response' function in a
        MiniSom object to return the number of vectors mapped into cells
        of a pre-trained SOM.
        Input:
        testing_data: an NxM dimensional ndarray of the testing data vector. N is the 
        number of sources; M is the number of columns to be concerned.
        '''
        return self.som.activation_response(testing_data) 
        
    def som_ind_to_1d(self, xi, yi):
        '''
        This function converts the 2-D indices of a SOM into a 1-D index of the flattend
        SOM. 
        Input:
        xi, yi: integers of the indices of a SOm
        '''
        return xi * self.som_dim + yi
        
    def get_winner_ind(self, testing_data):
        '''
        This function returns the 2-D indices and 1-D index of each data vector in the 
        testing data.
        Input:
        testing_data: an NxM dimensional ndarray of the testing data vector. N is the 
        number of sources; M is the number of columns to be concerned.
        Output:
        winner_x, winner_y: the x and y indices of the winning cell corresponding to 
        all the data vector.
        winner_1dind: the 1-D indices of the winning cell on the flattened SOM corresponding 
        to all the data vector.
        '''
        winner_x = np.zeros(testing_data.shape[0])
        winner_y = np.zeros(testing_data.shape[0])
        winner_1dind = np.zeros(testing_data.shape[0])

        for i in tqdm(range(len(testing_data))):
            data = testing_data[i]
            x,y = self.som.winner(data)
            winner_x[i] = x
            winner_y[i] = y
            winner_1dind[i] = self.som_ind_to_1d(x, y)
            
        return winner_x, winner_y, winner_1dind
    
    def get_feature_map(self, testing_data, col_ind, x=None, y=None, activation_map=None):
        '''
        This function returns the 2-D feature map of the testing data. The feature map is 
        defined as the mean value of one of the entry in the testing data in each of the 
        SOM cell.
        Input:
        testing_data: an NxM dimensional ndarray of the testing data vector. N is the 
            number of sources; M is the number of columns to be concerned.
        col_ind: the index of the data vector to construct the feature map.
        x, y: the pre-calculated x and y indices of the SOM. If not provided, then call 
            self.get_winner_ind to calculate them;
        activation_map: the activation_map corresponding to the testing data. If not provided,
            then call self.get_activation_map to calculate it.
        Output: 
        feqture_map_dict: a dictionary contains the columne name corresponding to the feature
            and the feature map.
        '''
        if x is None or y is None:
            x, y, winner_1dind = self.get_winner_ind(testing_data)
        feature_map = np.zeros((self.som_dim, self.som_dim))
        for i in range(len(testing_data)):
            data = testing_data[i]
            feature_map[x[i],y[i]] += data[col_ind]
        if activation_map is None:
            activation_map = self.get_activation_map(testing_data)
            
        feature_map = feature_map / activation_map
        feature_map[np.where(np.abs(feature_map==np.inf))] == np.nan
        feature_map_dict = {'colname': self.colnames[col_ind],
                            'feature_map': feature_map/activation_map}
        return feature_map_dict                 