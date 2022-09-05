import fitsio
from astropy.io import fits
import numpy as np
from minisom import MiniSom
import matplotlib as mpl
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 200

class cat2som:
    def __init__(self, training_data, colnames, **kwargs):
        self.training_data = training_data
        self.colnames = colnames
        self.som_dim = int((5*(training_data.shape[0])**0.5)**0.5)
        self.som = MiniSom(self.som_dim, self.som_dim, training_data.shape[1], 
                           **kwargs) # initialization of 6x6 SOM
        #som.random_weights_init(training_data)
        self.som.pca_weights_init(training_data)
        
    def train_som(self, n_iter):
        self.som.train(self.training_data, n_iter)
        
    def get_activation_map(self, testing_data):
        return self.som.activation_response(testing_data) 
        
    def som_ind_to_1d(self, xi, yi):
        return xi * self.som_dim + yi
        
    def get_winner_ind(self, testing_data):
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