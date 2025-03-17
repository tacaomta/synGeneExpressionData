from keras import layers
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model, save_model


class GVAE:
    '''
    Variational autoencoder is designed for gene expression data
    '''

    def __init__(self, network) -> None:
        self.network = network

    def fit(self, epochs=1000, batch_size=32, verbose=1, model_save=None):
        data = self.network.to_latents()
        max_value = np.argmax(data)
        min_value = np.argmin(data)
        input_layer = layers.Input(shape=(self.network.size*2,))
        encoded = layers.Dense(256, activation='relu')(input_layer)
        decoded = layers.Dense(self.network.size*2, activation='sigmoid')(encoded)
        decoded = decoded * (max_value - min_value) + min_value
        # Create the autoencoder model
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=verbose)
        if model_save is None:
            save_model(autoencoder, fr'models/ae_model_{self.network.size}_{self.network.filename}.h5')
        else:
            save_model(autoencoder, model_save)
    
    def sample(self, path, rows=None):
        '''
        Hàm sinh ra dữ liệu giả, khi tham số rows không được cung cấp thì sẽ sinh ra mạng
        có cùng số timestep như mạng huấn luyện
        '''
        autoencoder = load_model(fr'models/ae_model_{self.network.size}_{self.network.filename}.h5')
        if rows is None:            
            synthetic_data = autoencoder.predict(self.network.to_latents())
            synthetic_data = np.round(synthetic_data)
            synthetic_data = synthetic_data.astype(int)
            
        else:
            generated_latent_points = np.random.normal(size=(rows, self.network.size*2))
            synthetic_data = autoencoder.predict(generated_latent_points)
            synthetic_data = np.abs(np.round(synthetic_data))
            synthetic_data = np.clip(synthetic_data, 0, 2)
            synthetic_data = synthetic_data.astype(int)
        syn_network = []
        for i in range(len(synthetic_data)):
            syn_network.append(list(synthetic_data[i])[:self.network.size])
            if i==len(synthetic_data)-1:
                syn_network.append(list(synthetic_data[i])[self.network.size:])            
        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        np.savetxt(path, syn_network, delimiter='\t', header='Time\t'+self.network.getHeader(), fmt='%s')
        
