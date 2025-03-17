import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from gene import GeneNetwork
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt



class Vt_prediction:
    def __init__(self, network) -> None:
          self.network = network
          self.v_t, self.v_t_plus_1 = self.network.get_vt_vt_plus_1()
        
    def fit(self, lr=0.001, epochs=1000, model_save=None, loss_figure=None):
        # Define your data
        # v_t = np.array([[0, 1, 0, 2, 1, 1, 0, 0]], dtype=np.float32)
        # v_t_plus_1 = np.array([[2, 2, 0, 1, 0, 0, 1, 1]], dtype=np.float32)
        # Define the autoencoder architecture
        input_dim = self.v_t.shape[1]
        #latent_dim = 4 if si<=20 else 8 if si<=30 else 32 if si<=60 else 64 if si<=100 else 128
        latent_dim = self.network.size*2

        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(latent_dim, activation='relu')(input_layer)
        encoded = Dense(512, activation='relu')(encoded)

        # Decoder
        decoded = Dense(latent_dim, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)

        # Autoencoder
        self.autoencoder = Model(input_layer, decoded)
        optimizer = Adam(lr=lr)

        # Compile the autoencoder
        self.autoencoder.compile(optimizer='adam', loss='mse')
        #epochs = 1000 if si==10 else 5000 if si==50 else 10000
        # Train the autoencoder
        #autoencoder.fit(v_t[:80], v_t_plus_1[:80], epochs=10000, batch_size=1)
        early_stopping_cb = EarlyStopping(patience=2000, restore_best_weights=True)
        history = self.autoencoder.fit(self.v_t, self.v_t_plus_1, epochs=epochs, batch_size=1, 
                                  callbacks=[early_stopping_cb])
        pd.DataFrame(history.history).plot(figsize=(8, 5)) 
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        if loss_figure is None:
             plt.show()
        else:
            plt.savefig(loss_figure)
        if model_save is None:
             pass
        else:
            save_model(self.autoencoder, model_save)
        
    def predict(self, vt):
         return self.autoencoder.predict(vt)
