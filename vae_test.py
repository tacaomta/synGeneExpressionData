import keras
from keras import layers
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from gene import GeneNetwork


network = GeneNetwork(fr"C:\caocao\gnw-master\tave_gen\hybrid\s10\sample1_network.txt", 
                      fr"C:\caocao\gnw-master\tave_gen\hybrid\s10\sample1_goldstandard.txt",None, False, 
                                  print_out=False)
data = network.to_latents()
#print(data)
data = np.array(data)
print('Input data')


# Generate some synthetic discrete data
num_timesteps = network.timestepsNumber
input_dim = network.size*2
max_value = 2
min_value = 0

# Define the autoencoder architecture
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(256, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)  # Use softmax for discrete output
decoded = decoded * (max_value - min_value) + min_value
# Create the autoencoder model
autoencoder = models.Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')  # Use categorical loss for discrete data

# Train the autoencoder
autoencoder.fit(data, data, epochs=1000, batch_size=32, verbose=1)

# Use the trained autoencoder to generate synthetic data
synthetic_data_probs = autoencoder.predict(data)
print('Input data =========')
print(data)
print("Example of synthetic data probalities:")
print(synthetic_data_probs)
# Convert probabilities to discrete values
synthetic_data = np.round(synthetic_data_probs)

# Example usage: print the first synthetic data point
print("Example of synthetic data point:")
print(synthetic_data)

max_ = np.argmax(data)
min_ = np.argmin(data)
print(max_, min_)

