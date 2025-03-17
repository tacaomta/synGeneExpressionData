import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class GENEGAN:
    '''
    Class using GAN structure to synthesize gene expression data.
    '''

    def __init__(self, network) -> None:
        self.network = network

    def fit(self, epochs=1000, batch_size=8, lr=0.001, history_path=None):
        latent_dim = self.network.size*2
        self.generator = models.Sequential([
        layers.Dense(256, activation='relu', input_dim=latent_dim),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(latent_dim, activation='tanh')])
        #self.generator.add(layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.Constant(2)))

        self.discriminator = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=latent_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')])

        # loss
        bce_loss = tf.keras.losses.BinaryCrossentropy()

        # Optimizers
        optimizer_G = tf.keras.optimizers.Adam(lr)
        optimizer_D = tf.keras.optimizers.Adam(lr)

        data = self.network.to_latents()
        history={'epoch':[], 'd_loss':[], 'g_loss':[]}
        for epoch in range(epochs):
            d_loss = 0
            g_loss = 0
            for _ in range(len(data) // batch_size):
                # Train Discriminator
                real_data = data[np.random.choice(len(data), batch_size, replace=False)]
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))

                with tf.GradientTape() as tape:
                    # Train on real data
                    real_output = self.discriminator(real_data)
                    d_loss_real = bce_loss(real_labels, real_output)

                    # Train on fake data
                    z = tf.random.normal((batch_size, latent_dim))
                    fake_data = self.generator(z)
                    fake_output = self.discriminator(fake_data)
                    d_loss_fake = bce_loss(fake_labels, fake_output)

                    d_loss = d_loss_real + d_loss_fake

                grads_D = tape.gradient(d_loss, self.discriminator.trainable_variables)
                optimizer_D.apply_gradients(zip(grads_D, self.discriminator.trainable_variables))

                # Train Generator
                with tf.GradientTape() as tape:
                    z = tf.random.normal((batch_size, latent_dim))
                    fake_data = self.generator(z)
                    fake_output = self.discriminator(fake_data)
                    g_loss = bce_loss(real_labels, fake_output)

                grads_G = tape.gradient(g_loss, self.generator.trainable_variables)
                optimizer_G.apply_gradients(zip(grads_G, self.generator.trainable_variables))
            print(f'Epoch {epoch+1}/{epochs} | Discriminator Loss: {d_loss} | Generator Loss: {g_loss}')
            history['epoch'].append(epoch+1)
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)

        plt.plot(history['epoch'], history['d_loss'], color='red', label='d_loss')
        plt.plot(history['epoch'], history['g_loss'], color='blue', label='g_loss')

        # Add labels and legend
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('traing history')
        plt.legend()
        if history_path is not None:
            plt.savefig(history_path)

    def sample(self, number_sample=50, path=None):
        '''
        sample data
        '''
        z = tf.random.normal((number_sample, self.network.size*2))
        generated_samples = self.generator(z).numpy()

        # Denormalize the generated samples if needed
        generated_samples *= 2.0
        generated_samples = np.abs(np.round(generated_samples))
        generated_samples = np.clip(generated_samples, 0, 2)
        generated_samples = generated_samples.astype(int)
        print(generated_samples)
        syn_network = []
        for i in range(len(generated_samples)):
            syn_network.append(list(generated_samples[i])[:self.network.size])
            if i==len(generated_samples)-1:
                syn_network.append(list(generated_samples[i])[self.network.size:])            
        for i, row in enumerate(syn_network):
            row.insert(0, i)
        syn_network = np.array(syn_network)
        if path is not None:
            np.savetxt(path, syn_network, delimiter='\t', header='Time\t'+self.network.getHeader(), fmt='%s')
            print(f"The generated network is saved at: {path}")


