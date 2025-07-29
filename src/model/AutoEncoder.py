import keras
import tensorflow as tf
import numpy as np
from .components import MLP

class AutoEncoder(keras.Model):
    def __init__(self, input_size : int = 32, latent_dim : int = 8, num_layers : int = 3, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.nodes = num_layers
        self.encoder = MLP(latent_dim, num_layers, activation='relu', name='encoder')
        self.decoder = MLP(input_size, num_layers, activation='linear', name='decoder')


    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded