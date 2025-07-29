import keras
import tensorflow as tf
import numpy as np
from .components import MLP

class VarAutoEncoder(keras.Model):
    def __init__(self, input_size, latent_dim, nodes: int | list[int]  = 3 , **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.nodes = nodes
        if isinstance(nodes, int):
            num_layers = nodes
            self.encoder = MLP(
                output_dim= latent_dim,
                activation="linear",
                hidden_activation="relu",
                num_layers=num_layers,
                name="encoder"
            )
            self.decoder = MLP(
                output_dim=input_size,
                hidden_activation="relu",
                activation="linear",
                num_layers=num_layers,
                name="decoder"
            )
        elif isinstance(nodes, list):
            self.encoder = keras.Sequential([
                keras.layers.Input(shape=(input_size,)),
                *[keras.layers.Dense(node, activation='relu') for node in nodes[:-1]],
                keras.layers.Dense(latent_dim, activation='relu', name="encoder")
            ])
            self.decoder = keras.Sequential([
                keras.layers.Input(shape=(latent_dim,)),
                *[keras.layers.Dense(node, activation='relu') for node in nodes[:-1][::-1]],
                keras.layers.Dense(input_size, activation='linear', name="decoder")
            ])
        else:
            raise ValueError("nodes must be an int or a list of ints")

        self.mu_layer = keras.layers.Dense(latent_dim, name="mu_layer", activation='linear')
        self.log_var_layer = keras.layers.Dense(latent_dim, name="log_var_layer", activation='linear')

    def reparameterize(self, mu, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        encoded = self.encoder(inputs)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)
        z = self.reparameterize(mu, log_var)
        return tf.concat([z, mu, log_var], axis=-1)


    def get_loss_object(self):
        def loss_fn(inputs, outputs):
            z, mu, log_var = tf.split(outputs, [self.latent_dim, self.latent_dim, self.latent_dim], axis=-1)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - self.decoder(z)),axis = -1))
            kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
            return reconstruction_loss + kl_loss
        return loss_fn