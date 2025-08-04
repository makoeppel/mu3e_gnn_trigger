import keras
import tensorflow as tf
import numpy as np
from .components import MLP
from keras import ops

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VarAutoEncoder(keras.Model):
    def __init__(self, input_size, latent_dim, num_layers: int , **kwargs):
        super().__init__(**kwargs)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.compiled_loss = None
        self.input_size = input_size
        self.latent_dim = latent_dim
        encoder_mlp = MLP(
            output_dim= latent_dim,
            activation="linear",
            hidden_activation="relu",
            num_layers=num_layers,
            name="encoder"
        )
        decoder_mlp = MLP(
            output_dim=input_size,
            hidden_activation="relu",
            activation="linear",
            num_layers=num_layers,
            name="decoder"
        )


        mu_layer = keras.layers.Dense(latent_dim, name="mu_layer", activation='linear')
        log_var_layer = keras.layers.Dense(latent_dim, name="log_var_layer", activation='linear')
        sampling_layer = Sampling(name="sampling_layer")

        input = keras.layers.Input(shape=(input_size,))
        encoding = encoder_mlp(input)
        z_mean = mu_layer(encoding)
        z_log_var = log_var_layer(encoding)
        z = sampling_layer([z_mean, z_log_var])
        reconstructed = decoder_mlp(z)
        self.encoder = keras.Model(input, [z_mean, z_log_var, z], name="encoder_model")
        self.decoder = keras.Model(z, reconstructed, name="decoder_model")
    
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

    def encode(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return z_mean, z_log_var
    
    def decode(self, z):
        return self.decoder(z)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(ops.square(data - reconstruction), axis=1)
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }