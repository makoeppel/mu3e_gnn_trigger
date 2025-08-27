import keras
import tensorflow as tf


class ReconstructionQuality(keras.metrics.Metric):
    def __init__(self, name="reconstruction_quality", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, _, y_pred, sample_weight=None):
        latent_dim = y_pred.shape[-1] // 2
        z, ae_output = tf.split(y_pred, [latent_dim, latent_dim], axis=-1)
        loss = tf.reduce_mean(tf.square(z - ae_output))
        self.total_loss.assign_add(loss)
        self.count.assign_add(1)

    def result(self):
        return self.total_loss / self.count


class FeatureVariance(keras.metrics.Metric):
    def __init__(self, name="feature_variance", **kwargs):
        super().__init__(name=name, **kwargs)
        self.variance = self.add_weight(name="variance", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, _, y_pred, sample_weight=None):
        latent_dim = y_pred.shape[-1] // 2
        z, ae_output = tf.split(y_pred, [latent_dim, latent_dim], axis=-1)
        variance = tf.math.reduce_variance(z, axis=0)
        self.variance.assign_add(tf.reduce_mean(variance))
        self.count.assign_add(1)

    def result(self):
        return self.variance / self.count
