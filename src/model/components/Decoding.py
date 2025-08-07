import keras
import tensorflow as tf
from keras import layers


class DecoderQueries(layers.Layer):
    def __init__(self, num_queries, feature_dim, learnable = False, **kwargs):
        self.regularizer = kwargs.pop('regularizer', None)
        super(DecoderQueries, self).__init__(**kwargs)
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        self.queries = self.add_weight(
            shape=(self.num_queries, self.feature_dim),
            initializer="random_normal",
            trainable=True,
            regularizer=self.regularizer,
            name="queries",
        )


    def build(self, input_shape):
        super(DecoderQueries, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.broadcast_to(
            self.queries[tf.newaxis, ...],
            [batch_size, self.num_queries, self.feature_dim],
        )


    def get_config(self):
        config = super(DecoderQueries, self).get_config()
        config.update({
            "num_queries": self.num_queries,
            "feature_dim": self.feature_dim,
        })
        return config
    

class DecoderPoints(layers.Layer):
    """
    Generates a set of points evenly spaced in a given range.
    """
    def __init__(self, num_points, feature_dim, range = [-1,1,], **kwargs):
        super(DecoderPoints, self).__init__(**kwargs)
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.range = range
        self.points = self.add_weight(
            shape=(self.num_points, self.feature_dim),
            initializer="random_uniform",
            trainable=False,
            name="points",
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        scaled_points = self.points * (self.range[1] - self.range[0]) + self.range[0]
        points = tf.broadcast_to(
            scaled_points[tf.newaxis, ...],
            [batch_size, self.num_points, self.feature_dim],
        )
        return points


    def get_config(self):
        config = super(DecoderPoints, self).get_config()
        config.update({
            "num_points": self.num_points,
            "feature_dim": self.feature_dim,
        })
        return config