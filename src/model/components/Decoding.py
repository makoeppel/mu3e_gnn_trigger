import keras
import tensorflow as tf
from keras import layers


class DecoderQueries(layers.Layer):
    def __init__(self, num_queries, feature_dim, **kwargs):
        super(DecoderQueries, self).__init__(**kwargs)
        self.num_queries = num_queries
        self.feature_dim = feature_dim


    def build(self, input_shape):
        self.queries = self.add_weight(
            shape=(self.num_queries, self.feature_dim),
            initializer="random_normal",
            trainable=True,
            name="queries",
        )
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