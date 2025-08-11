import tensorflow as tf
import keras


class SplitMSE(keras.losses.Loss):
    """Computes the mean squared error for two inputs and returns the sum of both losses."""

    def __init__(self, **kwargs):
        super(SplitMSE, self).__init__(**kwargs)

    def call(self, _, y_pred):
        y_pred_1, y_pred_2 = tf.split(y_pred, 2, axis=-1)
        return keras.losses.mean_squared_error(y_pred_1, y_pred_2)
