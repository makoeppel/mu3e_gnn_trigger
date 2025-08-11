import keras
import tensorflow as tf
from keras.layers import Layer


class PositionalEncoding(Layer):
    def __init__(self, dim, mode="onehot", padding_value=-1, **kwargs):
        """
        Args:
            dim (int): Encoding dimension.
            mode (str): One of ['onehot', 'binary', 'sinusoid'].
            padding_value (int/float): Value to use for masked positions.
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.mode = mode
        self.padding_value = padding_value
        self.supports_masking = True  # Ensure compatibility with Keras masking

    def call(self, inputs, mask=None):
        """
        Args:
            x: Tensor of indices [B, N]
            mask: Boolean mask of shape [B, N, 1] (True = valid)

        Returns:
            Tensor of shape [B, N, dim]
        """

        x = tf.range(tf.shape(inputs)[-2], dtype=tf.int32)

        if self.mode == "onehot":
            enc = tf.one_hot(x, depth=self.dim, dtype=tf.float32)

        elif self.mode == "sinusoid":
            enc = self.sinusoidal(x)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        # Reshape to [B, N, dim]
        enc = tf.expand_dims(enc, axis=0)
        enc = tf.tile(enc, [tf.shape(inputs)[0], 1, 1])  # [B, N, dim]

        # Apply mask if present
        if mask is not None:
            # Ensure mask has shape [..., 1] to match encodings
            mask = tf.cast(mask, tf.bool)
            mask = tf.broadcast_to(mask, tf.shape(enc))  # [B, N, dim]
            enc = tf.where(mask, enc, tf.ones_like(enc) * self.padding_value)
        return enc

    def sinusoidal(self, x):
        x = tf.cast(x, tf.float32)
        x_shape = tf.shape(x)
        pos = tf.reshape(x, [-1])  # [B*N]

        i = tf.range(self.dim, dtype=tf.float32)
        angle_rates = 1 / tf.pow(
            10000.0, (2 * (i // 2)) / tf.cast(self.dim, tf.float32)
        )
        angles = tf.expand_dims(pos, -1) * angle_rates  # [B*N, dim]

        enc = tf.where(
            tf.cast(tf.expand_dims(i % 2 == 0, 0), tf.bool),
            tf.sin(angles),
            tf.cos(angles),
        )

        return tf.reshape(enc, tf.concat([x_shape, [self.dim]], axis=0))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.dim,)


class SumOverSequenceLength(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SumOverSequenceLength, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.broadcast_to(mask, tf.shape(inputs))
            inputs *= tf.cast(mask, inputs.dtype)  # Apply mask
        return tf.reduce_sum(inputs, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            1,
            input_shape[-1],
        ) 

class MaskedSetSorter(keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Sorts each set along a precomputed ranking magnitude, ignoring masked entries.
        """
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        """
        Args:
            inputs: Tuple (set_inputs, mag)
                - set_inputs: Tensor [B, N, d]
                - mag: Tensor [B, N, 1]
            mask: Tensor [B, N], True for valid entries
        Returns:
            sorted_x: Tensor [B, N, d]
        """
        set_inputs, mag = inputs
        mag = tf.abs(mag)

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)
            # Compute per-sample max mag: [B, 1, 1]
            max_per_sample = tf.reduce_max(mag, axis=1, keepdims=True)
            large_val = max_per_sample + 1.0
            mag = tf.where(mask, mag, tf.broadcast_to(large_val, tf.shape(mag)))

        # Sort by mag[..., 0]
        sort_indices = tf.argsort(mag[:, :, 0], axis=1, direction='ASCENDING')  # [B, N]
        sorted_x = tf.gather(set_inputs, sort_indices, batch_dims=1)

        return sorted_x

    def compute_output_shape(self, input_shape):
        return input_shape[0]
