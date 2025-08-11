import keras
import tensorflow as tf


class GenerateMask(keras.layers.Layer):
    def __init__(self, padding_value=-1, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, inputs):
        # For shape (batch, seq_len, dim), we reduce across dim to get (batch, seq_len, 1)
        not_pad = tf.not_equal(inputs, self.padding_value)
        mask = tf.reduce_any(not_pad, axis=-1)
        return tf.cast(mask, tf.bool)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({"padding_value": self.padding_value})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MaskOutput(keras.layers.Layer):
    def __init__(self, padding_value=-1, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, inputs, mask=None):
        if mask is not None:
            if mask.shape.rank == 2:
                mask = tf.expand_dims(mask, axis=-1)
            mask = tf.broadcast_to(mask, tf.shape(inputs))
            padding_tensor = (
                tf.ones_like(inputs, dtype=inputs.dtype) * self.padding_value
            )
            inputs = tf.where(mask, inputs, padding_tensor)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"padding_value": self.padding_value})
        return config


class GetSequenceLength(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Inputs expected to be mask of shape (batch_size, seq_len, 1)
        inputs = tf.cast(inputs, tf.float32)
        return tf.reduce_sum(
            inputs, axis=1, keepdims=True
        )  # (batch_size, 1) if inputs has keepdims=True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        return super().get_config()


class GenerateDecoderMask(keras.layers.Layer):
    def __init__(self, max_length=256, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def call(self, inputs):
        """
        inputs: Tensor of shape (batch_size,) or (batch_size, 1) representing predicted sequence lengths.
        Returns: Boolean mask of shape (batch_size, max_length, 1)
        """
        predicted_seq_length = tf.cast(tf.round(tf.reshape(inputs, [-1])), tf.int32)
        predicted_seq_length = tf.clip_by_value(
            predicted_seq_length, 0, self.max_length
        )
        mask = tf.sequence_mask(
            predicted_seq_length, maxlen=self.max_length, dtype=tf.bool
        )
        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_length, 1)

    def get_config(self):
        config = super().get_config()
        config.update({"max_length": self.max_length})
        return config
