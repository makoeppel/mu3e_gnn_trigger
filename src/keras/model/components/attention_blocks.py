"""
This module contains various attention block implementations for use in neural networks.
The attention blocks include:
- SelfAttentionBlock: A self-attention mechanism that allows the model to focus on different parts
  of the input sequence.
- SelfAttentionStack: A stack of self-attention blocks for deeper attention mechanisms.
- MultiHeadAttentionBlock: A multi-head attention mechanism that allows the model to jointly attend
  to information from different representation subspaces.
- MultiHeadAttentionStack: A stack of multi-head attention blocks for more complex attention patterns.
- CrossAttentionBlock: A cross-attention mechanism that attends two sequences to each other.
- CrossAttentionStack: A stack of cross-attention blocks for complex interactions between two sequences.
- PoolingAttentionBlock: An attention mechanism that pools information from a set of seed vectors.
- InducedSetAttentionBlock: An attention mechanism that uses induced sets for efficient attention computation.
- point_transformer: A transformer-like architecture for processing point cloud data.
"""

import keras
import tensorflow as tf
from keras import layers
from . import MLP


import tensorflow as tf
from keras import layers, regularizers


@keras.utils.register_keras_serializable(
    package="Custom", name="MultiHeadAttentionBlock"
)
class MultiHeadAttentionBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        ff_dim=None,
        dropout_rate=0,
        self_attention=False,
        pre_ln=True,
        regularizer=None,
        **kwargs,
    ):
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        super().__init__(**kwargs)
        self.self_attention = self_attention
        self.pre_ln = pre_ln
        self.regularizer = regularizer
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim or key_dim * 2
        self.dropout_rate = dropout_rate
        self.supports_masking = False
        # LayerNorms
        self.ln_q = layers.LayerNormalization(epsilon=1e-6, name="ln_q")
        self.ln_kv = (
            layers.LayerNormalization(epsilon=1e-6, name="ln_kv")
            if not self_attention and pre_ln
            else None
        )
        self.ln_ffn = (
            layers.LayerNormalization(epsilon=1e-6, name="ln_ffn") if pre_ln else None
        )

        # Attention
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim // num_heads,
            dropout=dropout_rate,
            name="multi_head_attention",
        )
        self.dropout_attn = layers.Dropout(dropout_rate, name="dropout_attn")

        # Feedforward network
        self.ffn_dense_1 = layers.Dense(
            self.ff_dim, activation="gelu", name="ffn_dense_1"
        )
        self.ffn_dense_2 = layers.Dense(key_dim, name="ffn_dense_2")
        self.ffn_dropout = layers.Dropout(dropout_rate, name="ffn_dropout_1")

    def call(
        self,
        query,
        value=None,
        key=None,
        query_mask=None,
        key_mask=None,
        value_mask=None,
        training=None,
    ):
        if value is None and not self.self_attention:
            raise ValueError("Value must be provided unless self-attention is used.")
        if self.pre_ln:
            return self._call_pre_ln(
                query=query,
                value=value,
                key=key,
                query_mask=query_mask,
                value_mask=value_mask,
                key_mask=key_mask,
                training=training,
            )
        else:
            return self._call_post_ln(
                query, value, key, query_mask, value_mask, key_mask, training
            )

    def _call_pre_ln(
        self, query, value, key, query_mask, value_mask, key_mask, training
    ):
        if key is None:
            key = value
            key_mask = value_mask

        if self.self_attention:
            q_norm = self.ln_q(query)
            k_norm = v_norm = q_norm
            # For self-attention, use query masks for key/value too
            key_mask = value_mask = query_mask
        else:
            q_norm = self.ln_q(query)
            if self.ln_kv is not None:
                k_norm = self.ln_kv(key)
                v_norm = self.ln_kv(value)
            else:
                k_norm = key
                v_norm = value

        attn_out = self.attn(
            q_norm,
            k_norm,
            v_norm,
            query_mask=query_mask,
            key_mask=key_mask,
            value_mask=value_mask,
            training=training,
        )
        attn_out = self.dropout_attn(attn_out, training=training)
        x = query + attn_out  # residual

        if self.ln_ffn is not None:
            ffn_in = self.ln_ffn(x)
        else:
            ffn_in = x

        ffn_out = self.ffn_dense_1(ffn_in)
        ffn_out = self.ffn_dropout(ffn_out, training=training)
        ffn_out = self.ffn_dense_2(ffn_out)
        return x + ffn_out

    def _call_post_ln(
        self, query, value, key, query_mask, value_mask, key_mask, training
    ):
        if key is None:
            key = value
            key_mask = value_mask
        # Attention
        if self.self_attention:
            attn_out = self.attn(
                query,
                query,
                query,
                query_mask=query_mask,
                key_mask=query_mask,
                value_mask=query_mask,
                training=training,
            )
        else:
            attn_out = self.attn(
                query,
                key,
                value,
                query_mask=query_mask,
                key_mask=key_mask,
                value_mask=value_mask,
                training=training,
            )

        attn_out = self.dropout_attn(attn_out, training=training)
        x = self.ln_q(query + attn_out)  # LN after residual

        # FFN
        ffn_out = self.ffn_dense_1(x)
        ffn_out = self.ffn_dropout(ffn_out, training=training)
        ffn_out = self.ffn_dense_2(ffn_out)

        if self.ln_ffn is not None:
            return self.ln_ffn(x + ffn_out)
        else:
            return x + ffn_out

    def build(self, query_shape, value_shape, key_shape=None):
        """Fixed build method for Keras compatibility"""
        if key_shape is None:
            key_shape = value_shape
        # Build layers
        self.ln_q.build(query_shape)
        if self.ln_kv is not None:
            self.ln_kv.build(value_shape)
        if self.ln_ffn is not None:
            self.ln_ffn.build(query_shape)

        self.attn.build(query_shape, value_shape, key_shape)
        self.ffn_dense_1.build(
            (None, None, self.key_dim)
        )  # Use None for flexible batch/seq dims
        self.ffn_dense_2.build((None, None, self.ff_dim))

        super().build(query_shape)

    def compute_output_shape(self, query_shape, key_shape=None, value_shape=None):
        if value_shape is None:
            value_shape = query_shape
        if key_shape is None:
            key_shape = query_shape
        return (query_shape[0], query_shape[1], self.key_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "self_attention": self.self_attention,
                "pre_ln": self.pre_ln,
                "regularizer": keras.regularizers.serialize(self.regularizer),
            }
        )
        return config


@keras.utils.register_keras_serializable(package="Custom")
class SelfAttentionBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        dropout_rate=0.0,
        ff_dim=None,
        name="self_attention_block",
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim or key_dim * 2

        self.attention = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            name=f"{name}_attention",
            regularizer=regularizer,
            self_attention=True,
            pre_ln=pre_ln,
        )

    def call(self, inputs, mask=None, training=None):

        x = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            key_mask=mask,
            value_mask=mask,
            query_mask=mask,
            training=training,
        )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "name": self.name if hasattr(self, "name") else None,
                "pre_ln": self.attention.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    def build(self, input_shape):
        # build all child layers explicitly
        self.attention.build(input_shape, input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def count_params(self):
        return self.attention.count_params()


@keras.utils.register_keras_serializable(package="Custom")
class SelfAttentionStack(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        stack_size=3,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super(SelfAttentionStack, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.stack_size = stack_size
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.attention_blocks = [
            SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                ff_dim=ff_dim,
                name=f"attention_block_{i+1}",
                pre_ln=pre_ln,
                regularizer=regularizer,
            )
            for i in range(stack_size)
        ]

    def call(self, inputs, mask=None, training=None):
        x = inputs
        for block in self.attention_blocks:
            x = block(x, mask=mask, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "stack_size": self.stack_size,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    def build(self, input_shape):
        x_shape = input_shape
        for block in self.attention_blocks:
            block.build(x_shape)
        super().build(input_shape)

    def count_params(self):
        return sum(block.count_params() for block in self.attention_blocks)


class MultiHeadAttentionStack(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        stack_size=3,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        self_attention=False,
        **kwargs,
    ):

        super(MultiHeadAttentionStack, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.stack_size = stack_size
        self.self_attention = self_attention
        self.pre_ln = pre_ln

        self.attention_blocks = [
            MultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                ff_dim=ff_dim,
                name=f"multi_head_attention_block_{i+1}",
                regularizer=regularizer,
                pre_ln=pre_ln,
                self_attention=self_attention,
            )
            for i in range(stack_size)
        ]

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        key_mask=None,
        value_mask=None,
        training=None,
    ):
        if key is None:
            key = value
        x = query
        if self.self_attention:
            value = query
            key = query
            key_mask = query_mask
            value_mask = query_mask
        for block in self.attention_blocks:
            x = block(
                query=x,
                value=value,
                key=key,
                query_mask=query_mask,
                key_mask=key_mask,
                value_mask=value_mask,
                training=training,
            )
        return x

    def compute_output_shape(self, query_shape, key_shape=None, value_shape=None):
        if value_shape is None:
            value_shape = query_shape
        if key_shape is None:
            key_shape = query_shape
        return (query_shape[0], query_shape[1], self.key_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "stack_size": self.stack_size,
                "self_attention": self.self_attention,
                "pre_ln": self.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
                "name": self.name if hasattr(self, "name") else None,
            }
        )
        return config

    def build(self, query_shape, value_shape):
        super(MultiHeadAttentionStack, self).build(query_shape)
        for block in self.attention_blocks:
            block.build(query_shape, value_shape)
        # Ensure the layer is built with the correct input shape

    def count_params(self):
        return sum(block.count_params() for block in self.attention_blocks)


@keras.utils.register_keras_serializable(package="Custom")
class CrossAttentionBlock(layers.Layer):
    """
    Cross Attention Block
    Implements cross-attention between two sequences.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super(CrossAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim or key_dim * 2
        self.regularizer = regularizer
        self.pre_ln = pre_ln

        # Create cross-attention blocks (not self-attention!)
        self.b_to_a_attention = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            regularizer=regularizer,
            pre_ln=pre_ln,
            self_attention=False,  # Important: this is cross-attention
            name="b_to_a_attention",
        )
        self.a_to_b_attention = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            regularizer=regularizer,
            pre_ln=pre_ln,
            self_attention=False,  # Important: this is cross-attention
            name="a_to_b_attention",
        )

    def call(self, a, b, a_mask, b_mask, training=None):
        """
        Args:
            a: First sequence (query)
            b: Second sequence (key/value)
            a_mask: Mask for the first sequence
            b_mask: Mask for the second sequence
            training: Whether in training mode
        Returns:
            tuple: (output_a, output_b)
        """
        # Handle different input formats

        # Cross-attention: a attends to b, b attends to a
        output_a = self.b_to_a_attention(
            query=a,
            key=b,
            value=b,
            query_mask=a_mask,
            key_mask=b_mask,
            value_mask=b_mask,
            training=training,
        )

        output_b = self.a_to_b_attention(
            query=b,
            key=a,
            value=a,
            query_mask=b_mask,
            key_mask=a_mask,
            value_mask=a_mask,
            training=training,
        )

        return (output_a, output_b)

    def build(self, a_shape, b_shape):
        """
        Build method that works with Keras conventions
        """
        # Build the attention layers with correct shapes
        # For cross-attention: query_shape, key_value_shape
        self.b_to_a_attention.build(a_shape, b_shape)  # a queries b
        self.a_to_b_attention.build(b_shape, a_shape)  # b queries a

        super().build(a_shape)

    def compute_output_shape(self, a_shape, b_shape):
        return (a_shape, b_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "regularizer": keras.regularizers.serialize(self.regularizer),
                "pre_ln": self.pre_ln,
            }
        )
        return config

    def count_params(self):
        """
        Count total parameters in both attention blocks
        """
        return (
            self.b_to_a_attention.count_params() + self.a_to_b_attention.count_params()
        )


@keras.utils.register_keras_serializable(package="Custom")
class CrossAttentionStack(layers.Layer):

    def __init__(
        self,
        num_heads,
        key_dim,
        stack_size=3,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super(CrossAttentionStack, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.stack_size = stack_size

        self.attention_blocks = [
            CrossAttentionBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                ff_dim=ff_dim,
                name=f"cross_attention_block_{i+1}",
                regularizer=regularizer,
                pre_ln=pre_ln,
            )
            for i in range(stack_size)
        ]

    def build(self, a_shape, b_shape):
        """
        Build method that works with Keras conventions
        """

        # Build the attention layers with correct shapes
        for block in self.attention_blocks:
            block.build(a_shape, b_shape)

        super().build((a_shape, b_shape))

    def call(self, a, b, a_mask, b_mask, training=None):
        """
        Args:
            inputs: Can be:
                - [a, b] where a, b are sequences
                - [a, b, a_mask, b_mask] with masks
                - dict with keys 'a', 'b', 'a_mask', 'b_mask'

        Returns:
            tuple: (output_a, output_b)
        """
        # Handle different input formats
        x_a = a
        x_b = b
        for block in self.attention_blocks:
            x_a, x_b = block(x_a, x_b, a_mask, b_mask, training=training)
        return x_a, x_b

    def compute_output_shape(self, a_shape, b_shape):
        return (a_shape, b_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "stack_size": self.stack_size,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config


class PoolingAttentionBlock(layers.Layer):
    """
    Set Transformer style Pooling by Multihead Attention (PMA)
    Based on equation (11): PMA_k(Z) = MAB(S, rFF(Z))

    Since MultiHeadAttentionBlock already implements MAB with feed-forward,
    this layer focuses on the PMA structure.
    """

    def __init__(
        self,
        key_dim,
        num_seeds,
        num_heads=4,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super(PoolingAttentionBlock, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer

        # Learnable seed vectors (S in the paper)
        self.seed_vectors = self.add_weight(
            shape=(num_seeds, key_dim),
            initializer="glorot_uniform",  # Better than random_normal
            trainable=True,
            name="seed_vectors",
            regularizer=self.regularizer,
        )

        if ff_dim is None:
            self.ff_dim = key_dim * 2
        else:
            self.ff_dim = ff_dim

        # Pre-processing feed-forward for inputs (rFF(Z) in equation 11)
        self.input_ff_1 = layers.Dense(
            self.ff_dim,
            activation="relu",
            name="input_ff_1",
            # kernel_regularizer=self.regularizer,
            # bias_regularizer=self.regularizer,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
        self.input_ff_2 = layers.Dense(
            key_dim,
            name="input_ff_2",
            # kernel_regularizer=self.regularizer,
            # bias_regularizer=self.regularizer,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )

        # Use your MultiHeadAttentionBlock which already implements MAB
        self.MHA = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            regularizer=self.regularizer,
            pre_ln=pre_ln,
            name="pooling_mha",
        )

    def build(self, input_shape):
        super(PoolingAttentionBlock, self).build(input_shape)

        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch_size, seq_len, features). Got {input_shape}."
            )

        if input_shape[-1] != self.key_dim:
            raise ValueError(
                f"Input feature dimension {input_shape[-1]} must match key_dim {self.key_dim}"
            )

        # Build the components
        seed_shape = (None, self.num_seeds, self.key_dim)
        processed_input_shape = (None, input_shape[1], self.key_dim)

        self.input_ff_1.build(input_shape)
        self.input_ff_2.build((*input_shape[:-1], self.ff_dim))
        self.MHA.build(seed_shape, processed_input_shape)

    def call(self, inputs, mask=None):
        """
        Implements PMA_k(Z) = MAB(S, rFF(Z)) from Set Transformer paper

        Args:
            inputs: (batch_size, seq_len, key_dim)
            mask: (batch_size, seq_len) boolean mask for inputs

        Returns:
            output: (batch_size, num_seeds, key_dim)
        """
        batch_size = tf.shape(inputs)[0]

        processed_inputs = self.input_ff_1(inputs)
        processed_inputs = self.input_ff_2(processed_inputs)

        seed_vectors_expanded = tf.broadcast_to(
            self.seed_vectors[tf.newaxis, ...],
            [batch_size, self.num_seeds, self.key_dim],
        )

        # Step 3: MAB(S, rFF(Z)) using your MultiHeadAttentionBlock
        # Query: seed vectors, Key/Value: processed inputs
        output = self.MHA(
            query=seed_vectors_expanded,
            value=processed_inputs,
            key=processed_inputs,
            key_mask=mask,
        )

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_seeds, self.key_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "num_seeds": self.num_seeds,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    def count_params(self):
        return (
            self.seed_vectors.shape[0] * self.seed_vectors.shape[1]
            + self.input_ff_1.count_params()
            + self.input_ff_2.count_params()
            + self.MHA.count_params()
        )


class InducedSetAttentionBlock(layers.Layer):
    def __init__(
        self, key_dim, num_seeds, num_heads=4, dropout_rate=0.0, ff_dim=None, **kwargs
    ):
        super(InducedSetAttentionBlock, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.seed_vectors = self.add_weight(
            shape=(num_seeds, key_dim),
            initializer="random_normal",
            trainable=True,
            name="seed_vectors",
        )
        self.IMAB = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            name="induced_set_attention_mha",
        )
        self.MAB = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            name="induced_set_attention_mab",
        )

    def build(self, input_shape):
        super(InducedSetAttentionBlock, self).build(input_shape)
        seed_vectors_shape = (None, self.num_seeds, self.key_dim)

        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch_size, num_points, key_dim). Got {input_shape}."
            )

        value_shape = (None, input_shape[1], self.key_dim)

        self.IMAB.build([seed_vectors_shape, value_shape])
        self.MAB.build([value_shape, seed_vectors_shape])
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, inputs, mask=None, training=None):
        # inputs: (batch_size, num_points, key_dim)
        batch_size = tf.shape(inputs)[0]

        # Expand seed vectors to match batch size and number of points
        seed_vectors_expanded = tf.broadcast_to(
            self.seed_vectors[tf.newaxis, ...],
            [batch_size, self.num_seeds, self.key_dim],
        )

        # Apply Induced MultiHeadAttention Block
        induced_output = self.IMAB(
            query=seed_vectors_expanded,
            value=inputs,
            key=inputs,
            key_mask=mask,
        )

        # Apply MultiHeadAttention Block
        output = self.MAB(
            query=inputs,
            value=induced_output,
            key=induced_output,
            query_mask=mask,
        )

        return output


class point_transformer(keras.layers.Layer):

    def __init__(self, dim=8, attn_hidden=4, pos_hidden=8, name=None, **kwargs):
        super(point_transformer, self).__init__(name=name, **kwargs)

        self.initializer = keras.initializers.HeNormal()

        self.linear1 = keras.layers.Dense(
            dim,
            activation="relu",
            kernel_initializer=self.initializer,
            name="self.linear1",
        )
        self.linear2 = keras.layers.Dense(
            dim,
            activation=None,
            kernel_initializer=self.initializer,
            name="self.linear2",
        )
        self.MLP_attn1 = layers.Dense(
            attn_hidden,
            activation="relu",
            kernel_initializer=self.initializer,
            name="attn_hidden",
        )
        self.MLP_attn2 = layers.Dense(
            dim,
            activation="relu",
            kernel_initializer=self.initializer,
            name="self.MLP_attn2",
        )
        self.MLP_pos1 = layers.Dense(
            pos_hidden,
            activation="relu",
            kernel_initializer=self.initializer,
            name="pos_hidden",
        )
        self.MLP_pos2 = layers.Dense(
            dim,
            activation="relu",
            kernel_initializer=self.initializer,
            name="self.MLP_pos2",
        )
        self.linear_query = layers.Dense(
            dim,
            activation="relu",
            kernel_initializer=self.initializer,
            name="self.linear_query",
        )
        self.linear_key = layers.Dense(
            dim,
            activation="relu",
            kernel_initializer=self.initializer,
            name="self.linear_key",
        )
        self.linear_value = layers.Dense(
            dim,
            activation="relu",
            kernel_initializer=self.initializer,
            name="self.linear_value",
        )

    def call(self, feature, pos, mask=None):

        n = pos.shape[-2]

        feature = self.linear1(feature)

        query = self.linear_query(feature)
        key = self.linear_key(feature)
        value = self.linear_value(feature)

        qk = (
            query[:, None, :, :] - key[:, :, None, :]
        )  # (B, 1, N, D) - (B, N, 1, D) -> (B, N, N, D)
        pos_rel = (
            pos[:, None, :, :] - pos[:, :, None, :]
        )  # (B, 1, N, D) - (B, N, 1, D) -> (B, N, N, D)

        value = value[:, None, :, :]  # (B, 1, N, D)

        pos_emb = self.MLP_pos1(pos_rel)  # (B, N, N, D)
        pos_emb = self.MLP_pos2(pos_emb)  # (B, N, N, D)

        value = value + pos_emb  # (B, N, N, D)

        mlp_attn1 = self.MLP_attn1(qk + pos_emb)
        if mask is not None:
            mask = tf.cast(mask, tf.bool)  # Ensure mask is boolean
            key_mask = mask[:, :, None]  # (B, N, 1)
            query_mask = mask[:, None, :]  # (B, 1, N)
            attention_mask = tf.math.logical_and(key_mask, query_mask)  # (B, N, N)
            attention_mask = tf.expand_dims(attention_mask, axis=-1)  # (B, N, N, 1)
        else:
            attention_mask = tf.ones(
                (tf.shape(feature)[0], n, n, 1), dtype=tf.bool
            )  # (B, N, N, 1)
        softmax_mask = tf.where(
            attention_mask, 0.0, -1e9
        )  # Apply mask to attention logits

        mlp2_attn = self.MLP_attn2(mlp_attn1) + pos_emb + softmax_mask  # (B, N, N, D)

        attn = tf.nn.softmax(mlp2_attn, axis=-2)  # (B, N, N, D)
        out = value * attn  # (B, N, N, D)
        out = tf.math.reduce_sum(out, axis=-2)  # (B, N, D)
        out = self.linear2(out)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.linear2.units
