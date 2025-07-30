import keras
import tensorflow as tf
from keras import layers


class SelfAttentionBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        dropout_rate=0.0,
        name="self_attention_block",
        **kwargs
    ):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.name = name

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name="self_attention_layer"
        )
        self.dropout1 = layers.Dropout(dropout_rate, name="dropout_1")
        self.layer_norm_1 = layers.LayerNormalization(name="layer_norm_1")

        self.ff_layer = layers.Dense(key_dim, activation="relu", name="ffn_layer")

        self.dropout2 = layers.Dropout(dropout_rate)
        self.layer_norm_2 = layers.LayerNormalization(name="layer_norm_2")

    def call(self, inputs, mask=None, training=None):
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = inputs + attention_output
        attention_output = self.layer_norm_1(attention_output)

        ff_output = self.ff_layer(attention_output)
        ff_output = self.dropout2(ff_output, training=training)
        ff_output = attention_output + ff_output
        ff_output = self.layer_norm_2(ff_output)

        return ff_output

    def build(self, input_shape):
        super(SelfAttentionBlock, self).build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # Build all sub-layers
        self.attention.build(input_shape, input_shape)
        self.dropout1.build(input_shape)
        self.layer_norm_1.build(input_shape)
        self.ff_layer.build(input_shape)
        self.dropout2.build(input_shape)
        self.layer_norm_2.build(input_shape)
        self.input_spec = layers.InputSpec(shape=input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
            }
        )
        return config

    def count_params(self):
        return (
            self.attention.count_params()
            + self.ff_layer.count_params()
            + self.layer_norm_1.count_params()
            + self.layer_norm_2.count_params()
            + self.dropout1.count_params()
            + self.dropout2.count_params()
        )


class SelfAttentionStack(layers.Layer):
    def __init__(self, num_heads, key_dim, stack_size=3, dropout_rate=0.0, **kwargs):
        super(SelfAttentionStack, self).__init__(**kwargs)
        self.attention_blocks = [
            SelfAttentionBlock(
                num_heads=num_heads, key_dim=key_dim, dropout_rate=dropout_rate
            )
            for _ in range(stack_size)  # Example: 2 attention blocks
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
                "num_heads": self.attention_blocks[0].num_heads,
                "key_dim": self.attention_blocks[0].key_dim,
                "stack_size": len(self.attention_blocks),
                "dropout_rate": self.attention_blocks[0].dropout_rate,
            }
        )
        return config

    def build(self, input_shape):
        super(SelfAttentionStack, self).build(input_shape)
        for block in self.attention_blocks:
            block.build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)

    def count_params(self):
        return sum(block.count_params() for block in self.attention_blocks)


class MultiHeadAttentionBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, **kwargs):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name="multi_head_attention_layer"
        )
        self.dropout = layers.Dropout(dropout_rate, name="multi_head_attention_dropout")
        self.layer_norm = layers.LayerNormalization(
            name="multi_head_attention_layer_norm"
        )
        self.ff_layer = layers.Dense(key_dim, activation="relu", name="ffn_layer")
        self.ff_dropout = layers.Dropout(dropout_rate, name="ffn_dropout")
        self.ff_layer_norm = layers.LayerNormalization(name="ffn_layer_norm")

    def build(self, input_shape):
        super(MultiHeadAttentionBlock, self).build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # Build all sub-layers
        self.attention.build(input_shape, input_shape)
        self.dropout.build(input_shape)
        self.layer_norm.build(input_shape)
        self.ff_layer.build(input_shape)
        self.ff_dropout.build(input_shape)
        self.ff_layer_norm.build(input_shape)
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(
        self,
        query,
        value,
        key=None,
        key_mask=None,
        value_mask=None,
        attention_mask=None,
        training=None,
    ):
        if key is None:
            key = query

        if attention_mask is not None:
            if attention_mask.shape.rank == 2:
                attention_mask = tf.expand_dims(attention_mask, axis=-1)
        attention_output = self.attention(
            query,
            value,
            key,
            attention_mask=attention_mask,
            key_mask=key_mask,
            value_mask=value_mask,
        )
        attention_output = self.dropout(attention_output, training=training)
        attention_output = self.layer_norm(attention_output + query)
        ff_output = self.ff_layer(attention_output)
        ff_output = self.ff_dropout(ff_output, training=training)
        ff_output = self.ff_layer_norm(ff_output + attention_output)
        return ff_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
            }
        )
        return config

    def count_params(self):
        return (
            self.attention.count_params()
            + self.ff_layer.count_params()
            + self.layer_norm.count_params()
            + self.dropout.count_params()
            + self.ff_dropout.count_params()
            + self.ff_layer_norm.count_params()
        )


class PoolingAttentionBlock(layers.Layer):
    def __init__(
        self,
        key_dim,
        num_heads,
        num_seed_vectors,
        mlp_ratio=4,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_seed_vectors = num_seed_vectors
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate

        # Learnable seed (latent) vectors
        self.seed_vectors = self.add_weight(
            shape=(num_seed_vectors, key_dim),
            initializer="random_normal",
            trainable=True,
            name="seed_vectors"
        )

        # Attention layer: seeds as query, input as key/value
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name="multihead_attention"
        )
        self.attn_norm = layers.LayerNormalization(epsilon=1e-6, name="attn_norm")
        self.attn_dropout = layers.Dropout(dropout_rate)

        # MLP block
        self.mlp = keras.Sequential([
            layers.Dense(key_dim * mlp_ratio, activation="gelu", name="mlp_dense_1"),
            layers.Dropout(dropout_rate),
            layers.Dense(key_dim, name="mlp_dense_2"),
        ])
        self.mlp_norm = layers.LayerNormalization(epsilon=1e-6, name="mlp_norm")
        self.mlp_dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]

        # Repeat seed vectors for batch
        latent = tf.tile(tf.expand_dims(self.seed_vectors, axis=0), [batch_size, 1, 1])

        # Attention: latent attends to input tokens
        attn_output = self.attn(
            query=latent,
            value=inputs,
            key=inputs,
            value_mask=mask,
            training=training
        )
        attn_output = self.attn_dropout(attn_output, training=training)
        latent = self.attn_norm(latent + attn_output)

        # Feedforward MLP
        mlp_output = self.mlp(latent, training=training)
        mlp_output = self.mlp_dropout(mlp_output, training=training)
        output = self.mlp_norm(latent + mlp_output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_seed_vectors, self.key_dim)

    def build(self, input_shape):
        super().build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)

    def count_params(self):
        return (
            self.seed_vectors.shape.num_elements()
            + self.attn.count_params()
            + self.mlp.count_params()
            + self.attn_norm.count_params()
            + self.mlp_norm.count_params()
            + self.attn_dropout.count_params()
            + self.mlp_dropout.count_params()
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "key_dim": self.key_dim,
            "num_heads": self.num_heads,
            "num_seed_vectors": self.num_seed_vectors,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
        })
        return config

from . import MLP
class PointTransformer(layers.Layer):
    def __init__(self, key_dim, mlp_depth = 2, dropout_rate = 0, **kwargs):
        super(PointTransformer, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.phi = MLP(
            key_dim,
            num_layers=mlp_depth,
            dropout_rate=dropout_rate,
            name="phi_mlp",
            hidden_activation="relu",
            activation="linear"
        )
        self.psi = MLP(
            key_dim,
            num_layers=mlp_depth,
            dropout_rate=dropout_rate,
            name="psi_mlp",
            hidden_activation="relu",
            activation="linear"
        )
        self.pos_encoder = MLP(
            key_dim,
            num_layers=mlp_depth,
            dropout_rate=dropout_rate,
            name="pos_encoder_mlp",
            hidden_activation="relu",
            activation="linear"
        )
        self.alpha = MLP(
            key_dim,
            num_layers=mlp_depth,
            dropout_rate=dropout_rate,
            name="alpha_mlp",
            hidden_activation="relu",
            activation="linear"
        )
        self.theta = MLP(
            key_dim,
            num_layers=mlp_depth,
            dropout_rate=dropout_rate,
            name="theta_mlp",
            hidden_activation="relu",
            activation="linear"
        )
        self.gamma = MLP(
            key_dim,
            num_layers=mlp_depth,
            dropout_rate=dropout_rate,
            name="gamma_mlp",
            hidden_activation="relu",
            activation="linear"
        )

    def build(self, input_shape):
        super(PointTransformer, self).build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.key_dim)
    
    def count_params(self):
        return (
            self.phi.count_params()
            + self.psi.count_params()
            + self.pos_encoder.count_params()
            + self.alpha.count_params()
            + self.theta.count_params()
            + self.gamma.count_params()
        )

    @tf.function
    def call(self, inputs, mask=None):
        # inputs: (batch_size, num_points, key_dim)
        batch_size = tf.shape(inputs)[0]
        num_points = tf.shape(inputs)[1]

        phi_x = self.phi(inputs)     # (batch, N, D)
        psi_x = self.psi(inputs)     # (batch, N, D)
        alpha_x = self.alpha(inputs) # (batch, N, D)

        # Pairwise differences
        phi_exp = tf.expand_dims(phi_x, axis=1)  # (batch, 1, N, D)
        psi_exp = tf.expand_dims(psi_x, axis=2)  # (batch, N, 1, D)
        feature_diff = phi_exp - psi_exp         # (batch, N, N, D)

        pos_enc = self.theta(feature_diff)       # (batch, N, N, D)
        attn_logits = self.gamma(feature_diff + pos_enc)  # (batch, N, N, D)

        # === Masking ===
        if mask is not None:
            # mask: (batch, num_points)
            mask = tf.cast(mask, dtype=tf.float32)                       # (B, N)
            mask_row = tf.expand_dims(mask, axis=1)                      # (B, 1, N)
            mask_col = tf.expand_dims(mask, axis=2)                      # (B, N, 1)
            pair_mask = mask_row * mask_col                              # (B, N, N)
            pair_mask = tf.expand_dims(pair_mask, axis=-1)              # (B, N, N, 1)
            attn_logits += (1.0 - pair_mask) * -1e9  # set logits of masked pairs to large negative

        # Vector-wise softmax: softmax over axis=2 (neighbor dim), for each feature channel
        attn_weights = tf.nn.softmax(attn_logits, axis=2)  # (batch, N, N, D)

        # Attention-weighted sum
        alpha_exp = tf.expand_dims(alpha_x, axis=1)  # (batch, 1, N, D)
        aggregated = attn_weights * (alpha_exp + pos_enc)  # (batch, N, N, D)
        output = tf.reduce_sum(aggregated, axis=2)          # (batch, N, D)

        return output

    def compute_mask(self, inputs, mask=None):
        # propagate the input mask to the output
        return mask
