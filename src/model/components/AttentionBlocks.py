import keras
import tensorflow as tf
from keras import layers
from . import MLP


class SelfAttentionBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        dropout_rate=0.0,
        ff_dim=None,
        name="self_attention_block",
        **kwargs,
    ):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.name = name
        if ff_dim is None:
            ff_dim = key_dim * 2

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name="self_attention_layer"
        )
        self.dropout1 = layers.Dropout(dropout_rate, name="dropout_1")
        self.layer_norm_1 = layers.LayerNormalization(name="layer_norm_1")

        self.ff_layer = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu", name="ffn_dense_1"),
                layers.Dense(key_dim, name="ffn_dense_2"),
            ],
            name="ffn_layer",
        )

        self.dropout2 = layers.Dropout(dropout_rate)
        self.layer_norm_2 = layers.LayerNormalization(name="layer_norm_2")
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        attention_output = self.attention(
            inputs, inputs, query_mask=mask, key_mask=mask
        )
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
    def __init__(
        self, num_heads, key_dim, stack_size=3, dropout_rate=0.0, ff_dim=None, **kwargs
    ):
        super(SelfAttentionStack, self).__init__(**kwargs)
        self.attention_blocks = [
            SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                ff_dim=ff_dim,
                name=f"attention_block_{i+1}",
            )
            for i in range(stack_size)  # Example: 2 attention blocks
        ]
        self.supports_masking = True

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
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, ff_dim=None, **kwargs):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        if ff_dim is None:
            ff_dim = key_dim * 2

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name="multi_head_attention_layer"
        )
        self.dropout = layers.Dropout(dropout_rate, name="multi_head_attention_dropout")
        self.layer_norm = layers.LayerNormalization(
            name="multi_head_attention_layer_norm"
        )
        self.ff_layer = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu", name="ffn_dense_1"),
                layers.Dense(key_dim, name="ffn_dense_2"),
            ],
            name="ffn_layer",
        )
        self.ff_dropout = layers.Dropout(dropout_rate, name="ffn_dropout")
        self.ff_layer_norm = layers.LayerNormalization(name="ffn_layer_norm")
        self.supports_masking = True


    def build(self, input_shape):
        super(MultiHeadAttentionBlock, self).build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            query_shape, key_shape = input_shape
        else:
            query_shape = key_shape = input_shape
        # Build all sub-layers
        self.attention.build(query_shape, key_shape)
        self.dropout.build(query_shape)
        self.layer_norm.build(query_shape)
        self.ff_layer.build(query_shape)
        self.ff_dropout.build(query_shape)
        self.ff_layer_norm.build(query_shape)
        self.input_spec = layers.InputSpec(shape=query_shape)

    def call(
        self,
        query,
        value,
        key=None,
        key_mask=None,
        value_mask=None,
        query_mask=None,
        attention_mask=None,
        training=None,
    ):
        if key is None:
            key = value

        attention_output = self.attention(
            query,
            value,
            key,
            attention_mask=attention_mask,
            query_mask=query_mask,
            key_mask=key_mask,
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

class MultiHeadAttentionStack(layers.Layer):
    def __init__(self, num_heads, key_dim, stack_size=3, dropout_rate=0.0, ff_dim=None, **kwargs):
        super(MultiHeadAttentionStack, self).__init__(**kwargs)
        self.attention_blocks = [
            MultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                ff_dim=ff_dim,
                name=f"multi_head_attention_block_{i+1}",
            )
            for i in range(stack_size)
        ]
        self.supports_masking = True

    def call(
        self,
        query,
        value,
        key=None,
        key_mask=None,
        value_mask=None,
        query_mask=None,
        attention_mask=None,
        training=None,
    ):
        x = query
        for block in self.attention_blocks:
            x = block(
                query=x,
                value=value,
                key=key,
                key_mask=key_mask,
                value_mask=value_mask,
                query_mask=query_mask,
                attention_mask=attention_mask,
                training=training,
            )
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
        super(MultiHeadAttentionStack, self).build(input_shape)
        for block in self.attention_blocks:
            block.build(input_shape)
        # Ensure the layer is built with the correct input shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)

    def count_params(self):
        return sum(block.count_params() for block in self.attention_blocks)


class PoolingAttentionBlock(layers.Layer):
    def __init__(self, key_dim, num_seeds, num_heads=4, dropout_rate=0.0, ff_dim = None, **kwargs):
        super(PoolingAttentionBlock, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.seed_vectors = self.add_weight(
            shape=(num_seeds, key_dim),
            initializer="random_normal",
            trainable=True,
            name="seed_vectors",
        )
        if ff_dim is None:
            ff_dim = key_dim * 2
        self.ff_layer = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu", name="ffn_dense_1"),
                layers.Dense(key_dim, name="ffn_dense_2"),
            ],
            name="ffn_layer",
        )
        self.MHA = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            name="pooling_attention_mha",
        )

    def build(self, input_shape):
        super(PoolingAttentionBlock, self).build(input_shape)
        seed_vectors_shape = (None, self.num_seeds, self.key_dim)

        if len(input_shape) != 3:
            raise ValueError(f"Expected input shape (batch_size, num_points, key_dim). Got {input_shape}.")

        value_shape = (None, input_shape[1], self.key_dim)

        self.ff_layer.build(input_shape)
        self.MHA.build([seed_vectors_shape, value_shape])
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
        # Apply feed-forward layer to seed vectors
        ff_inputs = self.ff_layer(inputs)
        # Apply MultiHeadAttention
        output = self.MHA(
            query=seed_vectors_expanded,
            value=ff_inputs,
            key=ff_inputs,
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
            }
        )
        return config

    def count_params(self):
        return (
            self.MHA.count_params()
            + self.seed_vectors.shape[0] * self.seed_vectors.shape[1]
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
            raise ValueError(f"Expected input shape (batch_size, num_points, key_dim). Got {input_shape}.")

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

    def __init__(
        self, dim=8, attn_hidden=4, pos_hidden=8, name=None, **kwargs
    ):
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

        qk = query[:, None, :, :] - key[:, :, None, :] # (B, 1, N, D) - (B, N, 1, D) -> (B, N, N, D)
        pos_rel = pos[:, None, :, :] - pos[:, :, None, :] # (B, 1, N, D) - (B, N, 1, D) -> (B, N, N, D)

        value = value[:, None, :, :] # (B, 1, N, D)

        pos_emb = self.MLP_pos1(pos_rel) # (B, N, N, D)
        pos_emb = self.MLP_pos2(pos_emb) # (B, N, N, D)

        value = value + pos_emb # (B, N, N, D)

        mlp_attn1 = self.MLP_attn1(qk + pos_emb)
        if mask is not None:
            mask = tf.cast(mask, tf.bool) # Ensure mask is boolean
            key_mask = mask[:, :, None] # (B, N, 1)
            query_mask = mask[:, None, :] # (B, 1, N)
            attention_mask = tf.math.logical_and(key_mask, query_mask) # (B, N, N)
            attention_mask = tf.expand_dims(attention_mask, axis=-1) # (B, N, N, 1)
        else:
            attention_mask = tf.ones((tf.shape(feature)[0], n, n, 1), dtype=tf.bool) # (B, N, N, 1)
        softmax_mask = tf.where(attention_mask, 0.0, -1e9)  # Apply mask to attention logits

        mlp2_attn = self.MLP_attn2(mlp_attn1) + pos_emb + softmax_mask # (B, N, N, D)

        attn = tf.nn.softmax(mlp2_attn, axis=-2) # (B, N, N, D)
        out = value * attn # (B, N, N, D)
        out = tf.math.reduce_sum(out, axis=-2) # (B, N, D)
        out = self.linear2(out)

        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.linear2.units