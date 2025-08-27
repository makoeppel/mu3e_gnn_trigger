import keras
import tensorflow as tf
from keras import regularizers


class MLP(keras.layers.Layer):
    def __init__(
        self,
        output_dim,
        num_layers=3,
        layer_norm=False,
        activation="linear",
        hidden_activation="relu",
        dropout_rate=0,
        name="mlp_dense",
        **kwargs,
    ):
        self.regularizer = kwargs.pop("regularizer", None)
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.activation = activation
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.name = name
        self.layers_list = []  # Populated in build()

    def build(self, input_shape):
        input_dim = input_shape[-1]
        input_shape_list = []
        node_list = [
            int(input_dim * ((self.output_dim / input_dim) ** (i / self.num_layers)))
            for i in range(1, self.num_layers)
        ]
        for i, width in enumerate(node_list):
            self.layers_list.append(
                keras.layers.Dense(
                    width,
                    activation=self.hidden_activation,
                    name=f"{self.name}_mlp_dense_{i}",
                    kernel_regularizer=self.regularizer,
                    bias_regularizer=self.regularizer,
                    activity_regularizer=self.regularizer,
                )
            )
            if i == 0:
                input_shape_list.append((input_shape[:-1], input_dim))
            else:
                input_shape_list.append((input_shape[:-1], node_list[i - 1]))
            if self.layer_norm:
                self.layers_list.append(
                    keras.layers.LayerNormalization(
                        name=f"{self.name}_mlp_layer_norm_{i}"
                    )
                )
                input_shape_list.append((input_shape[:-1], width))
            if self.dropout_rate > 0:
                self.layers_list.append(
                    keras.layers.Dropout(
                        self.dropout_rate, name=f"{self.name}_mlp_dropout_{i}"
                    )
                )
                input_shape_list.append((input_shape[:-1], width))

        self.layers_list.append(
            keras.layers.Dense(
                self.output_dim,
                activation=self.activation,
                name=f"{self.name}_mlp_dense_output",
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer,
                activity_regularizer=self.regularizer,
            )
        )
        input_shape_list.append((input_shape[:-1], node_list[-1]))
        if self.layer_norm:
            self.layers_list.append(
                keras.layers.LayerNormalization(
                    name=f"{self.name}_mlp_layer_norm_output"
                )
            )
            input_shape_list.append((input_shape[:-1], self.output_dim))
        if self.dropout_rate > 0:
            self.layers_list.append(
                keras.layers.Dropout(
                    self.dropout_rate, name=f"{self.name}_mlp_dropout_output"
                )
            )
            input_shape_list.append((input_shape[:-1], self.output_dim))
        # Build all layers
        for iter, layer in enumerate(self.layers_list):
            layer.build(input_shape_list[iter])

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_list:
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "layer_norm": self.layer_norm,
                "activation": self.activation,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    def count_params(self):
        return sum(layer.count_params() for layer in self.layers_list)
