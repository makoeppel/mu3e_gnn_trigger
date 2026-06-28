import keras
from qkeras import QDense, QActivation, quantized_bits


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z and adds the KL divergence loss."""

    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch = keras.ops.shape(z_mean)[0]
        dim = keras.ops.shape(z_mean)[1]

        epsilon = keras.random.normal((batch, dim))

        z = z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon

        kl_loss = -0.5 * keras.ops.mean(
            keras.ops.sum(
                1.0
                + z_log_var
                - keras.ops.square(z_mean)
                - keras.ops.exp(z_log_var),
                axis=-1,
            )
        )

        self.add_loss(kl_loss)

        return z


def create_vae(
    encoding_dim=3,
    layers=(8,),
    input_quant="quantized_bits(8,3)",
):
    q_bits = quantized_bits(8, 3, alpha=1)

    ####################################################################
    # Encoder
    ####################################################################

    inputs = keras.layers.Input(shape=(1,), name="vae_input")

    x = QActivation(input_quant, name="q_input_activation")(inputs)

    for i, dim in enumerate(layers):
        x = QDense(
            dim,
            kernel_quantizer=q_bits,
            bias_quantizer=q_bits,
            name=f"q_dense_en{i+1}",
        )(x)

        x = QActivation(
            "quantized_relu(8,3)",
            name=f"q_relu_en{i+1}",
        )(x)

    z_mean = QDense(
        encoding_dim,
        kernel_quantizer=q_bits,
        bias_quantizer=q_bits,
        name="z_mean",
    )(x)

    latent_model = keras.Model(inputs, z_mean)

    z_log_var = QDense(
        encoding_dim,
        kernel_quantizer=q_bits,
        bias_quantizer=q_bits,
        name="z_log_var",
    )(x)

    z = Sampling(name="z_sampling")([z_mean, z_log_var])

    ####################################################################
    # Decoder
    ####################################################################

    x = z

    for i, dim in enumerate(reversed(layers)):
        x = QDense(
            dim,
            kernel_quantizer=q_bits,
            bias_quantizer=q_bits,
            name=f"q_dense_de{i+1}",
        )(x)

        x = QActivation(
            "quantized_relu(8,3)",
            name=f"q_relu_de{i+1}",
        )(x)

    x = QDense(
        1,
        kernel_quantizer=q_bits,
        bias_quantizer=q_bits,
        name="dec_final_dense",
    )(x)

    outputs = QActivation(
        "quantized_relu(8,3)",
        name="vae_output",
    )(x)

    ####################################################################
    # Model
    ####################################################################

    vae = keras.Model(inputs, outputs, name="Quantized_VAE")

    vae.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
    )

    return vae, latent_model