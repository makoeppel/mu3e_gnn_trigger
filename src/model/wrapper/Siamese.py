import numpy as np
import tensorflow as tf
import keras

def make_siamese_autoencoder(input_shape, base_model : keras.Model, num_contrastive_views=4):
    """
    Create a Siamese autoencoder model that applies the base model to multiple views of the input data.
    
    Args:
        model (keras.Model): The base model to use for predictions.
        num_augmentations (int): Number of augmentations to apply.
        latent_output (keras.layers.Layer): Optional layer to extract latent output.

    Returns:
        keras.Model: A new model that outputs predictions and optionally latent representations.
    """
    views = [keras.layers.Input(shape=input_shape, name=f"view_{i}") for i in range(num_contrastive_views)]
    # Apply the base model to each view
    outputs = [base_model(view) for view in views]
    # Concatenate the latent spaces
    concatenated_latent = keras.layers.Concatenate(axis=-1, name="concatenated_latent")(outputs)

    # Create the Siamese model
    siamese_model = keras.Model(inputs=views, outputs=concatenated_latent, name="SiamesePredictionModel")

    return siamese_model