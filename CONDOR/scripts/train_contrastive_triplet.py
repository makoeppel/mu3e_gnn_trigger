import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn as sk
import sys

sys.path.append("../")
ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"

SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_pixel_spacetime.npy"
BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_pixel_spacetime.npy"
SIGNAL_MPPC_FILE = f"{DATA_DIR}/sig_mppc_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_mppc_spacetime.npy"
SIGNAL_ONLY_PIXEL_FILE = f"{DATA_DIR}/sig_only_pixel_spacetime.npy"
SIGNAL_ONLY_MPPC_FILE = f"{DATA_DIR}/sig_only_mppc_spacetime.npy"

bg_pixel_spacetime = np.load(BACKGROUND_PIXEL_FILE)
sig_only_pixel_spacetime = np.load(SIGNAL_ONLY_PIXEL_FILE)
bg_mppc_spacetime = np.load(BACKGROUND_MPPC_FILE)
sig_only_mppc_spacetime = np.load(SIGNAL_ONLY_MPPC_FILE)
sig_pixel_spacetime = np.load(SIGNAL_PIXEL_FILE)
sig_mppc_spacetime = np.load(SIGNAL_MPPC_FILE)

seq_length = bg_pixel_spacetime.shape[1]
input_dim = bg_pixel_spacetime.shape[2]

from src.utils import ContrastSamples

(
    base_pixel,
    base_mppc,
    contrast_pixel_signal,
    contrast_mppc_signal,
    contrast_pixel_background,
    contrast_mppc_background,
) = ContrastSamples(
    bg_pixel_spacetime,
    sig_only_pixel_spacetime,
    bg_mppc_spacetime,
    sig_only_mppc_spacetime,
    num_samples=100000,
    padding_value=-1,
)

pixel_input = keras.layers.Input(
    shape=(seq_length, input_dim), name="pixel_input", dtype=tf.float32
)
mppc_input = keras.layers.Input(
    shape=(seq_length, input_dim), name="mppc_input", dtype=tf.float32
)


from src.model.components import (
    point_transformer,
    SelfAttentionStack,
    MultiHeadAttentionBlock,
    MLP,
    GenerateMask,
    PoolingAttentionBlock,
    MultiHeadAttentionStack,
)

feature_dim = 8
latent_dim = 16
num_heads = 8
num_seeds = 4
regularizer = keras.regularizers.l2(1e-4)
dropout_rate = 0.05

pixel_mask = GenerateMask(-1, name="pixel_mask")(pixel_input)
mppc_mask = GenerateMask(-1, name="mppc_mask")(mppc_input)

pixel_embedding = MLP(output_dim=feature_dim, name="pixel_embedding")(pixel_input)

mppc_embedding = MLP(output_dim=feature_dim, name="mppc_embedding")(mppc_input)

pixel_attention = SelfAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    stack_size=3,
    name="pixel_attention",
)(pixel_embedding, pixel_mask)

mppc_attention = SelfAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    stack_size=3,
    name="mppc_attention",
)(mppc_embedding, mppc_mask)

mppc_attend_pixel = MultiHeadAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    stack_size=3,
    name="mppc_attend_pixel",
)(
    query=mppc_attention,
    value=pixel_attention,
    query_mask=mppc_mask,
    value_mask=pixel_mask,
    key_mask=pixel_mask,
)

pixel_attend_mppc = MultiHeadAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    stack_size=3,
    name="pixel_attend_mppc",
)(
    query=pixel_attention,
    value=mppc_attention,
    query_mask=pixel_mask,
    value_mask=mppc_mask,
    key_mask=mppc_mask,
)

pixel_attentions_pool = PoolingAttentionBlock(
    name="pixel_attentions_pool",
    key_dim=feature_dim,
    num_seeds=num_seeds,
    num_heads=num_heads,
)(pixel_attend_mppc, pixel_mask)

pixel_flattened_pool = keras.layers.Flatten(name="pixel_flattened_pool")(
    pixel_attentions_pool
)

mppc_attentions_pool = PoolingAttentionBlock(
    name="mppc_attentions_pool",
    key_dim=feature_dim,
    num_seeds=num_seeds,
    num_heads=num_heads,
)(mppc_attend_pixel, mppc_mask)

mppc_flattened_pool = keras.layers.Flatten(name="mppc_flattened_pool")(
    mppc_attentions_pool
)

latent_space = keras.layers.Concatenate(name="latent_space")(
    [
        pixel_flattened_pool,
        mppc_flattened_pool,
    ]
)

latent_output = MLP(
    num_layers=4,
    output_dim=latent_dim,
    name="latent_output",
    activation="linear",
)(latent_space)

transformer_embedding = keras.Model(
    inputs=[pixel_input, mppc_input],
    outputs=latent_output,
    name="contrastive_learning_model",
)

from src.model.wrapper.Siamese import make_siamese_encoder

siamese_model = make_siamese_encoder(
    input_shapes=[(seq_length, input_dim), (seq_length, input_dim)],
    base_model=transformer_embedding,
    num_contrastive_views=3,
)


from src.training import TripletLoss

siamese_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=TripletLoss(margin=1.0),
)

siamese_model.fit(
    x=[
        base_pixel,
        base_mppc,
        contrast_pixel_background,
        contrast_mppc_background,
        contrast_pixel_signal,
        contrast_mppc_signal,
    ],
    y=np.zeros(contrast_mppc_signal.shape[0]),  # Dummy labels
    validation_split=0.2,
    epochs=30,
    batch_size=128,
)

transformer_embedding.save(f"{MODEL_DIR}/transformer_embedding.keras")

from sklearn.model_selection import train_test_split

bg_pixel_train, bg_pixel_test, big_mppc_train, bg_mppc_test = train_test_split(
    bg_pixel_spacetime, bg_mppc_spacetime, test_size=0.2, random_state=42
)
sig_pixel_train, sig_pixel_test, sig_mppc_train, sig_mppc_test = train_test_split(
    sig_pixel_spacetime, sig_mppc_spacetime, test_size=0.2, random_state=42
)

signal_latent = transformer_embedding.predict([sig_pixel_test, sig_mppc_test])
background_latent = transformer_embedding.predict([bg_pixel_test, bg_mppc_test])

from src.evaluation import plot_latent_variable_distributions

fig, axes = plot_latent_variable_distributions(signal_latent, background_latent)
fig.savefig(f"{PLOTS_DIR}/latent_variable_distributions.png")
