import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append("../")
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Dataset
from tqdm import tqdm

ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"/data/dust/group/atlas/ttreco/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"
SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_only_with_layer_pixel_spacetime.npy"
SIGNAL_MPPC_FILE = f"{DATA_DIR}/sig_only_with_layer_mppc_spacetime.npy"

BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_with_layer_pixel_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_with_layer_mppc_spacetime.npy"



bg_pixel_spacetime = np.load(BACKGROUND_PIXEL_FILE)
bg_mppc_spacetime = np.load(BACKGROUND_MPPC_FILE)
sig_pixel_spacetime = np.load(SIGNAL_PIXEL_FILE)
sig_mppc_spacetime = np.load(SIGNAL_MPPC_FILE)

from src.utils import get_spacetime_data
bg_pixel_spacetime = get_spacetime_data(bg_pixel_spacetime)
bg_mppc_spacetime = get_spacetime_data(bg_mppc_spacetime)
sig_pixel_spacetime = get_spacetime_data(sig_pixel_spacetime)
sig_mppc_spacetime = get_spacetime_data(sig_mppc_spacetime)


import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


sequence_length = sig_pixel_spacetime.shape[1]
input_feature_dim = sig_mppc_spacetime.shape[2]
feature_dim = 16

sequence_length = sig_pixel_spacetime.shape[1]
input_feature_dim = sig_mppc_spacetime.shape[2]
feature_dim = 16

from src.keras.model.components import SelfAttentionStack, MLP, GenerateMask

encoder_pixel_input = keras.Input(
    shape=(sequence_length, input_feature_dim), name="encoder_pixel_input"
)
encoder_pixel_mask = GenerateMask(padding_value=-1)(encoder_pixel_input)

encoder_mppc_input = keras.Input(
    shape=(sequence_length, input_feature_dim), name="encoder_mppc_input"
)
encoder_mppc_mask = GenerateMask(padding_value=-1)(encoder_mppc_input)

encoder_pixel_embedding = MLP(16, 2, activation="relu", name = "pixel_embedding")(
    encoder_pixel_input
)
encoder_mppc_embedding = MLP(16, 2, activation="relu", name = "mppc_embedding")(
    encoder_mppc_input
)


encoder_pixel_output = SelfAttentionStack(
    key_dim=feature_dim,
    stack_size=2,
    num_heads=4,
    dropout_rate=0.1,
)(encoder_pixel_embedding, encoder_pixel_mask)

encoder_mppc_output = SelfAttentionStack(
    key_dim=feature_dim,
    stack_size=2,
    num_heads=4,
    dropout_rate=0.1,
)(encoder_mppc_embedding, encoder_mppc_mask)

pooled_pixel_output = keras.layers.GlobalAveragePooling1D()(
    encoder_pixel_output
)
pooled_mppc_output = keras.layers.GlobalAveragePooling1D()(
    encoder_mppc_output
)

pooled_concat = keras.layers.Concatenate()(
    [pooled_pixel_output, pooled_mppc_output]
)

encoded_output = MLP(16, 3, activation="relu", name = "encoded_output")(pooled_concat)


encoder_input = [encoder_pixel_input, encoder_mppc_input]



encoder = keras.Model(
    inputs=encoder_input,
    outputs=encoded_output,
    name="encoder",
)

from src.keras.model import AutoEncoder

autoencoder_input = keras.Input(shape=(16,), name="autoencoder_input")
autoencoder = MLP(input_feature_dim * sequence_length, 3, activation="relu")(autoencoder_input)


from src.keras.training import MultiObjectiveTrainer
from src.keras.training import VarianceCovarianceLoss

trainer = MultiObjectiveTrainer(
    encoder=encoder,
    autoencoder=autoencoder,
    lambda_var=1.0,
    variance_loss = VarianceCovarianceLoss(cov_penalty= 1/25.)
)
encoder_optimizer = keras.optimizers.Adam(learning_rate=0.001)
ae_optimizer = keras.optimizers.Adam(learning_rate=0.01)


from sklearn.model_selection import train_test_split

pixel_train, pixel_test, mppc_train, mppc_test = train_test_split(
    bg_pixel_spacetime,
    bg_mppc_spacetime,
    test_size=0.2,
    random_state=42,
)
train_dataset = tf.data.Dataset.from_tensor_slices(
    ([pixel_train, mppc_train])
).shuffle(buffer_size=1024).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices(
    ([pixel_test, mppc_test])
).batch(32)

for epoch in range(10):
    print(f"Epoch {epoch+1}")
    trainer.train_autoencoder_step(
        train_dataset, ae_optimizer, num_steps=10
    )
    
    for _ in range(5):
        trainer.train_encoder_variance_step(
            train_dataset, encoder_optimizer
        )

    for _ in range(5):
        trainer.train_encoder_step(
            test_dataset, encoder_optimizer
        )
    
    for _ in range(5):
        trainer.train_reconstruction_step(
            test_dataset, encoder_optimizer
        )



encoded_test = encoder.predict([pixel_test, mppc_test])
encoded_sig = encoder.predict([sig_pixel_spacetime, sig_mppc_spacetime])

autoencoder_test = autoencoder.predict(encoded_test)
autoencoder_sig = autoencoder.predict(encoded_sig)

norm_error_test = np.linalg.norm(encoded_test - autoencoder_test, axis=1)
norm_error_sig = np.linalg.norm(encoded_sig - autoencoder_sig, axis=1)

plt.hist(norm_error_test, bins=50, alpha=0.5, label="Background")
plt.hist(norm_error_sig, bins=50, alpha=0.5, label="Signal")
plt.yscale("log")
plt.xlabel("Reconstruction Error (L2 Norm)")
plt.ylabel("Counts")
plt.legend()
plt.show()


from src.plotting.evaluation import plot_latent_variable_distributions

fig, ax = plot_latent_variable_distributions(
    encoded_test,
    encoded_sig
)
fig.savefig("latent_variable_distributions.png")