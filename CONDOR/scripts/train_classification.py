import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn as sk
import sys

sys.path.append("../")
ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"{ROOT_DIR}/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"

SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_pixel_spacetime.npy"
BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_pixel_spacetime.npy"
SIGNAL_MPPC_FILE = f"{DATA_DIR}/sig_mppc_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_mppc_spacetime.npy"
SIGNAL_ONLY_PIXEL_FILE = f"{DATA_DIR}/sig_only_pixel_spacetime.npy"
SIGNAL_ONLY_MPPC_FILE = f"{DATA_DIR}/sig_only_mppc_spacetime.npy"

bg_pixel_spacetime = np.load(BACKGROUND_PIXEL_FILE)
bg_mppc_spacetime = np.load(BACKGROUND_MPPC_FILE)
sig_pixel_spacetime = np.load(SIGNAL_PIXEL_FILE)
sig_mppc_spacetime = np.load(SIGNAL_MPPC_FILE)

input_seq_len = bg_pixel_spacetime.shape[1]
input_dim = bg_pixel_spacetime.shape[2]  # Exclude timestamp

from src.model.components import (
    SelfAttentionStack,
    SelfAttentionBlock,
    CrossAttentionStack,
    CrossAttentionBlock,
    PoolingAttentionBlock,
    GenerateMask,
    MLP,
)

pixel_input = keras.Input(shape=(input_seq_len, input_dim), name="pixel_input")
mppc_input = keras.Input(shape=(input_seq_len, input_dim), name="mppc_input")

feature_dim = 8
num_heads = 6
dropout_rate = 0

pixel_mask = GenerateMask(name="mask")(pixel_input)
pixel_embedding = MLP(
    num_layers=4,
    output_dim=feature_dim,
    activation="relu",
    name="pixel_embedding",
    dropout_rate=dropout_rate,
)(pixel_input)

pixel_self_attention = SelfAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    stack_size = 3,
    name="pixel_self_attention",
    dropout_rate=dropout_rate,
    pre_ln=True,
)(pixel_embedding , mask = pixel_mask)
mppc_mask = GenerateMask(name="mppc_mask")(mppc_input)
mppc_embedding = MLP(
    num_layers=4,
    output_dim=feature_dim,
    activation="relu",
    name="mppc_embedding",
    dropout_rate=dropout_rate,
)(mppc_input)
mppc_self_attention = SelfAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    stack_size = 3,
    name="mppc_self_attention",
    dropout_rate=dropout_rate,
    pre_ln=True,
)(mppc_embedding , mask = mppc_mask)

pixel_attend_mppc, mppc_attend_pixel, _, _ = CrossAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    name="cross_attention",
    dropout_rate=dropout_rate,
    stack_size = 2,
    pre_ln=True,
)([pixel_self_attention, mppc_self_attention, pixel_mask, mppc_mask])


pixel_pooling = PoolingAttentionBlock(
    num_seeds = 1,
    key_dim=feature_dim,
    name="pooling_attention",
    dropout_rate=dropout_rate,
)(pixel_attend_mppc, mask=pixel_mask)

mppc_pooling = PoolingAttentionBlock(
    num_seeds = 1,
    key_dim=feature_dim,
    name="mppc_pooling_attention",
    dropout_rate=dropout_rate,
)(mppc_attend_pixel, mask=mppc_mask)


latent_space = keras.layers.Concatenate(name="latent_space")([pixel_pooling, mppc_pooling])
latent_space = keras.layers.Flatten(name="flatten")(latent_space)
output = MLP(
    num_layers=4,
    output_dim=1,
    activation="sigmoid",
    name="output",
    dropout_rate=dropout_rate,
)(latent_space)

model = keras.Model(
    inputs=[pixel_input, mppc_input],
    outputs=output,
    name="ClassificationModel",
)
from sklearn.model_selection import train_test_split

(
    X_pixel_train,
    X_pixel_test,
    X_mppc_train,
    X_mppc_test,
    y_train,
    y_test,
) = train_test_split(
    np.concatenate([bg_pixel_spacetime[:, :, :], sig_pixel_spacetime[:, :, :]], axis=0),
    np.concatenate([bg_mppc_spacetime[:, :, :], sig_mppc_spacetime[:, :, :]], axis=0),
    np.concatenate(
        [np.zeros(len(bg_pixel_spacetime)), np.ones(len(sig_pixel_spacetime))]
    ),
    test_size=0.2,
    random_state=42,
    shuffle=True,
)


model.compile(
    optimizer=keras.optimizers.Lion(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
)
model.fit(
    x=[X_pixel_train, X_mppc_train],
    y=y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=512,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ],
    class_weight = {label: 1/np.mean(y_train == label) for label in np.unique(y_train)}
)
keras.Model.save(model, f"{MODEL_DIR}/pre_train_transformer_embedding.keras")


model.compile(
    optimizer=keras.optimizers.Lion(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
)
model.fit(
    x=[X_pixel_train, X_mppc_train],
    y=y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=512,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ],
    class_weight = {label: 1/np.mean(y_train == label) for label in np.unique(y_train)}
)
keras.Model.save(model, f"{MODEL_DIR}/post_train_transformer_embedding.keras")


hahatest_seq_length = (X_pixel_test != -1).all(axis=-1).sum(axis=-1) + (
    X_mppc_test != -1
).all(axis=-1).sum(axis=-1)
test_mppc_length = (X_mppc_test != -1).all(axis=-1).sum(axis=-1)
test_pixel_length = (X_pixel_test != -1).all(axis=-1).sum(axis=-1)
train_mppc_length = (X_mppc_train != -1).all(axis=-1).sum(axis=-1)
train_pixel_length = (X_pixel_train != -1).all(axis=-1).sum(axis=-1)


mppc_lenght_input = keras.Input(shape=(1,), name="mppc_length_input")
pixel_length_input = keras.Input(shape=(1,), name="pixel_length_input")

input = keras.layers.Concatenate(name="input")(
    [
        mppc_lenght_input,
        pixel_length_input,
    ]
)
encoder = MLP(
    num_layers=3,
    output_dim=10,
    name="encoder",
    activation="relu",
)(input)
decoder = MLP(
    num_layers=3,
    output_dim=1,
    name="decoder",
    activation="sigmoid",
)(encoder)
seq_length_mlp = keras.Model(
    inputs=[mppc_lenght_input, pixel_length_input],
    outputs=decoder,
    name="SeqLengthMLP",
)

seq_length_mlp.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
)
seq_length_mlp.summary()


seq_length_mlp.fit(
    x=[train_mppc_length, train_pixel_length],
    y=y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ],
    class_weight={
        label: np.sum(y_train == label) / len(y_train)
        for label in np.unique(y_train)
        if label in [0, 1]
    },
)


test_predictions = model.predict([X_pixel_test, X_mppc_test])
test_seq_length = seq_length_mlp.predict([test_mppc_length, test_pixel_length])

from sklearn.metrics import confusion_matrix, roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, test_predictions)
fpr_seq_length, tpr_seq_length, thresholds_seq_length = roc_curve(
    y_test, test_seq_length
)
roc_auc_seq_length = auc(fpr_seq_length, tpr_seq_length)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="blue", label="ROC curve (area = {:.2f})".format(roc_auc))
ax.plot(
    fpr_seq_length,
    tpr_seq_length,
    color="green",
    label="MLP trained on number of hits of MPPC and Pixels (area = {:.2f})".format(
        roc_auc_seq_length
    ),
)
ax.plot([0, 1], [0, 1], color="red", linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curve")
ax.legend()
fig.savefig(f"{PLOTS_DIR}/roc_curve.png")
