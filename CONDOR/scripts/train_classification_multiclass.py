import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn as sk
import sys
import seaborn as sns
import os
sys.path.append("../")
ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"{ROOT_DIR}/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"
MODEL_NAME = "multi_class_classification"

os.makedirs(f"{MODEL_DIR}/{MODEL_NAME}", exist_ok=True)


SIGNAL_ONLY_PIXEL_FILE = f"{DATA_DIR}/sig_only_pixel_spacetime.npy"
SIGNAL_ONLY_MPPC_FILE = f"{DATA_DIR}/sig_only_mppc_spacetime.npy"
E5_PIXEL_FILE = f"{DATA_DIR}/5e_pixel_spacetime.npy"
E5_MPPC_FILE = f"{DATA_DIR}/5e_mppc_spacetime.npy"
FAMILON_PIXEL_FILE = f"{DATA_DIR}/familon_pixel_spacetime.npy"
FAMILON_MPPC_FILE = f"{DATA_DIR}/familon_mppc_spacetime.npy"
BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_pixel_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_mppc_spacetime.npy"

bg_pixel_spacetime = np.load(BACKGROUND_PIXEL_FILE)
bg_mppc_spacetime = np.load(BACKGROUND_MPPC_FILE)
sig_only_pixel_spacetime = np.load(SIGNAL_ONLY_PIXEL_FILE)
sig_only_mppc_spacetime = np.load(SIGNAL_ONLY_MPPC_FILE)
e5_pixel_spacetime = np.load(E5_PIXEL_FILE)
e5_mppc_spacetime = np.load(E5_MPPC_FILE)
familon_pixel_spacetime = np.load(FAMILON_PIXEL_FILE)
familon_mppc_spacetime = np.load(FAMILON_MPPC_FILE)

num_classes = 4

class_names = [
    "Background",
    "Signal",
    "5e",
    "Familon",
]


def make_multi_class_dataset(data_list: list[tuple]):
    X_pixel = np.concatenate([data[0] for data in data_list], axis=0)
    X_mppc = np.concatenate([data[1] for data in data_list], axis=0)
    y = np.concatenate(
        [np.full(len(data[0]), i) for i, data in enumerate(data_list)], axis=0
    )
    y = keras.utils.to_categorical(y, num_classes=num_classes)
    return X_pixel, X_mppc, y


X_pixel, X_mppc, y = make_multi_class_dataset(
    [
        (bg_pixel_spacetime, bg_mppc_spacetime),
        (sig_only_pixel_spacetime, sig_only_mppc_spacetime),
        (e5_pixel_spacetime, e5_mppc_spacetime),
        (familon_pixel_spacetime, familon_mppc_spacetime),
    ]
)


input_seq_len = sig_only_pixel_spacetime.shape[1]
input_dim = sig_only_pixel_spacetime.shape[2]

pixel_input = keras.Input(shape=(input_seq_len, input_dim), name="pixel_input")
mppc_input = keras.Input(shape=(input_seq_len, input_dim), name="mppc_input")


from src.model.components import (
    SelfAttentionStack,
    SelfAttentionBlock,
    CrossAttentionStack,
    PoolingAttentionBlock,
    GenerateMask,
    MLP,
)

feature_dim = 16
num_heads = 8
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
    stack_size=2,
    name="pixel_self_attention",
    dropout_rate=dropout_rate,
    pre_ln=True,
)(pixel_embedding, mask=pixel_mask)
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
    stack_size=2,
    name="mppc_self_attention",
    dropout_rate=dropout_rate,
    pre_ln=True,
)(mppc_embedding, mask=mppc_mask)


pixel_attend_mppc, mppc_attend_pixel = CrossAttentionStack(
    num_heads=num_heads,
    key_dim=feature_dim,
    stack_size=2,
    name="cross_attention",
    dropout_rate=dropout_rate,
    pre_ln=True,
)(pixel_self_attention, mppc_self_attention, 
    a_mask=pixel_mask, b_mask=mppc_mask)


pixel_pooling = PoolingAttentionBlock(
    num_seeds=1,
    key_dim=feature_dim,
    name="pooling_attention",
    dropout_rate=dropout_rate,
)(pixel_attend_mppc, mask=pixel_mask)

mppc_pooling = PoolingAttentionBlock(
    num_seeds=1,
    key_dim=feature_dim,
    name="mppc_pooling_attention",
    dropout_rate=dropout_rate,
)(mppc_attend_pixel, mask=mppc_mask)


latent_space = keras.layers.Concatenate(name="latent_space")(
    [pixel_pooling, mppc_pooling]
)
latent_space = keras.layers.Flatten(name="flatten")(latent_space)
output = MLP(
    num_layers=4,
    output_dim=num_classes,
    activation="softmax",
    name="output",
    dropout_rate=dropout_rate,
)(latent_space)

model = keras.Model(
    inputs=[pixel_input, mppc_input],
    outputs=output,
    name="ClassificationModel",
)

model.compile(
    optimizer=keras.optimizers.Lion(
        learning_rate = 1e-4
    ),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()],
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
    X_pixel,
    X_mppc,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

class_weights = {
    label: 1 / np.mean(y_train.argmax(axis=1) == label)
    for label in np.unique(y_train.argmax(axis=1))
}

model.fit(
    x=[X_pixel_train, X_mppc_train],
    y=y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_DIR}/{MODEL_NAME}/" + "{epoch:02d}-{val_loss:.2f}.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ],
)


test_predictions = model.predict([X_pixel_test, X_mppc_test])

from sklearn.metrics import confusion_matrix, roc_curve, auc

confusion_matrix_result = confusion_matrix(
    y_test.argmax(axis=1),
    test_predictions.argmax(axis=1),
    labels=np.arange(num_classes),
)
normed_confusion_matrix = confusion_matrix_result.astype("float") / (
    confusion_matrix_result.sum(axis=1)[:, np.newaxis] + 1e-6
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    normed_confusion_matrix,
    annot=True,
    cmap="Blues",
    cbar=False,
    ax=ax,
    square=True,
    linewidths=0.5,
    linecolor="black",
    annot_kws={"size": 16},
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
ax.set_xticks(np.arange(num_classes) + 0.5)
ax.set_yticks(np.arange(num_classes) + 0.5)
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlim(0, num_classes)
ax.set_ylim(0, num_classes)
ax.xaxis.set_ticks_position("bottom")
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=45)
fig.savefig(f"{PLOTS_DIR}/confusion_matrix.png")
