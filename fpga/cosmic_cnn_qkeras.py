import numpy as np
import keras
from qkeras import QConv2D, QDense, QActivation, quantized_bits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt


def create_cnn_matrix_dataset(beam_arr, cosmic_arr, cosmic_mix_fraction=0.5, max_hits=256, seed=42, use_time=True):
    np.random.seed(seed)
    num_beam_events = beam_arr.shape[0]
    num_cosmic_events = cosmic_arr.shape[0]

    matrices = []
    labels = []

    mix_mask = np.random.rand(num_beam_events) < cosmic_mix_fraction
    cosmic_indices_pool = list(np.random.permutation(num_cosmic_events))
    maxing = True

    for i in range(num_beam_events):
        beam_event = beam_arr[i]
        valid_beam = beam_event[beam_event[:, 0] != -999.0]
        if not use_time: valid_beam = valid_beam[:,:3]

        if mix_mask[i] and maxing:
            if len(cosmic_indices_pool) == 0:
                print(f"Warning: Out of unique cosmic events at beam index {i}. Stopping mix allocation.")
                maxing = False
                continue

            rand_cos_idx = cosmic_indices_pool.pop()

            cosmic_event = cosmic_arr[rand_cos_idx]
            valid_cosmic = cosmic_event[cosmic_event[:, 0] != -999.0]

            if not use_time: valid_cosmic = valid_cosmic[:,:3]

            if len(valid_cosmic) >= 4:

                combined_hits = np.random.permutation(
                    np.vstack([valid_beam, valid_cosmic])
                )
                label = 1.0  
            else:
                combined_hits = valid_beam
                label = 0.0
        else:
            combined_hits = valid_beam
            label = 0.0

        if len(combined_hits) == 0:
            continue

        if len(combined_hits) > max_hits:
            combined_hits = combined_hits[:max_hits]

        if use_time: event_matrix = np.zeros((1, max_hits, 4))
        if not use_time: event_matrix = np.zeros((1, max_hits, 3))

        num_actual_hits = combined_hits.shape[0]
        # Assign hits directly into the width timeline
        event_matrix[0, :num_actual_hits, :] = combined_hits

        matrices.append(event_matrix)
        labels.append(label)

    X = np.array(matrices)
    y = np.array(labels)

    return X, y


def create_qkeras_model(max_hits=256, num_channels=[8, 16, 32], dense_dim=32, input_quant='quantized_bits(8,3)', use_time=True):
    q_bits = quantized_bits(8, 3, alpha=1)

    if use_time: inputs = keras.layers.Input(shape=(1, max_hits, 4))
    if not use_time: inputs = keras.layers.Input(shape=(1, max_hits, 3))
    x = QActivation(input_quant, name='q_input_activation')(inputs)

    for i, filters in enumerate(num_channels):
        layer_idx = i + 1
        x = QConv2D(
            filters=filters, 
            kernel_size=(1, 5), 
            padding='same',
            kernel_quantizer=q_bits,
            bias_quantizer=q_bits,
            name=f'q_conv{layer_idx}'
        )(x)
        x = QActivation('quantized_relu(8, 3)', name=f'q_relu{layer_idx}')(x)
        x = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', name=f'q_pool{layer_idx}')(x)

    x = keras.layers.Flatten()(x)
    x = QDense(dense_dim, kernel_quantizer=q_bits, bias_quantizer=q_bits, name='q_dense1')(x)
    x = QActivation('quantized_relu(8, 3)', name=f'q_relu_dense')(x)

    outputs = QDense(1, kernel_quantizer=q_bits, bias_quantizer=q_bits, activation='sigmoid', name='q_output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"Quantized_Spacetime_{len(num_channels)}Layer_CNN")
    return model

def plot_efficiency_vs_acceptance(model, X_test, y_test, max_hits, num_params):
    preds = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, preds)

    plt.figure(figsize=(10, 5))
    plt.scatter(
        fpr, tpr,
        color='#E24A33',
        marker='D',
        s=40, 
        label=f'QKeras Model ({num_params:,} parameters)'
    )

    plt.xscale('log')
    plt.xlim(5e-5, 1e-1)
    plt.ylim(0.0, 1.1)

    plt.xlabel('Background Acceptance', fontsize=11)
    plt.ylabel('Reconstruction Efficiency', fontsize=11)
    plt.title(f'QKeras Quantized Reconstruction Efficiency (max_hits={max_hits})', fontsize=13, pad=10)

    plt.grid(True, which="both", linestyle='-', alpha=0.5)
    plt.legend(loc='lower left', frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig("recon_plot.pdf")

def plot_multiple_events_xy(X_batch, y_batch, num_events=20):
    plt.figure(figsize=(7, 7))

    beam_label_added = False
    cosmic_label_added = False

    for idx in range(min(num_events, len(X_batch))):
        x_coords = X_batch[idx, 0, :, 0]
        y_coords = X_batch[idx, 0, :, 1]

        mask = (x_coords != 0.0) | (y_coords != 0.0)
        valid_x = x_coords[mask]
        valid_y = y_coords[mask]

        if y_batch[idx] == 1.0:
            color = 'blue'
            label = 'Cosmic Event Mix' if not cosmic_label_added else ""
            cosmic_label_added = True
        else:
            color = 'red'
            label = 'Pure Beam Event' if not beam_label_added else ""
            beam_label_added = True

        plt.scatter(valid_x, valid_y, color=color, marker='o', s=12, alpha=0.5, label=label)

    plt.xlabel('X Position', fontsize=11)
    plt.ylabel('Y Position', fontsize=11)
    plt.title(f'Accumulated 2D Hit Layout (First {num_events} Events)', fontsize=12, pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("cnn_matrix_xy.png")

def plot_raw_data_xy(beam_arr, cosmic_arr, num_events=20):
    plt.figure(figsize=(7, 7))

    beam_label_added = False
    for i in range(min(num_events, len(beam_arr))):
        event = beam_arr[i]
        valid_hits = event[event[:, 0] != -999.0]
        if len(valid_hits) > 0:
            label = 'Raw Beam Data' if not beam_label_added else ""
            beam_label_added = True
            plt.scatter(valid_hits[:, 0], valid_hits[:, 1], color='red', marker='o', s=12, alpha=0.4, label=label)

    cosmic_label_added = False
    for i in range(min(num_events, len(cosmic_arr))):
        event = cosmic_arr[i]
        valid_hits = event[event[:, 0] != -999.0]
        if len(valid_hits) > 0:
            label = 'Raw Cosmic Data' if not cosmic_label_added else ""
            cosmic_label_added = True
            plt.scatter(valid_hits[:, 0], valid_hits[:, 1], color='blue', marker='x', s=14, alpha=0.4, label=label)

    plt.xlabel('X Position', fontsize=11)
    plt.ylabel('Y Position', fontsize=11)
    plt.title(f'Raw Space Coordinates Overlay (First {num_events} Events)', fontsize=12, pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("raw_xy.png")

def plot_auc_curve(model, X_test, y_test):
    preds = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Acceptance)', fontsize=11)
    plt.ylabel('True Positive Rate (Signal Efficiency)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, pad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig("auc.pdf")

if __name__ == "__main__":
    TUNABLE_MAX_HITS = 256
    TUNABLE_CONV_CHANNELS = [8]
    TUNABLE_DENSE_DIM = 8
    TUNABLE_USE_TIME = True

    print("Setting up tracking datasets...")
    beam_path = "../data/beam100k_pixel_spacetime.npy"
    cosmic_path = "../data/cosmic100k_pixel_spacetime.npy"
    sig_path = "../data/sig30k_pixel_spacetime.npy"

    beam_data = np.load(beam_path)
    cosmic_data = np.load(cosmic_path)

    plot_raw_data_xy(beam_data, cosmic_data)

    print(f"Loaded Beam Array: {beam_data.shape} | Cosmic Array: {cosmic_data.shape}")

    X, y = create_cnn_matrix_dataset(beam_data, cosmic_data, cosmic_mix_fraction=1, max_hits=TUNABLE_MAX_HITS, use_time=TUNABLE_USE_TIME)
    plot_multiple_events_xy(X, y)
    print("Ratio cosmic", sum(y)/len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_qkeras_model(
        max_hits=TUNABLE_MAX_HITS, 
        num_channels=TUNABLE_CONV_CHANNELS, 
        dense_dim=TUNABLE_DENSE_DIM,
        use_time=TUNABLE_USE_TIME
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    model.summary()
    total_params = model.count_params()

    print("\nBeginning QKeras Quantization Aware Training Execution loop...")
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=1
    )

    print("\nGenerating final tracking evaluation profile...")
    plot_efficiency_vs_acceptance(model, X_test, y_test, TUNABLE_MAX_HITS, total_params)
    plot_auc_curve(model, X_test, y_test)
