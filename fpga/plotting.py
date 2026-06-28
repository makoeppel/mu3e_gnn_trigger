import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_event(event, name):
    """
    event: ndarray of shape (4, 256)
           rows = [x, y, z, t]
    """

    event = event.astype(float)

    # Hide missing hits
    event[event == 0] = np.nan

    vmax = np.nanmax(np.abs(event))

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax,
    )

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("black")      # 0 -> black

    plt.figure(figsize=(14,3))

    plt.imshow(
        event,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    plt.yticks(
        [0,1,2,3],
        ["x","y","z","t"]
    )

    plt.xlabel("Hit index")
    plt.colorbar(label="Coordinate / Time")
    plt.tight_layout()
    plt.savefig(name)

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
