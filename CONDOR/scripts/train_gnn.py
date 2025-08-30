import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append("../")
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Dataset
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"/data/dust/group/atlas/ttreco/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"
SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_only_with_layer_pixel_spacetime.npy"
SIGNAL_MPPC_FILE = f"{DATA_DIR}/sig_only_with_layer_mppc_spacetime.npy"

BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_with_layer_pixel_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_with_layer_mppc_spacetime.npy"


sig_mppc_spacetime = np.load(SIGNAL_MPPC_FILE)
sig_pixel_spacetime = np.load(SIGNAL_PIXEL_FILE)
bg_pixel_spacetime = np.load(BACKGROUND_PIXEL_FILE)
bg_mppc_spacetime = np.load(BACKGROUND_MPPC_FILE)

X_pixel = np.concatenate([sig_pixel_spacetime, bg_pixel_spacetime], axis=0)
X_mppc = np.concatenate([sig_mppc_spacetime, bg_mppc_spacetime], axis=0)
y = np.concatenate(
    [np.ones(sig_pixel_spacetime.shape[0]), np.zeros(bg_pixel_spacetime.shape[0])],
    axis=0,
)


# --------------------------------
# Functions to convert MPPC and pixel IDs to positions in space
# --------------------------------
from sklearn.model_selection import train_test_split

X_pixel_train, X_pixel_val, X_mppc_train, X_mppc_val, y_train, y_val = train_test_split(
    X_pixel, X_mppc, y, test_size=0.2, random_state=42, stratify=y
)
del X_pixel, X_mppc, y

from src.torch.pre_processing import EventProcessor, CombinedGraphBuilder
event_processor = EventProcessor(CombinedGraphBuilder(connect_layers=True, mppc_timing_cutoff= 0.1))
train_graphs = event_processor.process_to_graphs(X_pixel = X_pixel_train,X_mppc = X_mppc_train,labels =  y_train)
val_graphs = event_processor.process_to_graphs(X_pixel = X_pixel_val,X_mppc = X_mppc_val,labels =  y_val)

# Clean up memory
del (
    X_pixel_train,
    X_pixel_val,
    X_mppc_train,
    X_mppc_val,
    y_train,
    y_val,
)
del sig_pixel_spacetime, sig_mppc_spacetime, bg_pixel_spacetime, bg_mppc_spacetime

graph_size = np.array([g.num_nodes for g in train_graphs + val_graphs])
graph_labels = np.array([g.y.item() for g in train_graphs + val_graphs])
x_max = np.percentile(graph_size, 95).astype(int)
x_max = (x_max // 2 + 1) * 2  # Round up to nearest 2
bins = np.linspace(0, x_max, x_max // 2 + 1)
fig, ax = plt.subplots(
    figsize=(8, 5),
)
ax.hist(
    graph_size[graph_labels == 0],
    bins=bins,
    alpha=0.5,
    label="Background",
    color="blue",
    density=True,
)
ax.hist(
    graph_size[graph_labels == 1],
    bins=bins,
    alpha=0.5,
    label="Signal",
    color="orange",
    density=True,
)
ax.set_xlabel("Number of nodes in graph")
ax.set_ylabel("Density")
ax.set_title("Distribution of graph sizes")
ax.legend()
plt.savefig(f"{PLOTS_DIR}/graph_size_distribution.png")


from src.torch.model.graph_classifier import SimpleGraphClassifier
from importlib import reload
from src.torch.training import get_class_weights
model = SimpleGraphClassifier(
    node_input_dim=train_graphs[0].x.shape[1],
    hidden_dim=48,
    num_conv_layers=7,
    dropout=0.2,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)
weight=get_class_weights(train_graphs).float().to(device)
bce_loss = torch.nn.BCELoss().to(device)


from src.torch.training import train_graph_classifier
train_loader = DataLoader(train_graphs, batch_size=512, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=512, shuffle=False)

model, aucs = train_graph_classifier(
    train_loader, val_loader,model, num_epochs=50, optimizer=optimizer, scheduler=scheduler, criterion=bce_loss, MODEL_DIR=MODEL_DIR, MODEL_NAME="graph_classifier", device=device
)


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(aucs["train_aucs"], label="Train AUC")
ax.plot(aucs["val_aucs"], label="Validation AUC")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("Training and Validation AUC over Epochs")
ax.legend()
plt.savefig(f"{PLOTS_DIR}/graph_classifier_aucs.png")


from sklearn.metrics import roc_curve, auc
fig, ax = plt.subplots(figsize=(8, 5))
val_labels = []
val_preds = []
model.eval()
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating on validation set"):
        out = model(batch)
        val_labels.append(batch.y.cpu())
        val_preds.append(out.cpu())
val_labels = torch.cat(val_labels).numpy()
val_preds = torch.cat(val_preds).numpy()
fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend(loc="lower right")
plt.savefig(f"{PLOTS_DIR}/graph_classifier_roc_curve.png")
