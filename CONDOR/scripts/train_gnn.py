import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import os
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, EdgeConv, DynamicEdgeConv
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy


sys.path.append("../")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"/data/dust/group/atlas/ttreco/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"
MODEL_NAME = "classification_single_seq"

os.makedirs(f"{MODEL_DIR}/{MODEL_NAME}", exist_ok=True)

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


def get_labelled_points(
    bg_pixel_spacetime, bg_mppc_spacetime, sig_pixel_spacetime, sig_mppc_spacetime
):
    X_pixel = np.concatenate((bg_pixel_spacetime, sig_pixel_spacetime), axis=0)
    X_mppc = np.concatenate((bg_mppc_spacetime, sig_mppc_spacetime), axis=0)
    y = np.concatenate(
        (np.zeros(bg_pixel_spacetime.shape[0]), np.ones(sig_pixel_spacetime.shape[0])),
        axis=0,
    )

    # Shuffle the data
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X_pixel = X_pixel[indices]
    X_mppc = X_mppc[indices]
    y = y[indices]

    pixel_label = np.zeros(X_pixel.shape[:-1] + (1,))
    pixel_label[(X_pixel[:, :, :] == -1).all(axis=-1)] = -1
    X_pixel_labelled = np.concatenate((X_pixel, pixel_label), axis=-1)
    mppc_label = np.ones(X_mppc.shape[:-1] + (1,))
    mppc_label[(X_mppc[:, :, :] == -1).all(axis=-1)] = -1
    X_mppc_labelled = np.concatenate((X_mppc, mppc_label), axis=-1)

    X = np.concatenate((X_pixel_labelled, X_mppc_labelled), axis=1)
    return X, y


X, y = get_labelled_points(
    bg_pixel_spacetime, bg_mppc_spacetime, sig_pixel_spacetime, sig_mppc_spacetime
)


def set_to_graph(features, label):
    # remove padded nodes (all zeros)
    mask = ~(features == -1).all(axis=1)
    x = torch.tensor(features[mask], dtype=torch.float)

    n = x.size(0)
    if n == 0:
        return None  # skip empty graphs
    y = torch.nn.functional.one_hot(
        torch.tensor([label], dtype=torch.long), num_classes=2
    ).type(torch.float)
    return Data(x=x, y=y)


graphs = [set_to_graph(s, l) for s, l in zip(X, y)]
graphs = [g for g in graphs if g is not None]  # filter out None
del X, y  # free memory

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

test_graphs, val_graphs = train_test_split(graphs, test_size=0.1, random_state=42)

loader = DataLoader(test_graphs, batch_size=512, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=512, shuffle=True)
del graphs, test_graphs, val_graphs  # free memory


class EdgeFunction(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeFunction, self).__init__()
        self.diff_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, concat_input: torch.Tensor) -> torch.Tensor:
        x_i, diff_x = concat_input.chunk(2, dim=-1)
        diff_out = self.diff_mlp(diff_x)
        pos_out = self.pos_mlp(x_i)
        out = torch.cat([diff_out, pos_out], dim=-1)
        out = self.final_mlp(out)
        return out


def get_mlp(in_channels, hidden_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, out_channels),
    )


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GNN, self).__init__()
        self.num_classes = num_classes
        self.mlp_hidden = hidden_dim
        self.mlp_1 = get_mlp(in_dim, self.mlp_hidden, hidden_dim)

        self.conv1 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, hidden_dim),
            k=10,
            aggr="sum",
        )
        self.mlp_2 = get_mlp(hidden_dim, self.mlp_hidden, hidden_dim)

        self.conv2 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, hidden_dim),
            k=5,
            aggr="sum",
        )
        self.mlp_3 = get_mlp(hidden_dim, self.mlp_hidden, hidden_dim)
        self.conv3 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, hidden_dim),
            k=10,
            aggr="sum",
        )
        self.mlp_4 = get_mlp(hidden_dim, self.mlp_hidden, hidden_dim)

        self.conv4 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, 2 * hidden_dim),
            k=5,
            aggr="sum",
        )

        concat_dim = hidden_dim * 3 + hidden_dim * 2
        latent_dim = hidden_dim * 2**4

        self.mlp_5 = get_mlp(concat_dim, concat_dim * 2**3, latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, latent_dim // 8),
            nn.ReLU(),
            nn.Linear(latent_dim // 8, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        x = self.mlp_1(x)
        conv_1 = self.conv1(x, data.batch)
        conv_2 = self.mlp_2(conv_1)
        conv_2 = self.conv2(conv_2, data.batch)
        conv_3 = self.mlp_3(conv_2)
        conv_3 = self.conv3(conv_3, data.batch)
        conv_4 = self.mlp_4(conv_3)
        conv_4 = self.conv4(conv_4, data.batch)
        output = torch.concat([conv_1, conv_2, conv_3, conv_4], dim=-1)

        mlp_5 = self.mlp_5(output)
        x = torch_geometric.nn.global_mean_pool(mlp_5, data.batch)  # Global pooling
        x = self.fc(x).squeeze()
        return F.log_softmax(x, dim=-1)


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(GNN, self).__init__()
        self.num_classes = num_classes
        self.mlp_hidden = hidden_dim
        self.mlp_1 = get_mlp(in_dim, self.mlp_hidden, hidden_dim)

        self.conv1 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, hidden_dim),
            k=10,
            aggr="sum",
        )
        self.mlp_2 = get_mlp(hidden_dim, self.mlp_hidden, hidden_dim)

        self.conv2 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, hidden_dim),
            k=5,
            aggr="sum",
        )
        self.mlp_3 = get_mlp(hidden_dim, self.mlp_hidden, hidden_dim)
        self.conv3 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, hidden_dim),
            k=10,
            aggr="sum",
        )
        self.mlp_4 = get_mlp(hidden_dim, self.mlp_hidden, hidden_dim)

        self.conv4 = DynamicEdgeConv(
            nn=EdgeFunction(hidden_dim, 2 * hidden_dim),
            k=5,
            aggr="sum",
        )

        concat_dim = hidden_dim * 3 + hidden_dim * 2
        latent_dim = hidden_dim * 2**4

        self.mlp_5 = get_mlp(concat_dim, concat_dim * 2**3, latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, latent_dim // 8),
            nn.ReLU(),
            nn.Linear(latent_dim // 8, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        x = self.mlp_1(x)
        conv_1 = self.conv1(x, data.batch)
        conv_2 = self.mlp_2(conv_1)
        conv_2 = self.conv2(conv_2, data.batch)
        conv_3 = self.mlp_3(conv_2)
        conv_3 = self.conv3(conv_3, data.batch)
        conv_4 = self.mlp_4(conv_3)
        conv_4 = self.conv4(conv_4, data.batch)
        output = torch.concat([conv_1, conv_2, conv_3, conv_4], dim=-1)

        mlp_5 = self.mlp_5(output)
        x = torch_geometric.nn.global_mean_pool(mlp_5, data.batch)  # Global pooling
        x = self.fc(x).squeeze()
        return F.log_softmax(x, dim=-1)


def get_class_weights(loader):
    labels = []
    for batch_data in loader:
        labels.append(batch_data.y)
    labels = torch.cat(labels).argmax(dim=-1)
    class_sample_count = torch.tensor(
        [(labels == t).sum() for t in torch.unique(labels, sorted=True)]
    )
    weight = 1.0 / class_sample_count.float()
    class_weights = weight / weight.sum() * len(torch.unique(labels))
    return class_weights


model = GNN(in_dim=5, hidden_dim=20, num_classes=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=get_class_weights(loader)).to(device)

print(
    f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
)

auc_metric = MulticlassAUROC(num_classes=2)
accuracy_metric = MulticlassAccuracy(num_classes=2)
metrics = {"auc": auc_metric, "accuracy": accuracy_metric}
history = {"loss": [], "val_loss": []}
history.update({"val_" + metric: [] for metric in metrics.keys()})
# Move model and criterion to the appropriate device
criterion

for epoch in range(1):
    auc_metric.reset()  # Reset metric at the start of each epoch
    accuracy_metric.reset()
    train_losses = []
    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        out = model(batch_data)
        loss = criterion(out, batch_data.y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.to("cpu").item())
    val_losses = []
    for device_batch_data in val_loader:
        y = batch_data.y
        device_batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(device_batch_data).to("cpu")
            val_losses.append(criterion(out, y).item())
            auc_metric.update(out, torch.argmax(y, dim=-1))
            accuracy_metric.update(out, torch.argmax(y, dim=-1))
    val_loss_mean = np.mean(val_losses)
    train_loss_mean = np.mean(train_losses)
    history["loss"].append(loss.item())
    history["val_loss"].append(val_loss_mean.item())
    for name, metric in metrics.items():
        history["val_" + name].append(metric.compute().item())
    print(
        f"Epoch {epoch}, Loss {train_loss_mean.item():.4f}, Val Loss {val_loss_mean.item():.4f}, "
        + ", ".join(
            [
                f"Val {name.upper()} {metric.compute().item():.4f}"
                for name, metric in metrics.items()
            ]
        )
    )

# Save the model
torch.save(model.state_dict(), f"{MODEL_DIR}/gnn_model.pth")

# Plot training and validation loss
fig, ax = plt.subplots(len(metrics) + 1, 1, figsize=(8, 6 * (len(metrics) + 1)))
if len(metrics) == 0:
    ax = [ax]
ax[0].plot(history["loss"], label="Train Loss")
ax[0].plot(history["val_loss"], label="Val Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()
for i, (name, _) in enumerate(metrics.items(), start=1):
    ax[i].plot(history["val_" + name], label="Val " + name.upper())
    ax[i].set_xlabel("Epoch")
    ax[i].set_ylabel(name.upper())
    ax[i].legend()
fig.tight_layout()
fig.savefig(f"{PLOTS_DIR}/gnn_training_history.png")

# Plot the ROC curve
from sklearn.metrics import roc_curve, auc

model.eval()
all_labels = []
all_probs = []
with torch.no_grad():
    for batch_data in val_loader:
        batch_data = batch_data.to(device)
        out = model(batch_data)
        probs = torch.softmax(out, dim=-1)[:, 1]  # Probability of the positive class
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch_data.y.cpu().numpy())
all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels).argmax(axis=-1)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color="blue", label="ROC curve (area = {:.2f})".format(roc_auc))
ax.plot([0, 1], [0, 1], color="red", linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curve")
ax.legend()
fig.savefig(f"{PLOTS_DIR}/gnn_roc_curve.png")
