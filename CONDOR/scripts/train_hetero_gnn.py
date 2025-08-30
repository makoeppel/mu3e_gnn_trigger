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

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

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


from sklearn.model_selection import train_test_split


X_pixel_train, X_pixel_test, X_mppc_train, X_mppc_test, y_train, y_test = (
    train_test_split(X_pixel, X_mppc, y, test_size=0.2, random_state=42, stratify=y)
)


del (
    sig_pixel_spacetime,
    sig_mppc_spacetime,
    bg_pixel_spacetime,
    bg_mppc_spacetime,
    X_pixel,
    X_mppc,
    y,
)

import src.torch.pre_processing.graph_batching as gc
from importlib import reload

reload(gc)

from torch_geometric.loader import DataLoader

event_processor = gc.EventProcessor(gc.HeteroGraphBuilder())

hetero_graph_train = event_processor.process_to_graphs(
    X_pixel=X_pixel_train, X_mppc=X_mppc_train, labels=y_train
)
hetero_graph_test = event_processor.process_to_graphs(
    X_pixel=X_pixel_test, X_mppc=X_mppc_test, labels=y_test
)

train_loader = DataLoader(hetero_graph_train, batch_size=512, shuffle=True)
test_loader = DataLoader(hetero_graph_test, batch_size=512, shuffle=False)

del X_pixel_train, X_pixel_test, X_mppc_train, X_mppc_test, y_train, y_test


import torch
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool, global_max_pool, BatchNorm, SAGPooling
import torch.nn.functional as F
from src.torch.model.components import get_mlp
import torch
from torch_geometric.nn import (
    HeteroConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    BatchNorm,
    SAGPooling,
)
import torch.nn.functional as F
from src.torch.model.components import get_mlp


class EventEdgeHeteroGNN(torch.nn.Module):
    def __init__(
        self,
        node_dims,
        edge_types,
        hidden_dim=32,
        num_layers=4,
        dropout=0.1,
        aggregation_scheme: list | str | None = None,
        sagpool_ratio: float = 0.5,
        apply_pooling_after_layer: int = 1,  # Apply pooling after this layer
    ):
        super(EventEdgeHeteroGNN, self).__init__()
        self.node_dims = node_dims
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sagpool_ratio = sagpool_ratio
        self.apply_pooling_after_layer = apply_pooling_after_layer

        if aggregation_scheme is None:
            self.aggregation_scheme = ["mean"] * num_layers
        elif isinstance(aggregation_scheme, str):
            self.aggregation_scheme = [aggregation_scheme] * num_layers
        elif isinstance(aggregation_scheme, list):
            if len(aggregation_scheme) != num_layers:
                raise ValueError(
                    "Length of aggregation_scheme list must match num_layers"
                )
            self.aggregation_scheme = aggregation_scheme
        else:
            raise ValueError(
                "aggregation_scheme must be None, a string, or a list of strings"
            )

        # Initial linear transformations for each node type
        self.node_lin = torch.nn.ModuleDict(
            {
                node_type: get_mlp(in_dim, hidden_dim, num_layers=3, dropout=dropout)
                for node_type, in_dim in node_dims.items()
            }
        )

        # SAGPooling for MPPC node type redundancy reduction
        self.mppc_pool = SAGPooling(hidden_dim, ratio=sagpool_ratio)

        # HeteroConv layers + per-node-type batchnorms
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for conv_layer_index in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                    for edge_type in edge_types
                },
                aggr=self.aggregation_scheme[conv_layer_index],
            )
            self.convs.append(conv)

            # BatchNorm for each node type in this layer
            self.bns.append(
                torch.nn.ModuleDict(
                    {node_type: BatchNorm(hidden_dim) for node_type in node_dims}
                )
            )

        # Final linear layer for classification
        self.classifier = get_mlp(
            2 * hidden_dim * len(node_dims), 1, num_layers=3, dropout=dropout
        )

    def apply_mppc_pooling(self, x_dict, edge_index_dict, batch_dict):
        """Apply SAGPooling to MPPC nodes and update edge connections"""
        if "mppc" not in x_dict:
            return x_dict, edge_index_dict, batch_dict
            
        # Extract MPPC data
        mppc_x = x_dict["mppc"]
        mppc_batch = batch_dict["mppc"]
        
        # Find MPPC self-connections edge index
        mppc_to_mppc_edge = None
        for edge_type in edge_index_dict.keys():
            if edge_type == ("mppc", "to", "mppc"):
                mppc_to_mppc_edge = edge_index_dict[edge_type]
                break
        
        # Apply SAGPooling
        if mppc_to_mppc_edge is not None and mppc_to_mppc_edge.size(1) > 0:
            pooled_x, pooled_edge_index, _, pooled_batch, perm, _ = self.mppc_pool(
                mppc_x, mppc_to_mppc_edge, batch=mppc_batch
            )
        else:
            # If no self-connections, pool without edge information
            # Create dummy edge index for pooling
            dummy_edges = torch.zeros((2, 0), dtype=torch.long, device=mppc_x.device)
            pooled_x, _, _, pooled_batch, perm, _ = self.mppc_pool(
                mppc_x, dummy_edges, batch=mppc_batch
            )
            pooled_edge_index = None
        
        # Update dictionaries
        new_x_dict = x_dict.copy()
        new_edge_index_dict = {}  # Start fresh to avoid referencing deleted keys
        new_batch_dict = batch_dict.copy()
        
        new_x_dict["mppc"] = pooled_x
        new_batch_dict["mppc"] = pooled_batch
        
        # Create mapping from old indices to new indices
        old_to_new = torch.full((mppc_x.size(0),), -1, dtype=torch.long, device=perm.device)
        old_to_new[perm] = torch.arange(len(perm), device=perm.device)
        
        # Process all edge types
        for edge_type in edge_index_dict.keys():
            old_edge_index = edge_index_dict[edge_type]
            
            if edge_type == ("mppc", "to", "mppc"):
                # MPPC to MPPC edges - use pooled edge index if available
                if pooled_edge_index is not None and pooled_edge_index.size(1) > 0:
                    new_edge_index_dict[edge_type] = pooled_edge_index
                # Skip if no edges remain
                
            elif edge_type[0] == "mppc" and edge_type[2] != "mppc":
                # MPPC is source, other type is target
                if old_edge_index.size(1) > 0:
                    # Filter edges where source MPPC node still exists
                    mask = old_to_new[old_edge_index[0]] >= 0
                    if mask.sum() > 0:
                        new_edge_index = old_edge_index[:, mask].clone()
                        new_edge_index[0] = old_to_new[new_edge_index[0]]
                        new_edge_index_dict[edge_type] = new_edge_index
                
            elif edge_type[0] != "mppc" and edge_type[2] == "mppc":
                # Other type is source, MPPC is target
                if old_edge_index.size(1) > 0:
                    # Filter edges where target MPPC node still exists
                    mask = old_to_new[old_edge_index[1]] >= 0
                    if mask.sum() > 0:
                        new_edge_index = old_edge_index[:, mask].clone()
                        new_edge_index[1] = old_to_new[new_edge_index[1]]
                        new_edge_index_dict[edge_type] = new_edge_index
                        
            else:
                # Edge type doesn't involve MPPC - keep as is
                new_edge_index_dict[edge_type] = old_edge_index
        
        return new_x_dict, new_edge_index_dict, new_batch_dict

    def forward(self, input_data):
        x_dict, edge_index_dict, batch_dict = (
            input_data.x_dict,
            input_data.edge_index_dict,
            input_data.batch_dict,
        )
        if set(x_dict.keys()) != set(self.node_dims.keys()):
            print(f"Expected node types: {self.node_dims.keys()}")
            print(f"Received node types: {x_dict.keys()}")
            raise ValueError("Node types in input do not match model configuration.")

        # ---- Initial node feature transformation ----
        x_dict = {
            node_type: torch.relu(self.node_lin[node_type](x))
            for node_type, x in x_dict.items()
        }

        # ---- Apply HeteroConv layers ----
        for layer, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            # Apply BN + ReLU + Dropout per node type
            new_x_dict = {}
            for node_type, x in x_dict.items():
                x = self.bns[layer][node_type](x)
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[node_type] = x
            x_dict = new_x_dict

            # Apply MPPC pooling after specified layer
            if layer == self.apply_pooling_after_layer:
                x_dict, edge_index_dict, batch_dict = self.apply_mppc_pooling(
                    x_dict, edge_index_dict, batch_dict
                )

        # ---- Global pooling (mean + max for each node type) ----
        pooled = []
        for node_type, x in x_dict.items():
            pooled.append(global_mean_pool(x, batch_dict[node_type]))
            pooled.append(global_max_pool(x, batch_dict[node_type]))

        # Concatenate pooled features from all node types
        h = torch.cat(pooled, dim=1)

        # ---- Classification ----
        out = self.classifier(h).squeeze()
        return torch.sigmoid(out).squeeze(-1)


# Usage example:
# model = EventEdgeHeteroGNN(
#     node_dims={"mppc": 64, "event": 32},
#     edge_types=[("mppc", "to", "event"), ("mppc", "to", "mppc"), ("event", "to", "mppc")],
#     sagpool_ratio=0.3,              # Keep 30% of MPPC nodes
#     apply_pooling_after_layer=1     # Apply pooling after layer 1
# )sample_graph = hetero_graph_train[0]

sample_graph = hetero_graph_train[0]
node_dims = {
    node_type: sample_graph.x_dict[node_type].shape[1]
    for node_type in sample_graph.x_dict
}
edge_types = list(sample_graph.edge_index_dict.keys())

model = EventEdgeHeteroGNN(
    node_dims=node_dims,
    edge_types=edge_types,
    hidden_dim=32,
    num_layers=6,
    dropout=0.2,
    aggregation_scheme=["mean", "mean", "max", "mean", "mean" ,"max"],
    sagpool_ratio=0.5,
)


def get_class_weights(train_data, alpha = 2):
    total_samples = 0
    positive_samples = 0

    for data in train_data:
        labels = data.y
        total_samples += 1
        positive_samples += labels.sum().item()

    negative_samples = total_samples - positive_samples

    weight_for_0 = (1 / negative_samples) * (total_samples) / 2.0
    weight_for_1 = (1 / positive_samples) * (total_samples) / 2.0

    positive_weight = weight_for_1 / weight_for_0 if weight_for_0 > 0 else 1.0

    return torch.tensor(positive_weight ** alpha, dtype=torch.float)
weight = get_class_weights(hetero_graph_train)
weight.to(device)
bce_loss = torch.nn.BCELoss().to(device)

import src.torch.training as train
from importlib import reload

reload(train)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss = train.FocalLoss(alpha=0.25, gamma=2.0, from_logits=False).to(device)

trained_model, history = train.train_graph_classifier(
    train_loader,
    test_loader,
    model,
    50,
    optimizer=optimizer,
    scheduler=None,
    criterion=bce_loss,
    MODEL_DIR=MODEL_DIR,
    MODEL_NAME="hetero_gnn",
    device=device,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history["train_aucs"], label="Train AUC")
ax.plot(history["val_aucs"], label="Validation AUC")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("Training and Validation AUC over Epochs")
ax.legend()
plt.savefig(f"{PLOTS_DIR}/hetero_gnn_training_auc.png")


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(
    [data.y.item() for data in hetero_graph_test],
    [
        trained_model(data.unsqueeze(0).to(trained_model.device)).item()
        for data in hetero_graph_test
    ],
)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig(f"{PLOTS_DIR}/hetero_gnn_roc_curve.png")