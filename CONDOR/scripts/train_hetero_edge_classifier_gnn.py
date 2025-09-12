import sys
from tqdm import tqdm

sys.path.append("..")


import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"/data/dust/group/atlas/ttreco/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"
MODEL_NAME = "contrastive_loss"

signal_prefix = f"{DATA_DIR}/sig"
background_prefix = f"{DATA_DIR}/bg"
signal_only_prefix = f"{DATA_DIR}/sig_only"

import src.torch.pre_processing.graph_batching as graph_batching

train_dataset, test_dataset = graph_batching.create_dataset(
    signal_prefix,
    has_layer_feature=True,
    n_events=100000,
    split=(0.8, 0.2),
    type="layer_separated",
    whole_event_mode=False,
    timing_cutoff=8,
    mppc_timing_cutoff=2,)

from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
del train_dataset, test_dataset # save memory


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, TransformerConv


def get_mlp(in_dim, out_dim, num_layers=2, hidden_dim=64, dropout=0.1):
    layers = []
    d = in_dim
    for i in range(num_layers - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TrackingHeteroGNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # --- Input embeddings per node type ---
        self.input_embeddings = nn.ModuleDict({
            "layer_1": get_mlp(3, hidden_dim, num_layers=2, hidden_dim=hidden_dim, dropout=dropout),
            "layer_2": get_mlp(3, hidden_dim, num_layers=2, hidden_dim=hidden_dim, dropout=dropout),
            "layer_3": get_mlp(3, hidden_dim, num_layers=2, hidden_dim=hidden_dim, dropout=dropout),
            "layer_4": get_mlp(3, hidden_dim, num_layers=2, hidden_dim=hidden_dim, dropout=dropout),
            "mppc":    get_mlp(4, hidden_dim, num_layers=2, hidden_dim=hidden_dim, dropout=dropout),
        })

        # --- Convs ---
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    # every relation uses the same conv class
                    edge_type: TransformerConv((-1, -1), hidden_dim,concat=False, heads=2, dropout=dropout)
                    for edge_type in [
                        ("layer_1", "to", "layer_2"),
                        ("layer_2", "to", "layer_1"),
                        ("layer_2", "to", "mppc"),
                        ("mppc", "to", "layer_2"),
                        ("mppc", "to", "mppc"),
                        ("mppc", "to", "layer_3"),
                        ("layer_3", "to", "mppc"),
                        ("layer_3", "to", "layer_4"),
                        ("layer_4", "to", "layer_3"),
                        ("layer_4", "to", "layer_4"),
                    ]
                },
                aggr="mean",
            )
            self.convs.append(conv)

        # --- Edge classifiers ---
        self.edge_classifiers = nn.ModuleDict()
        for edge_type in [
            ("layer_1", "to", "layer_2"),
            ("layer_2", "to", "layer_1"),
            ("layer_2", "to", "mppc"),
            ("mppc", "to", "layer_2"),
            ("mppc", "to", "mppc"),
            ("mppc", "to", "layer_3"),
            ("layer_3", "to", "mppc"),
            ("layer_3", "to", "layer_4"),
            ("layer_4", "to", "layer_3"),
            ("layer_4", "to", "layer_4"),
        ]:
            name = "_".join(edge_type)
            # input: src || dst || Δx,Δy,Δz,(Δt if applicable)
            in_dim = 2 * hidden_dim + (4 if "mppc" in edge_type else 3)
            self.edge_classifiers[name] = get_mlp(in_dim, 1, num_layers=3, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # --- Embed nodes ---
        for ntype, x in x_dict.items():
            x = self.input_embeddings[ntype](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_dict[ntype] = x

        # --- Message passing ---
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # --- Edge predictions ---
        edge_preds = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, _, dst = edge_type
            src_x = x_dict[src][edge_index[0]]
            dst_x = x_dict[dst][edge_index[1]]

            # build edge features
            raw_src = data[src].x[edge_index[0]]
            raw_dst = data[dst].x[edge_index[1]]
            diff = raw_dst[:, :3] - raw_src[:, :3]  # Δx,Δy,Δz
            if src == "mppc" or dst == "mppc":
                # include Δt if mppc involved
                t_src = raw_src[:, 3] if raw_src.size(1) > 3 else torch.zeros_like(raw_src[:, 0])
                t_dst = raw_dst[:, 3] if raw_dst.size(1) > 3 else torch.zeros_like(raw_dst[:, 0])
                dt = (t_dst - t_src).unsqueeze(-1)
                edge_feat = torch.cat([src_x, dst_x, diff, dt], dim=-1)
            else:
                edge_feat = torch.cat([src_x, dst_x, diff], dim=-1)

            pred = self.edge_classifiers["_".join(edge_type)](edge_feat).view(-1)
            edge_preds[edge_type] = pred

        return edge_preds


edge_classifier = TrackingHeteroGNN(
    num_layers=5,
    hidden_dim=64,
    dropout=0.1,
)

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from functools import partial

# --- Loss function ---
def hetero_edge_loss(edge_preds, edge_labels, pos_weight_dict=None):
    """
    BCE loss averaged across edge types, optional pos_weight per type
    """
    losses = []
    for edge_type, pred in edge_preds.items():
        target = edge_labels[edge_type].float().to(pred.device)
        pos_weight = None
        if pos_weight_dict and edge_type in pos_weight_dict:
            pos_weight = torch.tensor([pos_weight_dict[edge_type]], device=pred.device)
        loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        losses.append(loss)
    return torch.stack(losses).mean()


# --- Evaluation function ---
@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()
    all_metrics = {}
    for batch in loader:
        batch = batch.to(device)
        edge_preds = model(batch)
        edge_labels = batch.edge_labels_dict
        for edge_type, pred in edge_preds.items():
            target = edge_labels[edge_type].float().to(pred.device)
            prob = torch.sigmoid(pred)
            pred_bin = (prob > 0.5).float()

            acc = (pred_bin == target).float().mean().item()
            try:
                auc = roc_auc_score(target.cpu().numpy(), prob.cpu().numpy())
            except ValueError:
                auc = float("nan")  # only one class in batch

            if edge_type not in all_metrics:
                all_metrics[edge_type] = {"acc": [], "auc": []}
            all_metrics[edge_type]["acc"].append(acc)
            all_metrics[edge_type]["auc"].append(auc)

    # Average metrics per edge type
    for edge_type in all_metrics:
        all_metrics[edge_type]["acc"] = sum(all_metrics[edge_type]["acc"]) / len(all_metrics[edge_type]["acc"])
        all_metrics[edge_type]["auc"] = sum(all_metrics[edge_type]["auc"]) / len(all_metrics[edge_type]["auc"])

    return all_metrics


# --- Training loop ---
def train(
    model,
    train_loader,
    val_loader=None,
    edge_weight_dict=None,
    num_epochs=20,
    lr=1e-3,
    device="cuda"
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = partial(hetero_edge_loss, pos_weight_dict=edge_weight_dict)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_preds = model(batch)
            loss = criterion(edge_preds, batch.edge_labels_dict)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

        # Optional validation
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            print(f"Validation metrics @ epoch {epoch+1}:")
            for etype, vals in metrics.items():
                print(f"  {etype}: acc={vals['acc']:.3f}, auc={vals['auc']:.3f}")

edge_weight_dict = {}
for batch in train_loader:
    for edge_type, labels in batch.edge_labels_dict.items():
        num_pos = labels.sum().item()
        num_neg = len(labels) - num_pos
        if num_pos > 0:
            edge_weight_dict[edge_type] = num_neg / num_pos


train(
    edge_classifier,
    train_loader,
    val_loader=test_loader,
    edge_weight_dict=edge_weight_dict,
    num_epochs=40,
    lr=1e-3,
    device= "cuda" if torch.cuda.is_available() else "cpu"
)
