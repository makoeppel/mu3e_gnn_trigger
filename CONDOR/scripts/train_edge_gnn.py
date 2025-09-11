import sys

sys.path.append("..")


import matplotlib.pyplot as plt
import numpy as np
import torch

DATA_DIR = "../mu3e_trigger_data"
MODEL_DIR = "../models"
PLOTS_DIR = "../plots"
from src.data_preparation import load_numpy_files

signal_prefix = f"{DATA_DIR}/sig"
background_prefix = f"{DATA_DIR}/bg"
signal_only_prefix = f"{DATA_DIR}/sig_only"

import src.torch.pre_processing.graph_batching as graph_batching
from importlib import reload
reload(graph_batching)

train_dataset, test_dataset = graph_batching.create_dataset(
    signal_prefix,
    has_layer_feature=True,
    n_events=100000,
    split=(0.8, 0.2),
    type="hetero",
    whole_event_mode=False,
    timing_cutoff=8,
    mppc_timing_cutoff=2,)

from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

from torch_geometric.data import HeteroData
from src.torch.model.components import get_mlp
class EdgeClassifier(torch.nn.Module):
    def __init__(self, node_dims, edge_types, num_layers, hidden_dim = 64, dropout = 0.0):
        super(EdgeClassifier, self).__init__()
        from torch_geometric.nn import HeteroConv, SAGEConv, Linear
        self.dropout = dropout if dropout is not None else 0.0
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("Dropout must be in the range [0.0, 1.0).")

        self.input_embeddings = torch.nn.ModuleDict()
        for node_type, node_dim in node_dims.items():
            self.input_embeddings[node_type] = get_mlp(node_dim, hidden_dim, 2)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_dim)
                for edge_type in edge_types
            }, aggr='mean')
            self.convs.append(conv)
        self.edge_classifiers = torch.nn.ModuleDict()
        for edge_type in edge_types:
            self.edge_classifiers["_".join(edge_type)] = get_mlp(2 * hidden_dim, 1, 3)
        self.lin = Linear(2 * hidden_dim, 1)

    def forward(self, hetero_graph: HeteroData):
        x_dict = hetero_graph.x_dict
        edge_index_dict = hetero_graph.edge_index_dict

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.input_embeddings[node_type](x)
            x_dict[node_type] = torch.relu(x_dict[node_type])
            x_dict[node_type] = torch.nn.functional.dropout(x_dict[node_type], p=self.dropout, training=self.training)

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: torch.nn.functional.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        edge_preds = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_x = x_dict[src_type]
            dst_x = x_dict[dst_type]
            edge_feat = torch.cat(
                [src_x[edge_index[0]], dst_x[edge_index[1]]], dim=-1
            )
            edge_pred = self.edge_classifiers["_".join(edge_type)](edge_feat).view(-1)
            edge_preds[edge_type] = edge_pred
        return edge_preds

edge_classifier = EdgeClassifier(
    node_dims=train_dataset.get_node_dims(),
    edge_types=train_dataset.get_edge_types(),
    num_layers=5,
    hidden_dim=64,
    dropout=0.1
)


from src.torch.training import FocalLoss, HeteroLossWrapper
from sklearn.metrics import roc_auc_score
criterion = HeteroLossWrapper(FocalLoss(gamma=2.0))
optimizer = torch.optim.AdamW(edge_classifier.parameters(), lr=0.0001)

from tqdm import tqdm
num_epochs = 40
for epoch in range(num_epochs):
    edge_classifier.train()
    total_loss = 0
    for hetero_graph in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        out = edge_classifier(hetero_graph)
        # Assuming labels are stored in hetero_graph.y
        loss = criterion(out, hetero_graph.edge_labels_dict)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


predictions = {}
labels = {}
edge_classifier.eval()
with torch.no_grad():
    for hetero_graph in tqdm(test_loader, desc="Evaluating"):
        out = edge_classifier(hetero_graph)
        for edge_type in out:
            if edge_type not in predictions:
                predictions[edge_type] = []
                labels[edge_type] = []
            predictions[edge_type].append(torch.sigmoid(out[edge_type].cpu()))
            labels[edge_type].append(hetero_graph.edge_labels_dict[edge_type].cpu())
    for edge_type in predictions:
        predictions[edge_type] = torch.cat(predictions[edge_type])
        labels[edge_type] = torch.cat(labels[edge_type])
        auc = roc_auc_score(labels[edge_type].numpy(), predictions[edge_type].numpy())
        print(f"Edge type: {edge_type}, AUC: {auc:.4f}")