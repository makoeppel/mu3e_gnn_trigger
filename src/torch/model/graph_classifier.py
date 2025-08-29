import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import HeteroConv, SAGEConv
from .components import get_mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv
from torch_geometric.nn import BatchNorm, LayerNorm


class SimpleGraphClassifier(nn.Module):
    """
    Simplified graph classifier that's easier to use and debug.
    """

    def __init__(
        self,
        node_input_dim: int,
        hidden_dim: int = 64,
        num_conv_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Node feature processing
        self.node_projection = nn.Linear(node_input_dim, hidden_dim)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.aggregation_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_conv_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Multi-scale pooling
        self.dropout = dropout

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3x for multi-pooling
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, batch
    ) -> torch.Tensor:
        """
        Simplified forward pass.
        """
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        # Project node features
        x = F.relu(self.node_projection(x))

        # Apply convolutions
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual connection

        # Multi-scale pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)

        # Combine pooled features
        graph_features = torch.cat([mean_pool, max_pool, sum_pool], dim=-1)

        # Final classification
        logits = self.classifier(graph_features)
        return torch.sigmoid(logits).squeeze(-1)
class SimpleHeteroGraphClassifier(nn.Module):
    """
    Simplified heterogeneous graph classifier.
    """

    def __init__(
        self,
        node_dims: dict,
        edge_types: list,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_dims = node_dims
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial node feature projections
        self.node_projections = nn.ModuleDict()
        for node_type, dim in node_dims.items():
            self.node_projections[node_type] = nn.Linear(dim, hidden_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.batch_norms.append(nn.ModuleDict(
                {node_type: BatchNorm(hidden_dim) for node_type in node_dims}
            ))

        # Classifier
        self.classifier = nn.Linear(2 * hidden_dim * len(node_dims), 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_data):
        """
        Forward pass for heterogeneous graph.
        """
        x_dict, edge_index_dict, batch_dict = input_data.x_dict, input_data.edge_index_dict, input_data.batch_dict

        # Initial node feature projection
        for node_type, x in x_dict.items():
            x_dict[node_type] = F.relu(self.node_projections[node_type](x))

        # Apply heterogeneous convolutions
        for conv, bn in zip(self.convs, self.batch_norms):
            x_dict_new = conv(x_dict, edge_index_dict)
            for node_type in x_dict_new:
                x_new = bn[node_type](x_dict_new[node_type])
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                x_dict[node_type] = x_dict[node_type] + x_new  # Residual connection

        # Multi-scale pooling for each node type
        pooled = []
        for node_type, x in x_dict.items():
            pooled.append(global_mean_pool(x, batch_dict[node_type]))
            pooled.append(global_max_pool(x, batch_dict[node_type]))

        if len(pooled) != 2 * len(self.node_dims):
            print(f"Nodes: {x_dict.keys()}")
            raise ValueError("Pooled features length mismatch.")

        # Concatenate pooled features from all node types
        h = torch.cat(pooled, dim=1)

        # Final classification
        logits = self.classifier(self.dropout_layer(h))
        return logits.squeeze(-1)
    

class EventEdgeHeteroGNN(torch.nn.Module):
    def __init__(self, node_dims, edge_types, hidden_dim=32, num_layers=4, dropout=0.1):
        super(EventEdgeHeteroGNN, self).__init__()
        self.node_dims = node_dims
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial linear transformations for each node type
        self.node_lin = torch.nn.ModuleDict({
            node_type: get_mlp(in_dim, hidden_dim, num_layers=2, dropout=dropout)
            for node_type, in_dim in node_dims.items()
        })

        # HeteroConv layers + per-node-type batchnorms
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                    for edge_type in edge_types
                },
                aggr="mean",   # <--- changed from "sum"
            )
            self.convs.append(conv)

            # BatchNorm for each node type in this layer
            self.bns.append(torch.nn.ModuleDict({
                node_type: BatchNorm(hidden_dim) for node_type in node_dims
            }))

        # Final linear layer for classification
        self.classifier = get_mlp(
            2 * hidden_dim * len(node_dims), 1, num_layers=3, dropout=dropout
        )

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

        # Initial node feature transformation
        x_dict = {
            node_type: torch.relu(self.node_lin[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Apply HeteroConv layers
        for layer, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            # Apply BN + ReLU + Dropout per node type
            new_x_dict = {}
            for node_type, x in x_dict.items():
                x = self.bns[layer][node_type](x)   # BN first
                x = torch.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[node_type] = x
            x_dict = new_x_dict

        # Global pooling (mean + max for each node type)
        pooled = []
        for node_type, x in x_dict.items():
            pooled.append(global_mean_pool(x, batch_dict[node_type]))
            pooled.append(global_max_pool(x, batch_dict[node_type]))

        # Concatenate pooled features from all node types
        h = torch.cat(pooled, dim=1)

        # Classification 
        out = self.classifier(h).squeeze()
        return torch.sigmoid(out).squeeze(-1)
