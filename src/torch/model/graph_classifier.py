import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import HeteroConv, SAGEConv
from .components import get_mlp

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
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified forward pass.
        """
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
        return torch.sigmoid(logits.squeeze(-1))
    
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
    

class HeteroEdgeNetwork(nn.Module):
    """
    Edge network for heterogeneous graphs.
    """

    def __init__(self, node_dims: dict, edge_types: list, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.node_dims = node_dims
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial node feature projections
        self.node_projections = nn.ModuleDict()
        for node_type, dim in node_dims.items():
            self.node_projections[node_type] = get_mlp(dim, hidden_dim, num_layers=3,  dropout=dropout)

        # Learn edge weights for each edge type
        self.edge_nets = nn.ModuleDict()
        for edge_type in edge_types:
            self.edge_nets[edge_type] = get_mlp(3 * hidden_dim, 1, num_layers=3, dropout=dropout)
        
        # Convolution layers
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
        Forward pass for heterogeneous graph with edge networks.
        """
        x_dict, edge_index_dict, batch_dict = input_data.x_dict, input_data.edge_index_dict, input_data.batch_dict

        # Initial node feature projection
        for node_type, x in x_dict.items():
            x_dict[node_type] = F.relu(self.node_projections[node_type](x))

        # Compute edge weights and apply heterogeneous convolutions
        for conv, bn in zip(self.convs, self.batch_norms):
            edge_weights = {}
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                src_x = x_dict[src_type]
                dst_x = x_dict[dst_type]
                src_nodes = edge_index[0]
                dst_nodes = edge_index[1]

                edge_features = torch.cat([
                    src_x[src_nodes],
                    dst_x[dst_nodes],
                    torch.abs(src_x[src_nodes] - dst_x[dst_nodes])
                ], dim=-1)

                edge_weights[edge_type] = torch.sigmoid(self.edge_nets[edge_type](edge_features)).squeeze(-1)

            x_dict_new = conv(x_dict, edge_index_dict, edge_weight=edge_weights)
            for node_type in x_dict_new:
                x_new = bn[node_type](x_dict_new[node_type])
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
                x_dict[node_type] = x_dict[node_type] + x_new
                # Residual connection

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