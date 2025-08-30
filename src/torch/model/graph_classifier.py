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

import torch
from torch_geometric.nn import (
    HeteroConv,
    # Bipartite-compatible convolution layers:
    SAGEConv,  # GraphSAGE - works with bipartite
    GCNConv,  # Graph Convolutional - works with bipartite
    GraphConv,  # Higher-order - works with bipartite
    GINConv,  # Graph Isomorphism - works with bipartite
    TransformerConv,  # Graph Transformer - works with bipartite
    GATConv,  # Graph Attention - works with bipartite
    TAGConv,  # Topology Adaptive - works with bipartite
    global_mean_pool,
    global_max_pool,
    BatchNorm,
    SAGPooling,
)
import torch.nn.functional as F
from src.torch.model.components import get_mlp


class EventEdgeHeteroGNN_MultiConv(torch.nn.Module):
    def __init__(
        self,
        node_dims,
        edge_types,
        hidden_dim=32,
        num_layers=4,
        dropout=0.1,
        conv_type="sage",  # Single conv type OR list of conv types per layer
        aggregation_scheme: list | str | None = None,
        sagpool_ratio: float = 0.5,
        apply_pooling_after_layer: int = 1,
        # Layer-specific parameters (can be single values or lists)
        heads=4,  # For GAT/Transformer layers
        train_eps=True,  # For GIN layers
        K=3,  # For TAG layers
        # Mixed layer configurations
        layer_configs: list = None,  # Override individual layer settings
    ):
        super(EventEdgeHeteroGNN_MultiConv, self).__init__()
        self.node_dims = node_dims
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sagpool_ratio = sagpool_ratio
        self.apply_pooling_after_layer = apply_pooling_after_layer

        # Handle conv_type as single value or list
        if isinstance(conv_type, str):
            self.conv_types = [conv_type] * num_layers
        elif isinstance(conv_type, list):
            if len(conv_type) != num_layers:
                raise ValueError("Length of conv_type list must match num_layers")
            self.conv_types = conv_type
        else:
            raise ValueError("conv_type must be a string or list of strings")

        # Handle layer-specific parameters
        self.heads_per_layer = self._ensure_list(heads, num_layers)
        self.train_eps_per_layer = self._ensure_list(train_eps, num_layers)
        self.K_per_layer = self._ensure_list(K, num_layers)

        # Layer configurations override (for fine-grained control)
        if layer_configs is not None:
            if len(layer_configs) != num_layers:
                raise ValueError("Length of layer_configs must match num_layers")
            self.layer_configs = layer_configs
        else:
            # Build default configs
            self.layer_configs = []
            for i in range(num_layers):
                config = {
                    "conv_type": self.conv_types[i],
                    "heads": self.heads_per_layer[i],
                    "train_eps": self.train_eps_per_layer[i],
                    "K": self.K_per_layer[i],
                    "dropout": dropout,
                }
                self.layer_configs.append(config)

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

        # Ensure we use safe aggregation schemes
        safe_aggr = []
        for aggr in self.aggregation_scheme:
            if aggr in ["add", "sum"]:
                safe_aggr.append("mean")  # Replace problematic aggregations
            else:
                safe_aggr.append(aggr)
        self.aggregation_scheme = safe_aggr

        # Initial linear transformations for each node type
        self.node_lin = torch.nn.ModuleDict(
            {
                node_type: get_mlp(in_dim, hidden_dim, num_layers=3, dropout=dropout)
                for node_type, in_dim in node_dims.items()
            }
        )

        # SAGPooling for MPPC node type redundancy reduction
        self.mppc_pool = SAGPooling(hidden_dim, ratio=sagpool_ratio)

        # Create different convolution types per layer
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for layer_idx in range(num_layers):
            config = self.layer_configs[layer_idx]
            conv_type = config["conv_type"]
            heads = config["heads"]
            train_eps = config["train_eps"]
            K = config["K"]
            layer_dropout = config["dropout"]

            conv_dict = {}

            for edge_type in edge_types:
                if conv_type == "sage":
                    conv_dict[edge_type] = SAGEConv(
                        (hidden_dim, hidden_dim), hidden_dim
                    )

                elif conv_type == "gcn":
                    conv_dict[edge_type] = GCNConv(
                        hidden_dim, hidden_dim, add_self_loops=False
                    )

                elif conv_type == "gat":
                    # GATConv outputs heads * out_dim, so we need to ensure output is hidden_dim
                    if layer_idx == num_layers - 1:  # Last layer - single head
                        conv_dict[edge_type] = GATConv(
                            hidden_dim,
                            hidden_dim,
                            heads=1,
                            dropout=layer_dropout,
                            concat=False,
                            add_self_loops=False,
                        )
                    else:
                        # Multi-head: either concat=True with out_dim=hidden_dim//heads, or concat=False with out_dim=hidden_dim
                        conv_dict[edge_type] = GATConv(
                            hidden_dim,
                            hidden_dim,
                            heads=heads,
                            dropout=layer_dropout,
                            concat=False,
                            add_self_loops=False,
                        )

                elif conv_type == "gin":
                    # GIN requires a neural network for each edge type
                    nn = torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                    )
                    conv_dict[edge_type] = GINConv(nn, train_eps=train_eps)

                elif conv_type == "graph":
                    conv_dict[edge_type] = GraphConv(hidden_dim, hidden_dim, aggr="add")

                elif conv_type == "transformer":
                    conv_dict[edge_type] = TransformerConv(
                        hidden_dim,
                        hidden_dim,
                        heads=heads,
                        dropout=layer_dropout,
                        concat=False,
                    )

                elif conv_type == "tag":
                    conv_dict[edge_type] = TAGConv(hidden_dim, hidden_dim, K=K)

                else:
                    raise ValueError(
                        f"Unknown convolution type: {conv_type} at layer {layer_idx}"
                    )

            conv = HeteroConv(conv_dict, aggr=self.aggregation_scheme[layer_idx])
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

    def _ensure_list(self, value, length):
        """Ensure parameter is a list of specified length"""
        if isinstance(value, list):
            if len(value) != length:
                raise ValueError(
                    f"Length of parameter list must match num_layers ({length})"
                )
            return value
        else:
            return [value] * length

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
            dummy_edges = torch.zeros((2, 0), dtype=torch.long, device=mppc_x.device)
            pooled_x, _, _, pooled_batch, perm, _ = self.mppc_pool(
                mppc_x, dummy_edges, batch=mppc_batch
            )
            pooled_edge_index = None

        # Update dictionaries
        new_x_dict = x_dict.copy()
        new_edge_index_dict = {}
        new_batch_dict = batch_dict.copy()

        new_x_dict["mppc"] = pooled_x
        new_batch_dict["mppc"] = pooled_batch

        # Create mapping from old indices to new indices
        old_to_new = torch.full(
            (mppc_x.size(0),), -1, dtype=torch.long, device=perm.device
        )
        old_to_new[perm] = torch.arange(len(perm), device=perm.device)

        # Process all edge types
        for edge_type in edge_index_dict.keys():
            old_edge_index = edge_index_dict[edge_type]

            if edge_type == ("mppc", "to", "mppc"):
                if pooled_edge_index is not None and pooled_edge_index.size(1) > 0:
                    new_edge_index_dict[edge_type] = pooled_edge_index

            elif edge_type[0] == "mppc" and edge_type[2] != "mppc":
                if old_edge_index.size(1) > 0:
                    mask = old_to_new[old_edge_index[0]] >= 0
                    if mask.sum() > 0:
                        new_edge_index = old_edge_index[:, mask].clone()
                        new_edge_index[0] = old_to_new[new_edge_index[0]]
                        new_edge_index_dict[edge_type] = new_edge_index

            elif edge_type[0] != "mppc" and edge_type[2] == "mppc":
                if old_edge_index.size(1) > 0:
                    mask = old_to_new[old_edge_index[1]] >= 0
                    if mask.sum() > 0:
                        new_edge_index = old_edge_index[:, mask].clone()
                        new_edge_index[1] = old_to_new[new_edge_index[1]]
                        new_edge_index_dict[edge_type] = new_edge_index
            else:
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

        # Initial node feature transformation
        x_dict = {
            node_type: torch.relu(self.node_lin[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Apply HeteroConv layers
        for layer, conv in enumerate(self.convs):
            # Filter out empty edge indices before passing to conv
            filtered_edge_index_dict = {
                edge_type: edge_index
                for edge_type, edge_index in edge_index_dict.items()
                if edge_index.size(1) > 0
            }

            # Skip layer if no edges remain
            if not filtered_edge_index_dict:
                continue

            x_dict = conv(x_dict, filtered_edge_index_dict)

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