import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_scatter import scatter_add, scatter_mean, scatter_max
from typing import Optional, Union


from torch_geometric.nn import MessagePassing
from .mlp import get_mlp, Classifier


# ----------- Edge Function for Dynamic Edge Weights ----------- #
class EdgeFunction(nn.Module):
    """Function to dynamically generate edge weights, e.g. for DynamicEdgeConv.
    Based on the implementation in PointNet++ (https://arxiv.org/abs/1706.02413)."""

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
        )

    def forward(self, concat_input: torch.Tensor) -> torch.Tensor:
        x_i, diff_x = concat_input.chunk(2, dim=-1)
        diff_out = self.diff_mlp(diff_x)
        pos_out = self.pos_mlp(x_i)
        out = torch.cat([diff_out, pos_out], dim=-1)
        out = self.final_mlp(out)
        return out


# ----------- Full Model Combining GNN and Classifier ----------- #
class SequenceGNNClassifier(nn.Module):
    """Complete model combining GNN and sequence classifier."""

    def __init__(self, gnn: nn.Module, classifier_module: nn.Module):
        super().__init__()
        self.gnn = gnn
        self.classifier_module = classifier_module

    def forward(self, batch):
        """
        Args:
            batch: PyG Batch object with attributes:
                - x: [num_nodes_total, feature_dim]
                - edge_index: [2, num_edges_total]
                - batch: [num_nodes_total] mapping nodes to graphs
                - event_batch: [num_graphs_total] mapping graphs to events
        Returns:
            classifier_outputs: [num_events, num_classes]
        """
        if not hasattr(batch, "event_idx"):
            raise ValueError("Batch must have 'event_batch' attribute")

        x, edge_index, graph_batch, event_idx = (
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.event_idx,
        )

        graph_embeddings = self.gnn(x, edge_index, graph_batch)
        classifier_outputs = self.classifier_module(graph_embeddings, event_idx)

        return classifier_outputs


# ----------- GNN Components ----------- #
class EdgeFeatureGenerator(nn.Module):
    """Learns initial edge weights from node pairs."""

    def __init__(self, node_dim=3, edge_features=16, num_layers=3):
        super().__init__()
        self.edge_mlp = get_mlp(2 * node_dim + 1, edge_features, num_layers)

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_features = torch.cat(
            [x[src], x[dst], torch.norm([src] - x[dst], p=2)], dim=-1
        )
        edge_features = self.edge_mlp(edge_features).squeeze(-1)
        return edge_features


class WeightedMessagePassing(MessagePassing):
    """Message passing layer that uses learned edge weights."""

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * self.node_mlp(x_j)


class EdgeFeatureUpdater(nn.Module):
    """Updates edge weights based on node features and current edge weights."""

    def __init__(self, node_dim, edge_dim, target_edge_dim=None, num_layers=3):
        super().__init__()
        if target_edge_dim is None:
            out_dim = edge_dim
        self.edge_function = get_mlp(2 * node_dim + edge_dim, edge_dim, num_layers=3)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        edge_inputs = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        updated_edge_attr = self.edge_function(edge_inputs)
        return updated_edge_attr


def get_mlp(
    input_dim: int,
    output_dim: int,
    num_layers: int = 3,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
):
    """
    Create a multi-layer perceptron.
    """
    if hidden_dim is None:
        hidden_dim = max(input_dim, output_dim)

    layers = []
    current_dim = input_dim

    for i in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))

    return nn.Sequential(*layers)


class EdgeConvLayer(MessagePassing):
    """
    Graph convolution layer that operates on edge features.
    Updates both node and edge features through message passing.
    """

    def __init__(
        self,
        input_node_dim: int,
        edge_dim: int,
        out_node_dim: int,
        out_edge_dim: int,
        aggr: str = "add",
        update_edges: bool = True,
        **kwargs
    ):
        super(EdgeConvLayer, self).__init__(aggr=aggr, **kwargs)

        self.input_node_dim = input_node_dim
        self.edge_dim = edge_dim
        self.out_node_dim = out_node_dim
        self.out_edge_dim = out_edge_dim
        self.update_edges = update_edges

        # Message function: combines source node, target node, and edge features
        self.message_mlp = get_mlp(
            input_dim=2 * input_node_dim + edge_dim,
            output_dim=out_node_dim,
            num_layers=3,
            hidden_dim=max(2 * input_node_dim + edge_dim, out_node_dim),
        )

        # Node update function
        self.node_update_mlp = get_mlp(
            input_dim=input_node_dim + out_node_dim,
            output_dim=out_node_dim,
            num_layers=2,
        )

        # Edge update function (optional)
        if update_edges:
            self.edge_update_mlp = get_mlp(
                input_dim=2 * input_node_dim + edge_dim,
                output_dim=out_edge_dim,
                num_layers=2,
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple:
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Tuple of (updated_node_features, updated_edge_features)
        """
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update nodes
        x_new = self.node_update_mlp(torch.cat([x, out], dim=-1))

        # Update edges if requested
        edge_attr_new = edge_attr
        if self.update_edges:
            row, col = edge_index
            edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
            edge_attr_new = self.edge_update_mlp(edge_input)

        return x_new, edge_attr_new

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Create messages by combining source node, target node, and edge features.
        """
        # x_i: target nodes, x_j: source nodes
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(message_input)


class EdgeAttributeConvNet(nn.Module):
    """
    Complete network that initializes edge attributes from node pairs and performs convolution.
    """

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_hidden_dim: int = 8,
        node_hidden_dim: int = 8,
        num_conv_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1,
        batch_norm: bool = True,
        edge_init_layers: int = 3,
    ):
        super(EdgeAttributeConvNet, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

        # Edge attribute initializer (replaces your EdgeFeatureGenerator)
        self.edge_initializer = get_mlp(
            input_dim=2 * node_input_dim + 1,  # src_node + dst_node + distance
            output_dim=edge_hidden_dim,
            num_layers=edge_init_layers,
            hidden_dim=max(2 * node_input_dim + 1, edge_hidden_dim),
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Initial node feature projection
        self.node_projector = get_mlp(
            input_dim=node_input_dim,
            output_dim=node_hidden_dim,
            num_layers=2,
            hidden_dim=node_hidden_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        current_node_dim = node_hidden_dim
        current_edge_dim = edge_hidden_dim

        for i in range(num_conv_layers):
            # Optionally reduce dimensions in later layers
            out_node_dim = node_hidden_dim
            out_edge_dim = edge_hidden_dim

            self.conv_layers.append(
                EdgeConvLayer(
                    input_node_dim=current_node_dim,
                    edge_dim=current_edge_dim,
                    out_node_dim=out_node_dim,
                    out_edge_dim=out_edge_dim,
                    update_edges=(
                        i < num_conv_layers - 1
                    ),  # Don't update edges in last layer
                )
            )

            current_node_dim = out_node_dim
            current_edge_dim = out_edge_dim

        # Final edge classifier
        self.edge_classifier = get_mlp(
            input_dim=current_edge_dim + 2 * current_node_dim,  # edge + both nodes
            output_dim=output_dim,
            num_layers=3,
            hidden_dim=current_edge_dim,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Batch normalization layers
        if batch_norm:
            self.node_batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(layer.out_node_dim) for layer in self.conv_layers]
            )
            self.edge_batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(layer.out_edge_dim) for layer in self.conv_layers[:-1]]
            )
        else:
            self.node_batch_norms = None
            self.edge_batch_norms = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the network.

        Args:
            x: Node features [num_nodes, node_input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes [num_nodes] (optional)

        Returns:
            Edge predictions [num_edges, output_dim]
        """
        # Initialize edge attributes from node pairs
        row, col = edge_index
        distances = torch.norm(x[row] - x[col], p=2, dim=-1, keepdim=True)
        edge_input = torch.cat([x[row], x[col], distances], dim=-1)
        edge_attr = self.edge_initializer(edge_input)

        # Initial node feature processing
        x = self.node_projector(x)

        # Apply convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            x_new, edge_attr_new = conv_layer(x, edge_index, edge_attr)

            # Apply batch normalization and dropout
            if self.node_batch_norms is not None:
                x_new = self.node_batch_norms[i](x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            if i < len(self.conv_layers) - 1 and self.edge_batch_norms is not None:
                edge_attr_new = self.edge_batch_norms[i](edge_attr_new)
                edge_attr_new = F.dropout(
                    edge_attr_new, p=self.dropout, training=self.training
                )

            # Residual connections (if dimensions match)
            if x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new

            if edge_attr.shape[-1] == edge_attr_new.shape[-1]:
                edge_attr = edge_attr + edge_attr_new
            else:
                edge_attr = edge_attr_new

        # Final edge prediction
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_predictions = self.edge_classifier(edge_features)

        return edge_predictions


class EdgeClassificationNet(EdgeAttributeConvNet):
    """
    Specialized version for binary edge classification (e.g., link prediction).
    """

    def __init__(self, **kwargs):
        kwargs["output_dim"] = 1  # Binary classification
        super().__init__(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with sigmoid activation for binary classification.
        """
        logits = super().forward(x, edge_index, batch)
        return torch.sigmoid(logits.squeeze(-1))
