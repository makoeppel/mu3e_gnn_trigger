import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax

from torch_geometric.nn import MessagePassing


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


# ===== GNN COMPONENTS =====
class EdgeWeightGenerator(nn.Module):
    """Learns initial edge weights from node pairs."""

    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_feat = torch.cat([x[src], x[dst]], dim=-1)
        edge_weight = self.edge_mlp(edge_feat).squeeze(-1)
        edge_weight = softmax(edge_weight, src)
        return edge_weight


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


class EdgeWeightUpdater(nn.Module):
    """Recomputes edge weights after message passing."""

    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_feat = torch.cat([x[src], x[dst]], dim=-1)
        new_weight = self.edge_mlp(edge_feat).squeeze(-1)
        new_weight = softmax(new_weight, src)
        return new_weight
