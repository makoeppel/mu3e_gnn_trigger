import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from torch_geometric.nn import MessagePassing


class LearnableEdgeWeight(MessagePassing):
    def __init__(self, nn):
        """
        A GNN layer that learns edge weights based on node features.
        Args:
            nn (nn.Module): A neural network that takes concatenated node features
                           and outputs a scalar edge weight.
        """
        super().__init__(aggr='add')
        self.nn = nn

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col]], dim=-1)
        edge_weight = self.nn(edge_feat).squeeze(-1)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)