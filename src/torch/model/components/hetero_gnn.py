import torch
from torch import nn
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.typing import Adj, OptTensor, PairTensor

class EdgeAttrSAGEConv(SAGEConv):
    def __init__(self, in_channels, out_channels, edge_dim=None, **kwargs):
        """
        Extended SAGEConv that supports edge attributes.

        Args:
            in_channels (int or tuple): Input feature size(s).
            out_channels (int): Output feature size.
            edge_dim (int, optional): Edge feature size. If set, edge attributes
                                      will be projected and added into messages.
        """
        super().__init__(in_channels, out_channels, **kwargs)

        if edge_dim is not None:
            self.edge_lin = nn.Linear(edge_dim, out_channels, bias=False)
        else:
            self.edge_lin = None

    def message(self, x_j: torch.Tensor, edge_attr: OptTensor = None) -> torch.Tensor:
        """
        Build messages from neighbors.
        x_j: neighbor node features
        edge_attr: edge features [num_edges, edge_dim]
        """
        msg = x_j
        if edge_attr is not None and self.edge_lin is not None:
            msg = msg + self.edge_lin(edge_attr)
        return msg
