import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sys
sys.path.append("../")

ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"/data/dust/group/atlas/ttreco/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"
MODEL_NAME = "multi_class_classification"

SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_with_layer_pixel_spacetime.npy"
BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_with_layer_pixel_spacetime.npy"
SIGNAL_MPPC_FILE = f"{DATA_DIR}/sig_with_layer_mppc_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_with_layer_mppc_spacetime.npy"
SIGNAL_ONLY_PIXEL_FILE = f"{DATA_DIR}/sig_only_with_layer_pixel_spacetime.npy"
SIGNAL_ONLY_MPPC_FILE = f"{DATA_DIR}/sig_only_with_layer_mppc_spacetime.npy"


bg_pixel_spacetime = np.load(BACKGROUND_PIXEL_FILE)
bg_mppc_spacetime = np.load(BACKGROUND_MPPC_FILE)
#sig_pixel_spacetime = np.load(SIGNAL_PIXEL_FILE)
#sig_mppc_spacetime = np.load(SIGNAL_MPPC_FILE)
sig_only_pixel_spacetime = np.load(SIGNAL_ONLY_PIXEL_FILE)
sig_only_mppc_spacetime = np.load(SIGNAL_ONLY_MPPC_FILE)

from sklearn.model_selection import train_test_split

train_bg_pixel, test_bg_pixel, train_bg_mppc, test_bg_mppc = train_test_split(
    bg_pixel_spacetime, bg_mppc_spacetime, test_size=0.2, random_state=42
)
train_sig_pixel, test_sig_pixel, train_sig_mppc, test_sig_mppc = train_test_split(
    sig_only_pixel_spacetime, sig_only_mppc_spacetime, test_size=0.4, random_state=42
)
del bg_pixel_spacetime, bg_mppc_spacetime


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, MessagePassing
from torch_geometric.data import HeteroData, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class BipartiteConv(MessagePassing):
    """Custom bipartite convolution layer for heterogeneous message passing."""
    
    def __init__(self, in_channels_src: int, in_channels_dst: int, out_channels: int):
        super().__init__(aggr='mean')
        
        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels
        
        # Linear transformations for source and destination nodes
        self.lin_src = Linear(in_channels_src, out_channels)
        self.lin_dst = Linear(in_channels_dst, out_channels)
        self.lin_edge = Linear(out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_edge.reset_parameters()
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bipartite convolution.
        
        Args:
            x: Tuple of (source_features, destination_features)
            edge_index: Edge indices [2, num_edges]
        """
        x_src, x_dst = x
        
        # Transform features
        x_src_transformed = self.lin_src(x_src)
        x_dst_transformed = self.lin_dst(x_dst)
        
        # Propagate messages
        out = self.propagate(edge_index, x=(x_src_transformed, x_dst_transformed))
        
        # Add self-connections (transformed destination features)
        out = out + x_dst_transformed
        
        # Final transformation
        out = self.lin_edge(out)
        
        return out
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        """Message function - just pass the source features."""
        return x_j


class HeteroGraphEncoder(nn.Module):
    """Heterogeneous graph encoder that reduces node features from 4D to 2D."""
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        input_dim: int = 4,
        hidden_dim: int = 16,
        latent_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection for each node type
        self.input_projections = nn.ModuleDict({
            node_type: Linear(input_dim, hidden_dim)
            for node_type in node_types
        })
        
        # Heterogeneous convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            conv_dict = {}
            for src_type, rel_type, dst_type in edge_types:
                edge_key = (src_type, rel_type, dst_type)
                if src_type == dst_type:
                    # Homogeneous edge - use SAGEConv
                    conv_dict[edge_key] = SAGEConv(hidden_dim, hidden_dim)
                else:
                    # Bipartite edge - use custom BipartiteConv
                    conv_dict[edge_key] = BipartiteConv(hidden_dim, hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Final encoding layer to latent space
        final_conv_dict = {}
        for src_type, rel_type, dst_type in edge_types:
            edge_key = (src_type, rel_type, dst_type)
            if src_type == dst_type:
                final_conv_dict[edge_key] = SAGEConv(hidden_dim, latent_dim)
            else:
                final_conv_dict[edge_key] = BipartiteConv(hidden_dim, hidden_dim, latent_dim)
        self.final_conv = HeteroConv(final_conv_dict, aggr='mean')
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode heterogeneous graph to latent representations.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            
        Returns:
            Dictionary of latent representations for each node type
        """
        # Input projection
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type].size(0) > 0:
                h_dict[node_type] = F.relu(self.input_projections[node_type](x_dict[node_type]))
            else:
                # Handle empty node types
                device = next(iter(x_dict.values())).device if x_dict else torch.device('cpu')
                h_dict[node_type] = torch.empty(0, self.hidden_dim, device=device)
        
        # Heterogeneous convolutions
        for conv in self.convs:
            h_dict_new = conv(h_dict, edge_index_dict)
            # Apply activation and dropout only to non-empty tensors
            for node_type in h_dict_new:
                if h_dict_new[node_type].size(0) > 0:
                    h_dict_new[node_type] = F.relu(h_dict_new[node_type])
                    h_dict_new[node_type] = self.dropout_layer(h_dict_new[node_type])
            h_dict = h_dict_new
        
        # Final encoding to latent space
        z_dict = self.final_conv(h_dict, edge_index_dict)
        
        # Apply tanh activation to latent representations for better training stability
        for node_type in z_dict:
            if z_dict[node_type].size(0) > 0:
                z_dict[node_type] = torch.tanh(z_dict[node_type])
        
        return z_dict


class HeteroGraphDecoder(nn.Module):
    """Heterogeneous graph decoder that reconstructs node features from 2D to 4D."""
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        latent_dim: int = 2,
        hidden_dim: int = 16,
        output_dim: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First layer from latent to hidden
        first_conv_dict = {}
        for src_type, rel_type, dst_type in edge_types:
            edge_key = (src_type, rel_type, dst_type)
            if src_type == dst_type:
                first_conv_dict[edge_key] = SAGEConv(latent_dim, hidden_dim)
            else:
                first_conv_dict[edge_key] = BipartiteConv(latent_dim, latent_dim, hidden_dim)
        self.first_conv = HeteroConv(first_conv_dict, aggr='mean')
        
        # Hidden heterogeneous convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            conv_dict = {}
            for src_type, rel_type, dst_type in edge_types:
                edge_key = (src_type, rel_type, dst_type)
                if src_type == dst_type:
                    conv_dict[edge_key] = SAGEConv(hidden_dim, hidden_dim)
                else:
                    conv_dict[edge_key] = BipartiteConv(hidden_dim, hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Output projection for each node type
        self.output_projections = nn.ModuleDict({
            node_type: Linear(hidden_dim, output_dim)
            for node_type in node_types
        })
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, z_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Decode latent representations back to original node features.
        
        Args:
            z_dict: Dictionary of latent representations for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            
        Returns:
            Dictionary of reconstructed node features for each node type
        """
        # First convolution from latent to hidden
        h_dict = self.first_conv(z_dict, edge_index_dict)
        
        # Apply activation and dropout
        for node_type in h_dict:
            if h_dict[node_type].size(0) > 0:
                h_dict[node_type] = F.relu(h_dict[node_type])
                h_dict[node_type] = self.dropout_layer(h_dict[node_type])
        
        # Hidden convolutions
        for conv in self.convs:
            h_dict_new = conv(h_dict, edge_index_dict)
            # Apply activation and dropout
            for node_type in h_dict_new:
                if h_dict_new[node_type].size(0) > 0:
                    h_dict_new[node_type] = F.relu(h_dict_new[node_type])
                    h_dict_new[node_type] = self.dropout_layer(h_dict_new[node_type])
            h_dict = h_dict_new
        
        # Output projection
        x_recon_dict = {}
        for node_type in self.node_types:
            if node_type in h_dict and h_dict[node_type].size(0) > 0:
                x_recon_dict[node_type] = self.output_projections[node_type](h_dict[node_type])
            else:
                # Handle empty node types
                device = next(iter(h_dict.values())).device if h_dict else torch.device('cpu')
                x_recon_dict[node_type] = torch.empty(0, self.output_dim, device=device)
        
        return x_recon_dict


class HeteroGraphAutoencoder(nn.Module):
    """Complete heterogeneous graph autoencoder."""
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        input_dim: int = 4,
        hidden_dim: int = 16,
        latent_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = HeteroGraphEncoder(
            node_types=node_types,
            edge_types=edge_types,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = HeteroGraphDecoder(
            node_types=node_types,
            edge_types=edge_types,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, data: HeteroData) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through encoder and decoder.
        
        Args:
            data: HeteroData object containing node features and edge indices
            
        Returns:
            Tuple of (latent_dict, reconstruction_dict)
        """
        # Extract node features and edge indices
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data.node_types and hasattr(data[node_type], 'x'):
                x_dict[node_type] = data[node_type].x
            else:
                # Create empty tensor for missing node types
                device = next(iter([data[nt].x for nt in data.node_types if hasattr(data[nt], 'x')])).device
                x_dict[node_type] = torch.empty(0, self.input_dim, device=device)
        
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type in data.edge_types and hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index
        
        # Only proceed if we have some edges
        if not edge_index_dict:
            # If no edges, return empty dictionaries
            empty_dict = {nt: torch.empty(0, self.latent_dim) for nt in self.node_types}
            empty_recon = {nt: torch.empty(0, self.input_dim) for nt in self.node_types}
            return empty_dict, empty_recon
        
        # Encode
        latent_dict = self.encoder(x_dict, edge_index_dict)
        
        # Decode
        reconstruction_dict = self.decoder(latent_dict, edge_index_dict)
        
        return latent_dict, reconstruction_dict
    
    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Encode graph to latent representation."""
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data.node_types and hasattr(data[node_type], 'x'):
                x_dict[node_type] = data[node_type].x
            else:
                device = next(iter([data[nt].x for nt in data.node_types if hasattr(data[nt], 'x')])).device
                x_dict[node_type] = torch.empty(0, self.input_dim, device=device)
        
        edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type in data.edge_types and hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index
        
        if not edge_index_dict:
            return {nt: torch.empty(0, self.latent_dim) for nt in self.node_types}
        
        return self.encoder(x_dict, edge_index_dict)
    
    def decode(self, latent_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """Decode latent representation back to node features."""
        return self.decoder(latent_dict, edge_index_dict)


class HeteroGraphAutoencoderLoss(nn.Module):
    """Loss function for heterogeneous graph autoencoder."""
    
    def __init__(
        self,
        node_types: List[str],
        reconstruction_weight: float = 1.0,
        regularization_weight: float = 0.01,
        node_type_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.node_types = node_types
        self.reconstruction_weight = reconstruction_weight
        self.regularization_weight = regularization_weight
        self.node_type_weights = node_type_weights or {nt: 1.0 for nt in node_types}
        
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        original_dict: Dict[str, torch.Tensor],
        reconstruction_dict: Dict[str, torch.Tensor],
        latent_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss components.
        
        Args:
            original_dict: Original node features
            reconstruction_dict: Reconstructed node features
            latent_dict: Latent representations
            
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        total_reconstruction_loss = 0.0
        total_regularization_loss = 0.0
        total_nodes = 0
        
        # Reconstruction loss for each node type
        for node_type in self.node_types:
            if (node_type in original_dict and node_type in reconstruction_dict and
                original_dict[node_type].size(0) > 0 and reconstruction_dict[node_type].size(0) > 0):
                
                # Per-node reconstruction loss
                node_loss = self.mse_loss(
                    reconstruction_dict[node_type], 
                    original_dict[node_type]
                ).mean(dim=1)  # Average over features
                
                # Weight by node type importance
                weighted_loss = node_loss * self.node_type_weights[node_type]
                losses[f'{node_type}_reconstruction'] = weighted_loss.mean()
                
                total_reconstruction_loss += weighted_loss.sum()
                total_nodes += node_loss.size(0)
        
        # Regularization loss (encourage compact latent representations)
        num_latent_types = 0
        for node_type in self.node_types:
            if node_type in latent_dict and latent_dict[node_type].size(0) > 0:
                # L2 regularization on latent representations
                reg_loss = (latent_dict[node_type] ** 2).mean()
                losses[f'{node_type}_regularization'] = reg_loss
                total_regularization_loss += reg_loss
                num_latent_types += 1
        
        # Compute total loss
        if total_nodes > 0:
            avg_reconstruction_loss = total_reconstruction_loss / total_nodes
        else:
            device = next(iter(reconstruction_dict.values())).device if reconstruction_dict else torch.device('cpu')
            avg_reconstruction_loss = torch.tensor(0.0, device=device)
        
        if num_latent_types > 0:
            avg_regularization_loss = total_regularization_loss / num_latent_types
        else:
            device = next(iter(reconstruction_dict.values())).device if reconstruction_dict else torch.device('cpu')
            avg_regularization_loss = torch.tensor(0.0, device=device)
        
        losses['reconstruction_loss'] = avg_reconstruction_loss
        losses['regularization_loss'] = avg_regularization_loss
        losses['total_loss'] = (
            self.reconstruction_weight * avg_reconstruction_loss +
            self.regularization_weight * avg_regularization_loss
        )
        
        return losses


class HeteroGraphAutoencoderTrainer:
    """Trainer class for heterogeneous graph autoencoder."""
    
    def __init__(
        self,
        model: HeteroGraphAutoencoder,
        loss_fn: HeteroGraphAutoencoderLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        try:
            # Forward pass
            latent_dict, reconstruction_dict = self.model(batch)
            
            # Prepare original features
            original_dict = {}
            for node_type in self.model.node_types:
                if node_type in batch.node_types and hasattr(batch[node_type], 'x'):
                    original_dict[node_type] = batch[node_type].x
                else:
                    original_dict[node_type] = torch.empty(0, self.model.input_dim, device=self.device)
            
            # Compute loss
            loss_dict = self.loss_fn(original_dict, reconstruction_dict, latent_dict)
            
            # Backward pass
            if loss_dict['total_loss'].item() > 0:  # Only backprop if loss is non-zero
                loss_dict['total_loss'].backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
        except Exception as e:
            print(f"Error in training step: {e}")
            # Return zero losses if there's an error
            loss_dict = {f'{nt}_reconstruction': 0.0 for nt in self.model.node_types}
            loss_dict.update({f'{nt}_regularization': 0.0 for nt in self.model.node_types})
            loss_dict.update({'reconstruction_loss': 0.0, 'regularization_loss': 0.0, 'total_loss': 0.0})
        
        # Convert to float for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def val_step(self, batch: Batch) -> Dict[str, float]:
        """Single validation step."""
        self.model.eval()
        
        batch = batch.to(self.device)
        
        with torch.no_grad():
            try:
                # Forward pass
                latent_dict, reconstruction_dict = self.model(batch)
                
                # Prepare original features
                original_dict = {}
                for node_type in self.model.node_types:
                    if node_type in batch.node_types and hasattr(batch[node_type], 'x'):
                        original_dict[node_type] = batch[node_type].x
                    else:
                        original_dict[node_type] = torch.empty(0, self.model.input_dim, device=self.device)
                
                # Compute loss
                loss_dict = self.loss_fn(original_dict, reconstruction_dict, latent_dict)
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                # Return zero losses if there's an error
                loss_dict = {f'{nt}_reconstruction': 0.0 for nt in self.model.node_types}
                loss_dict.update({f'{nt}_regularization': 0.0 for nt in self.model.node_types})
                loss_dict.update({'reconstruction_loss': 0.0, 'regularization_loss': 0.0, 'total_loss': 0.0})
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = defaultdict(list)
        
        for batch in tqdm(dataloader, desc="Training"):
            step_losses = self.train_step(batch)
            for k, v in step_losses.items():
                epoch_losses[k].append(v)
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.train_losses.append(avg_losses['total_loss'])
        
        return avg_losses
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        epoch_losses = defaultdict(list)
        
        for batch in dataloader:
            step_losses = self.val_step(batch)
            for k, v in step_losses.items():
                epoch_losses[k].append(v)
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.val_losses.append(avg_losses['total_loss'])
        
        return avg_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        print_every: int = 10
    ):
        """Full training loop."""
        for epoch in range(epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = {}
            if val_loader is not None:
                val_losses = self.validate_epoch(val_loader)
            
            # Logging
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            if val_losses:
                print(f"  Val Loss: {val_losses['total_loss']:.4f}")
                
                # Print component losses
            for loss_name, loss_value in train_losses.items():
                if loss_name != 'total_loss' and not loss_name.endswith('_reconstruction') and not loss_name.endswith('_regularization'):
                    print(f"    {loss_name}: {loss_value:.4f}")


# Factory function to create model for layer-separated graphs
def create_layer_separated_autoencoder(
    input_dim: int = 4,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    num_layers: int = 3,
    dropout: float = 0.1,
    device: torch.device = torch.device('cpu')
) -> Tuple[HeteroGraphAutoencoder, HeteroGraphAutoencoderLoss]:
    """
    Create autoencoder and loss function for layer-separated heterogeneous graphs.
    
    Returns:
        Tuple of (model, loss_function)
    """
    # Define node and edge types for layer-separated graphs
    node_types = ["layer_1", "layer_2", "layer_3", "layer_4", "mppc"]
    edge_types = [
        ("layer_1", "to", "layer_2"),
        ("layer_2", "to", "layer_1"), 
        ("layer_2", "to", "mppc"),
        ("mppc", "to", "layer_2"),
        ("mppc", "to", "mppc"),
        ("mppc", "to", "layer_3"),
        ("layer_3", "to", "mppc"),
        ("layer_3", "to", "layer_4"),
        ("layer_4", "to", "layer_3")
    ]
    
    # Create model
    model = HeteroGraphAutoencoder(
        node_types=node_types,
        edge_types=edge_types,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create loss function with balanced weights
    node_type_weights = {
        "layer_1": 1.0,
        "layer_2": 1.2,  # Slightly higher weight for central layers
        "layer_3": 1.2,
        "layer_4": 1.0,
        "mppc": 1.5      # Higher weight for MPPC (includes timing info)
    }
    
    loss_fn = HeteroGraphAutoencoderLoss(
        node_types=node_types,
        reconstruction_weight=1.0,
        regularization_weight=0.01,
        node_type_weights=node_type_weights
    )
    
    return model.to(device), loss_fn


import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

from src.torch.pre_processing import LayerSeparatedHeteroGraphBuilder


class LayerSeparatedGraphAutoencoderPipeline:
    """Complete pipeline for layer-separated heterogeneous graph processing."""
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 16,
        latent_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1,
        mppc_timing_cutoff: float = 0.2,
        device: torch.device = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Create graph builder
        self.graph_builder = LayerSeparatedHeteroGraphBuilder(
            connect_layers=True,
            mppc_timing_cutoff=mppc_timing_cutoff
        )
        
        # Create autoencoder model and loss
        self.model, self.loss_fn = create_layer_separated_autoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=self.device
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.8)
        
        # Create trainer
        self.trainer = HeteroGraphAutoencoderTrainer(
            self.model, self.loss_fn, self.optimizer, self.device
        )
        
        print(f"Pipeline initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def prepare_node_features(self, pixel_data: torch.Tensor, mppc_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare node features for the autoencoder.
        Assumes input data has [x, y, z, layer, track, time] format.
        We'll use [x, y, z, normalized_layer] as the 4D features.
        """
        # Pixel features: [x, y, z, normalized_layer]
        pixel_features = pixel_data[:, :, [0, 1, 2, 3]]  # Take x, y, z, layer
        pixel_features[:, :, 3] = pixel_features[:, :, 3] / 4.0  # Normalize layer (1-4 -> 0.25-1.0)
        
        # MPPC features: [x, y, z, normalized_time] 
        mppc_features = mppc_data[:, :, [0, 1, 2, 5]]  # Take x, y, z, time
        # Normalize time to [0, 1] range
        valid_mask = mppc_features[:, :, 3] != -1
        if valid_mask.any():
            time_values = mppc_features[:, :, 3][valid_mask]
            time_min, time_max = time_values.min(), time_values.max()
            if time_max > time_min:
                mppc_features[:, :, 3][valid_mask] = (time_values - time_min) / (time_max - time_min)
            else:
                mppc_features[:, :, 3][valid_mask] = 0.5  # If all times are the same
        
        return pixel_features, mppc_features

    def create_graph_dataset(
        self, 
        pixel_data: torch.Tensor, 
        mppc_data: torch.Tensor,
        max_graphs_per_event: int = 5
    ) -> List:
        """Create dataset of graphs from event data."""
        # Prepare features
        pixel_features, mppc_features = self.prepare_node_features(pixel_data, mppc_data)
        
        all_graphs = []
        
        for event_idx in range(pixel_data.shape[0]):
            # Create spacetime data format expected by graph builder
            pixel_spacetime = torch.cat([
                pixel_data[event_idx, :, :3],  # x, y, z
                pixel_data[event_idx, :, 3:4],  # layer
                pixel_data[event_idx, :, 4:5],  # track
                pixel_data[event_idx, :, 5:6],  # time
            ], dim=1)
            
            mppc_spacetime = torch.cat([
                mppc_data[event_idx, :, :3],   # x, y, z
                mppc_data[event_idx, :, 3:4],  # layer (should be 2.5)
                mppc_data[event_idx, :, 4:5],  # track
                mppc_data[event_idx, :, 5:6],  # time
            ], dim=1)
            
            # Build graphs for this event
            event_graphs = self.graph_builder.build_graphs_from_event(
                mppc_spacetime, pixel_spacetime
            )
            
            # Add features to graphs and limit number of graphs
            for i, graph in enumerate(event_graphs[:max_graphs_per_event]):
                # Add 4D features to each node type
                for node_type in graph.node_types:
                    if hasattr(graph[node_type], 'x'):
                        original_features = graph[node_type].x
                        
                        if node_type.startswith('layer'):
                            # For layer nodes, use spatial coordinates + normalized layer
                            layer_id = int(node_type.split('_')[1])
                            features = torch.cat([
                                original_features[:, :3],  # x, y, z
                                torch.full((original_features.shape[0], 1), layer_id / 4.0)
                            ], dim=1)
                        elif node_type == 'mppc':
                            # For MPPC nodes, use spatial coordinates + normalized time
                            if original_features.shape[1] >= 4:
                                time_feature = original_features[:, 3:4]  # Already includes time
                                # Normalize time feature
                                time_min, time_max = time_feature.min(), time_feature.max()
                                if time_max > time_min:
                                    time_normalized = (time_feature - time_min) / (time_max - time_min)
                                else:
                                    time_normalized = torch.full_like(time_feature, 0.5)
                                
                                features = torch.cat([
                                    original_features[:, :3],  # x, y, z
                                    time_normalized
                                ], dim=1)
                            else:
                                # Fallback if no time info
                                features = torch.cat([
                                    original_features[:, :3],
                                    torch.full((original_features.shape[0], 1), 0.5)
                                ], dim=1)
                        
                        graph[node_type].x = features
                
                all_graphs.append(graph)
        
        print(f"Created {len(all_graphs)} graphs from {pixel_data.shape[0]} events")
        return all_graphs

    def train_model(
        self, 
        graphs: List, 
        validation_split: float = 0.2,
        batch_size: int = 8,
        epochs: int = 100,
        print_every: int = 10
    ):
        """Train the autoencoder model."""
        # Split into train/val
        n_val = int(len(graphs) * validation_split)
        indices = torch.randperm(len(graphs))
        
        train_graphs = [graphs[i] for i in indices[n_val:]]
        val_graphs = [graphs[i] for i in indices[:n_val]]
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False) if val_graphs else None
        
        print(f"Training on {len(train_graphs)} graphs, validating on {len(val_graphs)} graphs")
        
        # Train
        self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            print_every=print_every
        )
        
        return train_loader, val_loader

    def visualize_reconstruction(self, graphs: List, num_examples: int = 3):
        """Visualize reconstruction quality."""
        self.model.eval()
        
        with torch.no_grad():
            for i in range(min(num_examples, len(graphs))):
                graph = graphs[i].to(self.device)
                
                # Get reconstructions
                latent_dict, recon_dict = self.model(graph)
                
                print(f"\nGraph {i+1}:")
                
                for node_type in graph.node_types:
                    if hasattr(graph[node_type], 'x') and graph[node_type].x.shape[0] > 0:
                        original = graph[node_type].x.cpu()
                        reconstructed = recon_dict[node_type].cpu()
                        latent = latent_dict[node_type].cpu()
                        
                        # Compute reconstruction error
                        mse = F.mse_loss(reconstructed, original)
                        
                        print(f"  {node_type}: {original.shape[0]} nodes")
                        print(f"    MSE: {mse:.4f}")
                        print(f"    Original range: [{original.min():.3f}, {original.max():.3f}]")
                        print(f"    Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
                        print(f"    Latent range: [{latent.min():.3f}, {latent.max():.3f}]")

    def analyze_latent_space(self, graphs: List, max_graphs: int = 100):
        """Analyze the learned latent space."""
        self.model.eval()
        
        latent_data = {node_type: [] for node_type in self.model.node_types}
        
        with torch.no_grad():
            for i, graph in enumerate(graphs[:max_graphs]):
                graph = graph.to(self.device)
                latent_dict, _ = self.model(graph)
                
                for node_type in latent_dict:
                    if latent_dict[node_type].shape[0] > 0:
                        latent_data[node_type].append(latent_dict[node_type].cpu())
        
        # Concatenate all latent representations
        for node_type in latent_data:
            if latent_data[node_type]:
                latent_data[node_type] = torch.cat(latent_data[node_type], dim=0)
                print(f"{node_type}: {latent_data[node_type].shape[0]} nodes in latent space")
            else:
                print(f"{node_type}: No data")
        
        return latent_data

    def plot_latent_space(self, latent_data: Dict[str, torch.Tensor]):
        """Plot 2D latent space for each node type."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (node_type, data) in enumerate(latent_data.items()):
            if i < len(axes) and len(data) > 0:
                ax = axes[i]
                ax.scatter(data[:, 0], data[:, 1], c=colors[i % len(colors)], alpha=0.6, s=10)
                ax.set_title(f'{node_type} Latent Space')
                ax.set_xlabel('Latent Dim 1')
                ax.set_ylabel('Latent Dim 2')
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for j in range(len(latent_data), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()


def create_sample_data(
    n_events: int = 50,
    n_pixel_hits: int = 32,
    n_mppc_hits: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample data for testing."""
    
    # Create pixel data [x, y, z, layer, track, time]
    pixel_data = torch.zeros(n_events, n_pixel_hits, 6)
    
    for event_idx in range(n_events):
        n_valid_pixels = torch.randint(8, n_pixel_hits, (1,)).item()
        
        # Spatial coordinates
        pixel_data[event_idx, :n_valid_pixels, :3] = torch.randn(n_valid_pixels, 3) * 10
        
        # Layer assignments (1-4)
        pixel_data[event_idx, :n_valid_pixels, 3] = torch.randint(1, 5, (n_valid_pixels,)).float()
        
        # Track IDs
        pixel_data[event_idx, :n_valid_pixels, 4] = torch.randint(1, 6, (n_valid_pixels,)).float()
        
        # Time stamps
        pixel_data[event_idx, :n_valid_pixels, 5] = torch.randint(0, 64, (n_valid_pixels,)).float()
        
        # Pad invalid entries
        pixel_data[event_idx, n_valid_pixels:] = -1
    
    # Create MPPC data [x, y, z, layer=2.5, track, time]
    mppc_data = torch.zeros(n_events, n_mppc_hits, 6)
    
    for event_idx in range(n_events):
        n_valid_mppc = torch.randint(4, n_mppc_hits, (1,)).item()
        
        # Spatial coordinates
        mppc_data[event_idx, :n_valid_mppc, :3] = torch.randn(n_valid_mppc, 3) * 8
        
        # Layer (always 2.5 for MPPC)
        mppc_data[event_idx, :n_valid_mppc, 3] = 2.5
        
        # Track IDs (should match some pixel tracks)
        mppc_data[event_idx, :n_valid_mppc, 4] = torch.randint(1, 6, (n_valid_mppc,)).float()
        
        # Time stamps (MPPC timing)
        mppc_data[event_idx, :n_valid_mppc, 5] = torch.randint(0, 64, (n_valid_mppc,)).float() * 8
        
        # Pad invalid entries
        mppc_data[event_idx, n_valid_mppc:] = -1
    
    return pixel_data, mppc_data




"""Run the complete pipeline example."""
print("Creating sample data...")

print("Initializing pipeline...")
pipeline = LayerSeparatedGraphAutoencoderPipeline(
    input_dim=4,
    hidden_dim=12,
    latent_dim=2,
    num_layers=3,
    dropout=0.1
)

print("Creating graph dataset...")
train_pixel_data = torch.tensor(train_bg_pixel,dtype=torch.float32)
train_mppc_data = torch.tensor(train_bg_mppc,dtype=torch.float32)
graphs = pipeline.create_graph_dataset(train_pixel_data, train_mppc_data)
del train_pixel_data
del train_mppc_data


pipeline.train_model(
    graphs,
    validation_split=0.2,
    batch_size=512,
    epochs=20,
    print_every=1
)

trained_model = pipeline.model


import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt


def compute_anomaly_scores(
    signal_pixel_data: torch.Tensor,
    signal_mppc_data: torch.Tensor,
    background_pixel_data: torch.Tensor,
    background_mppc_data: torch.Tensor,
    model,
    graph_builder=None,
    device: torch.device = None,
    score_type: str = 'reconstruction',
    node_weights: Optional[Dict[str, float]] = None,
    max_graphs_per_event: int = 5
) -> Dict[str, np.ndarray]:
    """
    Lightweight function to compute anomaly scores for signal vs background data.
    
    Args:
        signal_pixel_data: Signal pixel data [batch, hits, 6] with [x,y,z,layer,track,time]
        signal_mppc_data: Signal MPPC data [batch, hits, 6] with [x,y,z,layer,track,time]  
        background_pixel_data: Background pixel data [batch, hits, 6]
        background_mppc_data: Background MPPC data [batch, hits, 6]
        model: Trained heterogeneous graph autoencoder
        graph_builder: Graph builder (will create default if None)
        device: Compute device
        score_type: 'reconstruction', 'weighted', 'latent', or 'combined'
        node_weights: Custom weights for node types
        max_graphs_per_event: Maximum graphs per event to process
        
    Returns:
        Dictionary with 'signal_scores' and 'background_scores' as numpy arrays
    """
    
    # Setup
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Default graph builder
    if graph_builder is None:
        graph_builder = LayerSeparatedHeteroGraphBuilder(connect_layers=True)
    
    # Default node weights
    if node_weights is None:
        node_weights = {
            "layer_1": 1.0, "layer_2": 1.2, "layer_3": 1.2, 
            "layer_4": 1.0, "mppc": 1.5
        }
    
    print(f"Processing signal: {signal_pixel_data.shape[0]} events")
    print(f"Processing background: {background_pixel_data.shape[0]} events")
    
    # Process signal data
    signal_scores = _process_data_batch(
        signal_pixel_data, signal_mppc_data, model, graph_builder,
        device, score_type, node_weights, max_graphs_per_event
    )
    
    # Process background data  
    background_scores = _process_data_batch(
        background_pixel_data, background_mppc_data, model, graph_builder,
        device, score_type, node_weights, max_graphs_per_event
    )
    
    print(f"Generated {len(signal_scores)} signal scores")
    print(f"Generated {len(background_scores)} background scores")
    
    return {
        'signal_scores': np.array(signal_scores),
        'background_scores': np.array(background_scores)
    }


def _process_data_batch(
    pixel_data: torch.Tensor,
    mppc_data: torch.Tensor, 
    model,
    graph_builder,
    device: torch.device,
    score_type: str,
    node_weights: Dict[str, float],
    max_graphs_per_event: int
) -> List[float]:
    """Process a batch of events and return anomaly scores."""
    
    all_scores = []
    
    with torch.no_grad():
        for event_idx in range(pixel_data.shape[0]):
            # Extract single event
            pixel_event = pixel_data[event_idx]
            mppc_event = mppc_data[event_idx]
            
            # Create graphs for this event
            try:
                graphs = graph_builder.build_graphs_from_event(mppc_event, pixel_event)
                
                # Limit number of graphs per event
                graphs = graphs[:max_graphs_per_event]
                
                # Process each graph
                for graph in graphs:
                    # Add 4D node features
                    graph = _prepare_graph_features(graph)
                    graph = graph.to(device)
                    
                    # Compute anomaly score
                    score = _compute_single_graph_score(
                        graph, model, score_type, node_weights
                    )
                    
                    if np.isfinite(score):
                        all_scores.append(score)
                        
            except Exception as e:
                # Skip problematic events
                continue
    
    return all_scores


def _prepare_graph_features(graph: HeteroData) -> HeteroData:
    """Add 4D features to graph nodes."""
    
    for node_type in graph.node_types:
        if hasattr(graph[node_type], 'x') and graph[node_type].x.shape[0] > 0:
            original_features = graph[node_type].x
            
            if node_type.startswith('layer'):
                # For layer nodes: [x, y, z, normalized_layer]
                layer_id = int(node_type.split('_')[1])
                features = torch.cat([
                    original_features[:, :3],  # x, y, z
                    torch.full((original_features.shape[0], 1), layer_id / 4.0)
                ], dim=1)
                
            elif node_type == 'mppc':
                # For MPPC nodes: [x, y, z, normalized_time]
                if original_features.shape[1] >= 4:
                    time_feature = original_features[:, 3:4]
                    # Simple normalization
                    time_min, time_max = time_feature.min(), time_feature.max()
                    if time_max > time_min:
                        time_normalized = (time_feature - time_min) / (time_max - time_min)
                    else:
                        time_normalized = torch.full_like(time_feature, 0.5)
                    
                    features = torch.cat([
                        original_features[:, :3],  # x, y, z
                        time_normalized
                    ], dim=1)
                else:
                    # Fallback
                    features = torch.cat([
                        original_features[:, :3],
                        torch.full((original_features.shape[0], 1), 0.5)
                    ], dim=1)
            
            # Ensure features are 4D
            if features.shape[1] != 4:
                # Pad or truncate to 4D
                if features.shape[1] < 4:
                    padding = torch.zeros(features.shape[0], 4 - features.shape[1])
                    features = torch.cat([features, padding], dim=1)
                else:
                    features = features[:, :4]
            
            graph[node_type].x = features
    
    return graph


def _compute_single_graph_score(
    graph: HeteroData,
    model,
    score_type: str,
    node_weights: Dict[str, float]
) -> float:
    """Compute anomaly score for a single graph."""
    
    try:
        # Forward pass
        latent_dict, reconstruction_dict = model(graph)
        
        # Extract original features
        original_dict = {}
        for node_type in model.node_types:
            if node_type in graph.node_types and hasattr(graph[node_type], 'x'):
                original_dict[node_type] = graph[node_type].x
        
        # Compute score based on type
        if score_type == 'reconstruction':
            score = _reconstruction_error(original_dict, reconstruction_dict)
        elif score_type == 'weighted':
            score = _weighted_reconstruction_error(original_dict, reconstruction_dict, node_weights)
        elif score_type == 'latent':
            score = _latent_magnitude(latent_dict)
        elif score_type == 'combined':
            recon = _reconstruction_error(original_dict, reconstruction_dict)
            latent = _latent_magnitude(latent_dict)
            score = 0.8 * recon + 0.2 * latent
        else:
            raise ValueError(f"Unknown score type: {score_type}")
        
        return score
        
    except Exception as e:
        return 0.0  # Default score for failed graphs


def _reconstruction_error(original_dict: Dict[str, torch.Tensor], 
                         reconstruction_dict: Dict[str, torch.Tensor]) -> float:
    """Compute reconstruction error."""
    total_error = 0.0
    total_nodes = 0
    
    for node_type in original_dict:
        if (node_type in reconstruction_dict and 
            original_dict[node_type].size(0) > 0 and 
            reconstruction_dict[node_type].size(0) > 0):
            
            mse = F.mse_loss(reconstruction_dict[node_type], 
                           original_dict[node_type], reduction='mean')
            num_nodes = original_dict[node_type].size(0)
            total_error += mse.item() * num_nodes
            total_nodes += num_nodes
    
    return total_error / total_nodes if total_nodes > 0 else 0.0


def _weighted_reconstruction_error(original_dict: Dict[str, torch.Tensor], 
                                 reconstruction_dict: Dict[str, torch.Tensor],
                                 node_weights: Dict[str, float]) -> float:
    """Compute weighted reconstruction error."""
    total_error = 0.0
    total_weight = 0.0
    
    for node_type in original_dict:
        if (node_type in reconstruction_dict and 
            original_dict[node_type].size(0) > 0 and 
            reconstruction_dict[node_type].size(0) > 0):
            
            mse = F.mse_loss(reconstruction_dict[node_type], 
                           original_dict[node_type], reduction='mean')
            weight = node_weights.get(node_type, 1.0)
            num_nodes = original_dict[node_type].size(0)
            
            total_error += mse.item() * weight * num_nodes
            total_weight += weight * num_nodes
    
    return total_error / total_weight if total_weight > 0 else 0.0


def _latent_magnitude(latent_dict: Dict[str, torch.Tensor]) -> float:
    """Compute latent magnitude."""
    total_magnitude = 0.0
    total_nodes = 0
    
    for node_type, latent in latent_dict.items():
        if latent.size(0) > 0:
            magnitude = torch.norm(latent, dim=1).mean().item()
            total_magnitude += magnitude * latent.size(0)
            total_nodes += latent.size(0)
    
    return total_magnitude / total_nodes if total_nodes > 0 else 0.0


def plot_score_distributions(score_dict: Dict[str, np.ndarray], 
                           title: str = "Anomaly Score Distributions",
                           bins: int = 50,
                           figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot histograms of signal vs background anomaly scores.
    
    Args:
        score_dict: Dictionary with 'signal_scores' and 'background_scores'
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms
    ax.hist(score_dict['background_scores'], bins=bins, alpha=0.7, 
             label=f'Background (n={len(score_dict["background_scores"])})', 
             color='blue', density=True)
    ax.hist(score_dict['signal_scores'], bins=bins, alpha=0.7, 
             label=f'Signal (n={len(score_dict["signal_scores"])})', 
             color='red', density=True)
    
    # Add statistics
    bg_mean = np.mean(score_dict['background_scores'])
    bg_std = np.std(score_dict['background_scores'])
    sig_mean = np.mean(score_dict['signal_scores'])
    sig_std = np.std(score_dict['signal_scores'])
    
    ax.axvline(bg_mean, color='blue', linestyle='--', alpha=0.8, 
                label=f'Bg mean: {bg_mean:.3f}±{bg_std:.3f}')
    ax.axvline(sig_mean, color='red', linestyle='--', alpha=0.8, 
                label=f'Sig mean: {sig_mean:.3f}±{sig_std:.3f}')
    
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


    # Print separation metrics
    separation = abs(sig_mean - bg_mean) / ((bg_std + sig_std) / 2)
    print(f"\nSeparation Score: {separation:.3f}")
    print(f"Background: {bg_mean:.3f} ± {bg_std:.3f}")
    print(f"Signal: {sig_mean:.3f} ± {sig_std:.3f}")

    return fig, ax

def quick_anomaly_analysis(
    signal_pixel_data: torch.Tensor,
    signal_mppc_data: torch.Tensor,
    background_pixel_data: torch.Tensor,
    background_mppc_data: torch.Tensor,
    model,
    score_types: List[str] = ['reconstruction', 'weighted'],
    plot: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Quick analysis with multiple score types and automatic plotting.
    
    Args:
        signal_pixel_data: Signal pixel data
        signal_mppc_data: Signal MPPC data  
        background_pixel_data: Background pixel data
        background_mppc_data: Background MPPC data
        model: Trained autoencoder model
        score_types: List of score types to compute
        plot: Whether to generate plots
        
    Returns:
        Dictionary of score dictionaries for each score type
    """
    
    all_results = {}
    
    for score_type in score_types:
        print(f"\n=== Computing {score_type} scores ===")
        
        scores = compute_anomaly_scores(
            signal_pixel_data=signal_pixel_data,
            signal_mppc_data=signal_mppc_data,
            background_pixel_data=background_pixel_data,
            background_mppc_data=background_mppc_data,
            model=model,
            score_type=score_type
        )
        
        all_results[score_type] = scores
        
        if plot:
            plot_score_distributions(
                scores, 
                title=f"{score_type.capitalize()} Score Distributions"
            )
    
    return all_results



# Compute anomaly scores
scores = compute_anomaly_scores(
    signal_pixel_data=test_sig_pixel,
    signal_mppc_data=test_sig_mppc, 
    background_pixel_data=test_bg_pixel,
    background_mppc_data=test_bg_mppc,
    model=trained_model,
    score_type='weighted'  # or 'reconstruction', 'latent', 'combined'
)

# Plot distributions
fig, ax = plot_score_distributions(scores)
fig.savefig(f"{PLOTS_DIR}/weighted_anomaly_scores.png")

# Quick analysis with multiple score types
all_scores = quick_anomaly_analysis(
    signal_pixel_data=test_sig_pixel,
    signal_mppc_data=test_sig_mppc, 
    background_pixel_data=test_bg_pixel,
    background_mppc_data=test_bg_mppc,
    model=trained_model,
    score_types=['reconstruction', 'weighted', 'combined']
)

for score_type, score_dict in all_scores.items():
    fig, ax = plot_score_distributions(score_dict, title=f"{score_type.capitalize()} Score Distributions")
    fig.savefig(f"{PLOTS_DIR}/{score_type}_anomaly_scores.png")