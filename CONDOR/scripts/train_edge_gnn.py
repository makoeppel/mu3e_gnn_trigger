import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append("../")
from tqdm import tqdm


ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
DATA_DIR = f"/data/dust/group/atlas/ttreco/mu3e_trigger_data"
PLOTS_DIR = f"{ROOT_DIR}/plots"
MODEL_DIR = f"{ROOT_DIR}/models"
MODEL_NAME = "multi_class_classification"

SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_with_layer_pixel_spacetime.npy"
SIGNAL_MPPC_FILE = f"{DATA_DIR}/sig_with_layer_mppc_spacetime.npy"

BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_with_layer_pixel_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_with_layer_mppc_spacetime.npy"


sig_mppc_spacetime = np.load(SIGNAL_MPPC_FILE)
sig_pixel_spacetime = np.load(SIGNAL_PIXEL_FILE)
bg_pixel_spacetime = np.load(BACKGROUND_PIXEL_FILE)
bg_mppc_spacetime = np.load(BACKGROUND_MPPC_FILE)


X_pixel = np.concatenate([sig_pixel_spacetime, bg_pixel_spacetime], axis=0)
X_mppc = np.concatenate([sig_mppc_spacetime, bg_mppc_spacetime], axis=0)
y = np.concatenate(
    [np.ones(sig_pixel_spacetime.shape[0]), np.zeros(bg_pixel_spacetime.shape[0])],
    axis=0,
)
del (
    sig_mppc_spacetime,
    sig_pixel_spacetime,
    bg_pixel_spacetime,
    bg_mppc_spacetime,
)

from sklearn.model_selection import train_test_split


X_pixel_train, X_pixel_test, X_mppc_train, X_mppc_test, y_train, y_test = (
    train_test_split(X_pixel, X_mppc, y, test_size=0.2, random_state=42, stratify=y)
)


del (
    X_pixel,
    X_mppc,
    y,
)
import src.torch.pre_processing.graph_batching as gc
from importlib import reload
import pickle

reload(gc)

from torch_geometric.loader import DataLoader

event_processor = gc.EventProcessor(gc.HeteroGraphBuilder())

hetero_graph_train = event_processor.process_to_graphs(
    X_pixel=X_pixel_train, X_mppc=X_mppc_train, labels=y_train
)
hetero_graph_test = event_processor.process_to_graphs(
    X_pixel=X_pixel_test, X_mppc=X_mppc_test, labels=y_test
)
from torch_geometric.data import Dataset

class HeteroGraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


train_dataset = HeteroGraphDataset(hetero_graph_train)
test_dataset = HeteroGraphDataset(hetero_graph_test)
#torch.save(train_dataset, f"{DATA_DIR}/hetero_graph_train.pt")
#torch.save(test_dataset, f"{DATA_DIR}/hetero_graph_test.pt")

from torch_geometric.loader import DataLoader

train_loader = DataLoader(hetero_graph_train, batch_size=512, shuffle=True)
test_loader = DataLoader(hetero_graph_test, batch_size=512, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, TransformerConv, LayerNorm
import math

class PixelMPPCEdgeClassifier(torch.nn.Module):
    """
    Specialized edge classifier for pixel-MPPC detector systems.
    Handles heterogeneous graphs with pixel and MPPC nodes.
    """
    
    def __init__(self, hidden_channels=64, num_layers=3, dropout=0.15, 
                 num_heads=4, use_attention=True, temperature=1.0):
        super().__init__()
        
        # Configuration
        self.node_dims = {"pixel": 3, "mppc": 4}
        self.edge_types = [
            ("pixel", "to", "pixel"),  # pixel-pixel connections
            ("mppc", "to", "mppc"),    # mppc-mppc connections  
            ("pixel", "to", "mppc"),   # pixel-mppc connections
            ("mppc", "to", "pixel"),   # mppc-pixel connections
        ]
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.temperature = temperature  # For calibrated predictions
        
        # Node-specific feature encoders
        self.pixel_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels),
            nn.ReLU()
        )
        
        self.mppc_encoder = nn.Sequential(
            nn.Linear(4, hidden_channels // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels),
            nn.ReLU()
        )
        
        # Layer normalization for initial features
        self.pixel_norm = LayerNorm(hidden_channels)
        self.mppc_norm = LayerNorm(hidden_channels)
        
        # Edge type specific embeddings (learned representations for each connection type)
        self.edge_type_embeddings = nn.ModuleDict({
            "pixel_to_pixel": nn.Embedding(1, hidden_channels // 4),
            "mppc_to_mppc": nn.Embedding(1, hidden_channels // 4),
            "pixel_to_mppc": nn.Embedding(1, hidden_channels // 4),
            "mppc_to_pixel": nn.Embedding(1, hidden_channels // 4),
        })
        
        # Heterogeneous graph convolution layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Layer normalization for each node type
            ln_dict = nn.ModuleDict({
                "pixel": LayerNorm(hidden_channels),
                "mppc": LayerNorm(hidden_channels)
            })
            self.layer_norms.append(ln_dict)
            
            # Choose convolution type based on attention setting
            if use_attention:
                conv_dict = {
                    edge_type: TransformerConv(
                        (-1, -1), 
                        hidden_channels // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True
                    ) for edge_type in self.edge_types
                }
            else:
                conv_dict = {
                    edge_type: GATv2Conv(
                        (-1, -1), 
                        hidden_channels // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True
                    ) for edge_type in self.edge_types
                }
            
            self.convs.append(HeteroConv(conv_dict, aggr="max"))
        
        # Edge-specific classifiers for different connection types
        self.edge_classifiers = nn.ModuleDict()
        
        # Pixel-Pixel classifier (same domain)
        self.edge_classifiers["pixel_to_pixel"] = self._build_edge_classifier(
            hidden_channels * 2 + hidden_channels // 4, "same_domain"
        )
        
        # MPPC-MPPC classifier (same domain) 
        self.edge_classifiers["mppc_to_mppc"] = self._build_edge_classifier(
            hidden_channels * 2 + hidden_channels // 4, "same_domain"
        )
        
        # Cross-domain classifiers (pixel-mppc, mppc-pixel)
        self.edge_classifiers["pixel_to_mppc"] = self._build_edge_classifier(
            hidden_channels * 2 + hidden_channels // 4, "cross_domain"
        )
        
        self.edge_classifiers["mppc_to_pixel"] = self._build_edge_classifier(
            hidden_channels * 2 + hidden_channels // 4, "cross_domain"
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _build_edge_classifier(self, input_dim, classifier_type):
        """Build edge classifier based on connection type."""
        if classifier_type == "same_domain":
            # Simpler classifier for same-type connections
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels // 2, 1)
            )
        else:  # cross_domain
            # More complex classifier for cross-domain connections
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels // 2, 1)
            )
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def forward(self, x_dict, edge_index_dict):
        # Initial node feature encoding
        processed_x = {}
        
        # Process pixel nodes
        if "pixel" in x_dict and x_dict["pixel"].numel() > 0:
            processed_x["pixel"] = self.pixel_encoder(x_dict["pixel"])
            processed_x["pixel"] = self.pixel_norm(processed_x["pixel"])
        
        # Process MPPC nodes  
        if "mppc" in x_dict and x_dict["mppc"].numel() > 0:
            processed_x["mppc"] = self.mppc_encoder(x_dict["mppc"])
            processed_x["mppc"] = self.mppc_norm(processed_x["mppc"])
        
        # Multi-layer message passing
        x_dict = processed_x
        for i in range(self.num_layers):
            # Apply graph convolution
            new_x_dict = self.convs[i](x_dict, edge_index_dict)
            
            # Apply layer normalization and dropout
            for node_type in new_x_dict.keys():
                if new_x_dict[node_type].numel() > 0:
                    # Layer normalization
                    new_x_dict[node_type] = self.layer_norms[i][node_type](new_x_dict[node_type])
                    
                    # Residual connection if not first layer
                    if i > 0 and node_type in x_dict:
                        new_x_dict[node_type] = new_x_dict[node_type] + x_dict[node_type]
                    
                    # Dropout
                    new_x_dict[node_type] = F.dropout(
                        new_x_dict[node_type], p=self.dropout, training=self.training
                    )
            
            x_dict = new_x_dict
        
        # Edge classification for each edge type
        out_dict = {}
        
        for edge_type in edge_index_dict.keys():
            if edge_index_dict[edge_type].numel() == 0:
                continue
                
            src_type, _, dst_type = edge_type
            edge_type_str = f"{src_type}_to_{dst_type}"
            
            # Check if we have valid node representations
            if (src_type not in x_dict or dst_type not in x_dict or 
                x_dict[src_type].numel() == 0 or x_dict[dst_type].numel() == 0):
                continue
            
            # Get node indices and features
            src_nodes = edge_index_dict[edge_type][0]
            dst_nodes = edge_index_dict[edge_type][1]
            src_features = x_dict[src_type][src_nodes]
            dst_features = x_dict[dst_type][dst_nodes]
            
            # Get edge type embedding
            device = src_nodes.device
            batch_size = src_nodes.size(0)
            edge_type_emb = self.edge_type_embeddings[edge_type_str](
                torch.zeros(batch_size, dtype=torch.long, device=device)
            )
            
            # Create comprehensive edge features
            edge_features = self._create_edge_features(
                src_features, dst_features, edge_type_emb, edge_type_str
            )
            
            # Apply edge-specific classifier
            logits = self.edge_classifiers[edge_type_str](edge_features).squeeze(-1)
            
            # Apply temperature scaling for better calibration
            logits = logits / self.temperature
            
            out_dict[edge_type] = torch.sigmoid(logits)
        
        return out_dict
    
    def _create_edge_features(self, src_features, dst_features, edge_type_emb, edge_type_str):
        """Create sophisticated edge features based on connection type."""
        
        # Basic concatenation
        concat_features = torch.cat([src_features, dst_features], dim=-1)
        
        # Add interaction features based on edge type
        if "cross_domain" in edge_type_str or src_features.size(-1) != dst_features.size(-1):
            # For cross-domain connections, add more sophisticated interactions
            # Element-wise interactions
            hadamard = src_features * dst_features  # Element-wise product
            difference = torch.abs(src_features - dst_features)  # Absolute difference
            
            # Distance-based features
            l2_distance = torch.norm(src_features - dst_features, dim=-1, keepdim=True)
            cosine_sim = F.cosine_similarity(src_features, dst_features, dim=-1, keepdim=True)
            
            # Combine all features
            interaction_features = torch.cat([
                hadamard, difference, l2_distance, cosine_sim
            ], dim=-1)
            
            # Project to consistent dimension
            interaction_proj = nn.Linear(
                interaction_features.size(-1), src_features.size(-1)
            ).to(src_features.device)
            interaction_features = interaction_proj(interaction_features)
            
            # Final edge representation
            edge_features = torch.cat([
                concat_features, interaction_features, edge_type_emb
            ], dim=-1)
        else:
            # For same-domain connections, simpler features suffice
            edge_features = torch.cat([concat_features, edge_type_emb], dim=-1)
        
        return edge_features

# Utility function to create the model with your specific configuration
def create_pixel_mppc_classifier(hidden_channels=64, num_layers=3, dropout=0.15, 
                                num_heads=4, use_attention=True):
    """
    Factory function to create a PixelMPPCEdgeClassifier with the correct configuration.
    
    Args:
        hidden_channels: Size of hidden representations (default: 64)
        num_layers: Number of graph convolution layers (default: 3)  
        dropout: Dropout probability (default: 0.15)
        num_heads: Number of attention heads (default: 4)
        use_attention: Whether to use TransformerConv vs GATv2Conv (default: True)
    
    Returns:
        Configured PixelMPPCEdgeClassifier model
    """
    return PixelMPPCEdgeClassifier(
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        use_attention=use_attention
    )

from src.torch.training import FocalLoss
from sklearn.metrics import roc_auc_score, average_precision_score

def train(model, train_loader, val_loader, epochs = 20, optimizer = None, criterion = None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if criterion is None:
        criterion = FocalLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            
            # Assuming binary classification for edges
            loss = 0
            for edge_type, edge_labels in data.edge_labels_dict.items():
                if edge_type in out:
                    loss += criterion(out[edge_type], edge_labels.float())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0
        predictions = []
        labels = []
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                data = data.to(device)
                out = model(data.x_dict, data.edge_index_dict)
                
                loss = 0
                for edge_type, edge_labels in data.edge_labels_dict.items():
                    if edge_type in out:
                        loss += criterion(out[edge_type], edge_labels.float())
                        predictions.append(out[edge_type].cpu())
                        labels.append(edge_labels.cpu())
                
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        auc_roc = roc_auc_score(torch.cat(labels).numpy(), torch.cat(predictions).numpy())
        auc_pr = average_precision_score(torch.cat(labels).numpy(), torch.cat(predictions).numpy())
        print(f"AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)

    return model, history


node_dims={"pixel": 3, "mppc": 4},
edge_types=[
    ("pixel", "to", "pixel"),
    ("mppc", "to", "mppc"),
    ("pixel", "to", "mppc"),
    ("mppc", "to", "pixel"),
],
model = create_pixel_mppc_classifier(hidden_channels=48, num_layers=5, dropout=0.2, num_heads=4, use_attention=True)

trained_model, history = train(model, train_loader, test_loader, epochs=40)

fig, ax = plt.subplots()
ax.plot(history['train_loss'], label='Train Loss')
ax.plot(history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss over Epochs')
ax.legend()
plt.savefig(f"{PLOTS_DIR}/{MODEL_NAME}_training_validation_loss.png")
plt.show()

torch.save(trained_model.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_state_dict.pth")

from sklearn.metrics import auc, roc_curve

fig, ax = plt.subplots()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model.to(device)
trained_model.eval()
predictions = []
labels = []
with torch.no_grad():
    for data in tqdm(test_loader, desc="Testing"):
        data = data.to(device)
        out = trained_model(data.x_dict, data.edge_index_dict)
        
        for edge_type, edge_labels in data.edge_labels_dict.items():
            if edge_type in out:
                predictions.append(out[edge_type].cpu())
                labels.append(edge_labels.cpu())

predictions = torch.cat(predictions).numpy()
labels = torch.cat(labels).numpy()

fpr, tpr, _ = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend()
plt.savefig(f"{PLOTS_DIR}/{MODEL_NAME}_roc_curve.png")
plt.show()
