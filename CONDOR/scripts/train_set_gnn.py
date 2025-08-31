import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn, radius_graph
import numpy as np
from typing import Tuple, Optional, List
import math

class EventDataset(torch.utils.data.Dataset):
    """Dataset class for loading preprocessed detector hit data"""
    
    def __init__(self, bg_pixel_data_path: str, bg_mppc_data_path: str, sig_pixel_data_path: str, sig_mppc_data_path: str,  padding_value: float = -1):
        """
        Args:
            pixel_data_path: Path to pixel spacetime data (.npy)
            mppc_data_path: Path to MPPC spacetime data (.npy) 
            labels_path: Path to event labels (.npy)
            padding_value: Value used for padding
        """
        # Load data
        bg_pixel = np.load(bg_pixel_data_path)
        bg_mppc = np.load(bg_mppc_data_path)
        sig_pixel = np.load(sig_pixel_data_path)
        sig_mppc = np.load(sig_mppc_data_path)
        
        self.labels = np.array([0]*len(bg_pixel) + [1]*len(sig_pixel))
        self.pixel_data = np.concatenate([bg_pixel, sig_pixel], axis=0)
        self.mppc_data = np.concatenate([bg_mppc, sig_mppc], axis=0)        
        
        del bg_pixel, bg_mppc, sig_pixel, sig_mppc


        self.padding_value = padding_value
        
        assert self.pixel_data.shape[0] == self.mppc_data.shape[0] == len(self.labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Convert event data to PyTorch Geometric format"""
        pixel_hits = self.pixel_data[idx]
        mppc_hits = self.mppc_data[idx]
        label = self.labels[idx]
        
        # Remove padding (assuming last dimension has multiple features)
        pixel_mask = ~np.all(pixel_hits == self.padding_value, axis=-1)
        mppc_mask = ~np.all(mppc_hits == self.padding_value, axis=-1)
        
        pixel_hits = pixel_hits[pixel_mask]
        mppc_hits = mppc_hits[mppc_mask]
        
        # Extract features (assuming format: [x, y, z, layer_index, track_id, timestamp])
        pixel_pos = pixel_hits[:, :3]  # x, y, z
        pixel_layers = pixel_hits[:, 3:4]  # layer index (1, 2, 3, 4)
        pixel_track_ids = pixel_hits[:, 4]  # Ground truth (not used as input)
        pixel_time = pixel_hits[:, 5:6]  # timestamp
        
        mppc_pos = mppc_hits[:, :3]  # x, y, z
        mppc_layers = mppc_hits[:, 3:4]  # layer index (2.5)
        mppc_track_ids = mppc_hits[:, 4]  # Ground truth (not used as input)
        mppc_time = mppc_hits[:, 5:6]  # timestamp
        
        # Combine all hits
        all_pos = np.vstack([pixel_pos, mppc_pos])
        all_layers = np.vstack([pixel_layers, mppc_layers])
        all_time = np.vstack([pixel_time, mppc_time])
        all_track_ids = np.hstack([pixel_track_ids, mppc_track_ids])
        
        # Create detector type indicator (0 = pixel, 1 = mppc)
        detector_type = np.hstack([
            np.zeros(len(pixel_hits)),  # 0 for pixel
            np.ones(len(mppc_hits))     # 1 for mppc
        ])
        
        # Combine features (excluding track_id from input)
        node_features = np.column_stack([
            all_pos,  # x, y, z
            all_time,  # timestamp
            detector_type[:, None],  # detector type
            all_layers  # actual layer index from data
        ])
        
        return {
            'pos': torch.tensor(all_pos, dtype=torch.float32),
            'x': torch.tensor(node_features, dtype=torch.float32),
            'track_ids': torch.tensor(all_track_ids, dtype=torch.long),  # Ground truth
            'y': torch.tensor(label, dtype=torch.long),
            'n_pixel_hits': len(pixel_hits),
            'n_mppc_hits': len(mppc_hits)
        }

class SpatialTemporalEdgeConv(MessagePassing):
    """Custom edge convolution that handles spatial and temporal relationships"""
    
    def __init__(self, in_channels: int, out_channels: int, temporal_weight: float = 0.1):
        super().__init__(aggr='mean')
        self.temporal_weight = temporal_weight
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + 2, out_channels),  # +2 for distance features
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x, edge_index, pos):
        """
        Args:
            x: Node features [n_nodes, in_channels]
            edge_index: Edge connectivity [2, n_edges]
            pos: Node positions [n_nodes, 4] (x, y, z, t)
        """
        return self.propagate(edge_index, x=x, pos=pos)
    
    def message(self, x_i, x_j, pos_i, pos_j):
        """Compute messages between connected nodes"""
        # Spatial distance
        spatial_dist = torch.norm(pos_i[:, :3] - pos_j[:, :3], dim=1, keepdim=True)
        
        # Temporal distance (weighted)
        temporal_dist = torch.abs(pos_i[:, 3:4] - pos_j[:, 3:4]) * self.temporal_weight
        
        # Combine features
        edge_features = torch.cat([x_i, x_j, spatial_dist, temporal_dist], dim=1)
        return self.mlp(edge_features)

class LayerAttention(nn.Module):
    """Attention mechanism to weight contributions from different detector layers"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim + 1, feature_dim // 2),  # +1 for layer index
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=0)
        )
        
    def forward(self, x, layer_indices):
        """
        Args:
            x: Node features [n_nodes, feature_dim]
            layer_indices: Layer index for each node [n_nodes, 1]
        """
        # Compute attention weights for each layer
        layer_features = torch.cat([x, layer_indices], dim=1)
        weights = self.attention(layer_features)
        return x * weights

class TrackGNN(nn.Module):
    """Graph Neural Network for learning track-level representations"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            SpatialTemporalEdgeConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.layer_attention = LayerAttention(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, pos, layer_indices):
        """
        Args:
            x: Node features [n_nodes, input_dim]
            edge_index: Edge connectivity [2, n_edges]
            pos: Node positions [n_nodes, 4] (x, y, z, t)
            layer_indices: Layer index for each node [n_nodes, 1]
        """
        h = self.input_proj(x)
        
        for layer in self.gnn_layers:
            h_new = layer(h, edge_index, pos)
            h = h + h_new  # Residual connection
            h = F.relu(h)
        
        # Apply layer attention
        h = self.layer_attention(h, layer_indices)
        
        return self.output_proj(h)

class EventClassificationGNN(nn.Module):
    """Main model for event classification using detector hits"""
    
    def __init__(
        self,
        input_dim: int = 6,  # x, y, z, t, detector_type, layer
        hidden_dim: int = 128,
        track_dim: int = 64,
        num_classes: int = 2,
        edge_radius: float = 10.0,  # mm
        temporal_radius: float = 5.0,  # ns
        k_neighbors: int = 8
    ):
        super().__init__()
        
        self.edge_radius = edge_radius
        self.temporal_radius = temporal_radius
        self.k_neighbors = k_neighbors
        
        # Track-level GNN
        self.track_gnn = TrackGNN(input_dim, hidden_dim, track_dim)
        
        # Inter-track correlation network
        self.correlation_net = nn.Sequential(
            nn.Linear(track_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Global pooling and classification
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Auxiliary tasks for physics-informed training
        self.track_regression = nn.Linear(track_dim, 3)  # Predict momentum direction
        self.vertex_regression = nn.Linear(track_dim, 3)  # Predict vertex position
        
    def build_graph(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Build graph edges based on spatial and temporal proximity"""
        
        # Split position into spatial and temporal components
        spatial_pos = pos[:, :3]  # x, y, z
        temporal_pos = pos[:, 3:4]  # t
        
        # Build spatial edges (within radius)
        spatial_edges = radius_graph(spatial_pos, r=self.edge_radius, batch=batch, max_num_neighbors=32)
        
        # Build k-nearest neighbor edges for robustness
        knn_edges = knn(spatial_pos, spatial_pos, k=self.k_neighbors, batch_x=batch, batch_y=batch)
        
        # Combine edge sets and remove duplicates
        edge_index = torch.cat([spatial_edges, knn_edges], dim=1)
        edge_index = torch.unique(edge_index, dim=1)
        
        # Filter edges by temporal compatibility
        if temporal_pos.numel() > 0 and edge_index.size(1) > 0:
            row, col = edge_index
            temporal_diff = torch.abs(temporal_pos[row] - temporal_pos[col]).squeeze()
            temporal_mask = temporal_diff < self.temporal_radius
            edge_index = edge_index[:, temporal_mask]
        
        return edge_index
    
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [n_nodes, input_dim]
                - pos: Node positions [n_nodes, 4] (x, y, z, t)
                - batch: Batch assignment [n_nodes]
        """
        x, pos, batch = data.x, data.pos, data.batch
        
        # Extract layer information
        layer_indices = x[:, -1:].long()  # Last feature is layer index
        
        # Build graph edges
        edge_index = self.build_graph(pos, batch)
        
        # Get track-level representations
        track_features = self.track_gnn(x, edge_index, pos, layer_indices)
        
        # Apply correlation network
        corr_features = self.correlation_net(track_features)
        
        # Global pooling for event-level representation
        event_features = global_mean_pool(corr_features, batch)
        
        # Classification
        logits = self.global_pool(event_features)
        
        # Auxiliary predictions (for training)
        aux_outputs = {
            'track_momentum': self.track_regression(track_features),
            'track_vertex': self.vertex_regression(track_features)
        }
        
        return logits, aux_outputs

def create_geometric_data(event_dict, device='cpu'):
    """Convert event dictionary to PyTorch Geometric Data object"""
    
    data = Data(
        x=event_dict['x'].to(device),
        pos=torch.cat([event_dict['pos'], event_dict['x'][:, 3:4]], dim=1).to(device),  # x,y,z,t
        track_ids=event_dict['track_ids'].to(device),
        y=event_dict['y'].to(device)
    )
    
    return data

def collate_events(batch_list):
    """Custom collate function for DataLoader"""
    geometric_data_list = [create_geometric_data(event) for event in batch_list]
    return Batch.from_data_list(geometric_data_list)

class PhysicsInformedLoss(nn.Module):
    """Loss function combining classification with physics-informed auxiliary tasks"""
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 track_weight: float = 0.1,
                 vertex_weight: float = 0.1):
        super().__init__()
        self.classification_weight = classification_weight
        self.track_weight = track_weight
        self.vertex_weight = vertex_weight
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
    def forward(self, predictions, targets, data):
        """
        Args:
            predictions: (logits, aux_outputs) from model
            targets: Ground truth labels
            data: Batch data with track_ids for auxiliary supervision
        """
        logits, aux_outputs = predictions
        
        # Main classification loss
        class_loss = self.classification_loss(logits, targets)
        
        # Auxiliary losses (using ground truth track_ids for supervision)
        track_loss = torch.tensor(0.0, device=logits.device)
        vertex_loss = torch.tensor(0.0, device=logits.device)
        
        if 'track_momentum' in aux_outputs and hasattr(data, 'track_ids'):
            # For track momentum, we could use track direction consistency
            # This is a simplified example - you might want more sophisticated supervision
            track_pred = aux_outputs['track_momentum']
            # Normalize predicted momentum vectors
            track_pred_norm = F.normalize(track_pred, dim=1)
            
            # Encourage similar predictions for hits from same track
            unique_tracks = torch.unique(data.track_ids)
            for track_id in unique_tracks:
                if track_id == -1:  # Skip padding
                    continue
                track_mask = data.track_ids == track_id
                if track_mask.sum() > 1:
                    track_preds = track_pred_norm[track_mask]
                    # Consistency loss within track
                    pairwise_diff = track_preds.unsqueeze(0) - track_preds.unsqueeze(1)
                    track_loss += torch.mean(torch.norm(pairwise_diff, dim=2))
        
        total_loss = (self.classification_weight * class_loss + 
                     self.track_weight * track_loss + 
                     self.vertex_weight * vertex_loss)
        
        return total_loss, {
            'classification': class_loss,
            'track': track_loss,
            'vertex': vertex_loss,
            'total': total_loss
        }

class EventClassificationTrainer:
    """Training wrapper for the event classification model"""
    
    def __init__(self, 
                 model: EventClassificationGNN,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = PhysicsInformedLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch)
            loss, loss_dict = self.criterion(predictions, batch.y, batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred_classes = torch.argmax(predictions[0], dim=1)
            correct += (pred_classes == batch.y).sum().item()
            total += len(batch.y)
            
        return total_loss / len(dataloader), correct / total
    
    def evaluate(self, dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                predictions = self.model(batch)
                loss, _ = self.criterion(predictions, batch.y, batch)
                
                total_loss += loss.item()
                pred_classes = torch.argmax(predictions[0], dim=1)
                correct += (pred_classes == batch.y).sum().item()
                total += len(batch.y)
        
        return total_loss / len(dataloader), correct / total

def train_model(
                bg_pixel_data_path: str,
                bg_mppc_data_path: str,
                sig_pixel_data_path: str,
                sig_mppc_data_path: str,
                num_epochs: int = 100,
                batch_size: int = 32,
                validation_split: float = 0.2):
    """
    Train the event classification model
    
    Args:
        pixel_data_path: Path to pixel spacetime data
        mppc_data_path: Path to MPPC spacetime data  
        labels_path: Path to event labels
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
    """
    
    # Load dataset
    dataset = EventDataset(bg_pixel_data_path=bg_pixel_data_path, 
                           bg_mppc_data_path=bg_mppc_data_path,
                           sig_pixel_data_path=sig_pixel_data_path,
                           sig_mppc_data_path=sig_mppc_data_path)
        
    # Train/validation split
    n_val = int(len(dataset) * validation_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_events
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_events
    )
    
    # Initialize model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EventClassificationGNN(
        input_dim=6,  # x, y, z, t, detector_type, layer_index
        hidden_dim=128,
        track_dim=64,
        num_classes=2,  # Signal vs background
        edge_radius=10.0,  # Adjust based on your detector geometry
        temporal_radius=5.0,  # Adjust based on your timing resolution
    )
    
    trainer = EventClassificationTrainer(model, device)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        trainer.scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return model, trainer, val_loader

# Additional utility functions for data analysis
def analyze_event_structure(pixel_data, mppc_data, padding_value=-1):
    """Analyze the structure of events in your dataset"""
    
    pixel_hits_per_event = []
    mppc_hits_per_event = []
    
    for i in range(len(pixel_data)):
        pixel_mask = ~np.all(pixel_data[i] == padding_value, axis=-1)
        mppc_mask = ~np.all(mppc_data[i] == padding_value, axis=-1)
        
        pixel_hits_per_event.append(pixel_mask.sum())
        mppc_hits_per_event.append(mppc_mask.sum())
    
    return {
        'pixel_hits_per_event': pixel_hits_per_event,
        'mppc_hits_per_event': mppc_hits_per_event,
        'avg_pixel_hits': np.mean(pixel_hits_per_event),
        'avg_mppc_hits': np.mean(mppc_hits_per_event),
        'total_events': len(pixel_data)
    }

def visualize_event(event_dict, event_idx: int = 0):
    """Simple visualization of an event (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        pos = event_dict['pos'].numpy()
        track_ids = event_dict['track_ids'].numpy()
        detector_type = event_dict['x'][:, 4].numpy()  # detector type feature
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot pixel hits
        pixel_mask = detector_type == 0
        if pixel_mask.any():
            ax.scatter(pos[pixel_mask, 0], pos[pixel_mask, 1], pos[pixel_mask, 2], 
                      c=track_ids[pixel_mask], marker='o', s=20, alpha=0.7, label='Pixel hits')
        
        # Plot MPPC hits  
        mppc_mask = detector_type == 1
        if mppc_mask.any():
            ax.scatter(pos[mppc_mask, 0], pos[mppc_mask, 1], pos[mppc_mask, 2],
                      c=track_ids[mppc_mask], marker='^', s=30, alpha=0.7, label='MPPC hits')
        
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.legend()
        ax.set_title(f'Event {event_idx} - Label: {event_dict["y"].item()}')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


DATA_DIR = "../mu3e_trigger_data"
MODEL_DIR = "../models"
PLOTS_DIR = "../plots"
SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_with_layer_pixel_spacetime.npy"
SIGNAL_MPPC_FILE = f"{DATA_DIR}/sig_with_layer_mppc_spacetime.npy"

BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_with_layer_pixel_spacetime.npy"
BACKGROUND_MPPC_FILE = f"{DATA_DIR}/bg_with_layer_mppc_spacetime.npy"


model, trainer, val_loader = train_model(
    bg_pixel_data_path=BACKGROUND_PIXEL_FILE,
    bg_mppc_data_path=BACKGROUND_MPPC_FILE,
    sig_pixel_data_path=SIGNAL_PIXEL_FILE,
    sig_mppc_data_path=SIGNAL_MPPC_FILE,
    num_epochs=50,
    batch_size=16  # Adjust based on your GPU memory
)

# Plot roc curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

model.eval()
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(trainer.device)
        logits, _ = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Save the trained model
torch.save(model.state_dict(), f"{MODEL_DIR}/event_classification_gnn.pth")