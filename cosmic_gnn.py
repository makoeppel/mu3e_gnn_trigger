import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_geometric.nn import EdgeConv


class CosmicInBeamEdgeClassifier(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64):
        super().__init__()
        
        # EdgeConv takes the combined features of source and target nodes
        # In features dimension = hidden_channels * 2 (or in_channels * 2)
        self.conv1 = EdgeConv(
            Sequential(Linear(in_channels * 2, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        )
        self.conv2 = EdgeConv(
            Sequential(Linear(hidden_channels * 2, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        )
        
        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        
        edge_index = native_knn_graph(pos, k=8, batch=batch)
        
        # EdgeConv expects the feature tensor and the connectivity index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        graph_embeddings = global_mean_pool(x, batch)
        return self.classifier(graph_embeddings)

@torch.no_grad()
def plot_efficiency_vs_acceptance(model, loader, device):
    model.eval()
    all_y_true = []
    all_y_scores = []
    
    # 1. Collect predictions and true labels
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).squeeze(-1)
        # Convert raw logits to probabilities [0, 1] using sigmoid
        probs = torch.sigmoid(out)
        
        all_y_true.extend(batch.y.cpu().numpy())
        all_y_scores.extend(probs.cpu().numpy())
        
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)
    
    # 2. Compute False Positive Rate (Acceptance) and True Positive Rate (Efficiency)
    # roc_curve automatically tests all relevant threshold cuts
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
    
    # 3. Create the plot
    plt.figure(figsize=(12, 5))
    
    # Plotting with diamond markers matching your screenshot
    plt.scatter(
        fpr, tpr, 
        color='#E24A33',    # Red/Orange diamond color
        marker='D',         # 'D' stands for diamond
        s=40, 
        label='Cosmics w/ GNN Classifier'
    )
    
    # 4. Format axes to match your style
    plt.xscale('log')  # Crucial: Log scale for background acceptance
    plt.xlim(5e-5, 1e-1) # Adjust bounds to match your screenshot's range
    plt.ylim(0.0, 1.1)
    
    plt.xlabel('Background Acceptance', fontsize=11)
    plt.ylabel('Reconstruction Efficiency', fontsize=11)
    plt.title('Reconstruction efficiency of cosmics as a function of the acceptance', fontsize=13, pad=10)
    
    plt.grid(True, which="both", linestyle='-', alpha=0.7)
    plt.legend(loc='lower left', frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.savefig("reconstruction_efficiency_plot.png", dpi=300)
    plt.show()


@torch.no_grad()
def plot_standard_roc_curve(model, loader, device):
    model.eval()
    all_y_true = []
    all_y_scores = []
    
    # 1. Gather model predictions and truth labels
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).squeeze(-1)
        probs = torch.sigmoid(out)  # Map logits to probabilities [0, 1]
        
        all_y_true.extend(batch.y.cpu().numpy())
        all_y_scores.extend(probs.cpu().numpy())
        
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)
    
    # 2. Compute ROC metrics and AUC score
    fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 3. Create the traditional ROC Plot
    plt.figure(figsize=(8, 7))
    
    # Plot the GNN performance line
    plt.plot(
        fpr, tpr, 
        color='darkorange', 
        lw=2, 
        label=f'GNN Classifier (AUC = {roc_auc:.4f})'
    )
    
    # Plot the random classifier baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.5000)')
    
    # 4. Axis limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Acceptance)', fontsize=12)
    plt.ylabel('True Positive Rate (Reconstruction Efficiency)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("standard_roc_curve.png", dpi=300)
    plt.show()

def plot_detector_hits_2d(beam_arr, cosmic_arr, max_events_to_display=50):
    """
    Plots a 2D projection (X/Y) of beam hits (red) and cosmic hits (blue).
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Extract and clean Beam hits (X is index 0, Y is index 1)
    beam_subset = beam_arr[:max_events_to_display].reshape(-1, 4)
    beam_mask = beam_subset[:, 0] != -999.0
    beam_hits = beam_subset[beam_mask]
    
    # 2. Extract and clean Cosmic hits
    cosmic_subset = cosmic_arr[:max_events_to_display].reshape(-1, 4)
    cosmic_mask = cosmic_subset[:, 0] != -999.0
    cosmic_hits = cosmic_subset[cosmic_mask]
    
    # 3. Plot Beam hits (Red)
    plt.scatter(
        beam_hits[:, 0], beam_hits[:, 1], 
        c='red', 
        s=4,           # Slightly larger size for 2D clarity
        alpha=0.4,     
        label=f'Beam Events ({max_events_to_display} frames)'
    )
    
    # 4. Plot Cosmic hits (Blue)
    plt.scatter(
        cosmic_hits[:, 0], cosmic_hits[:, 1], 
        c='blue', 
        s=4, 
        alpha=0.4, 
        label=f'Cosmic Events ({max_events_to_display} frames)'
    )
    
    # Labeling the axes
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title('2D X/Y Hit Distribution: Beam vs Cosmic', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Adjust legend sizing
    lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
    for handle in lgnd.legend_handles:
        handle.set_sizes([40.0])
        handle.set_alpha(1.0)
        
    plt.tight_layout()
    plt.show()

# ==========================================
# 1. DATA LOADING & MOCK GENERATION
# ==========================================
print("Setting up data...")
beam_path = "data/beam_pixel_spacetime.npy"
cosmic_path = "data/cosmic_pixel_spacetime.npy"

if os.path.exists(beam_path) and os.path.exists(cosmic_path):
    beam_data = np.load(beam_path)
    cosmic_data = np.load(cosmic_path)
else:
    print("-> Creating mock files for testing...")
    os.makedirs("data", exist_ok=True)
    mock_beam = np.random.uniform(-30, 30, (500, 256, 4))
    mock_cosmic = np.random.uniform(-50, 50, (500, 256, 4))
    for i in range(500):
        mock_beam[i, np.random.randint(50, 200):, :] = -999.0
        mock_cosmic[i, np.random.randint(20, 100):, :] = -999.0
    np.save(beam_path, mock_beam)
    np.save(cosmic_path, mock_cosmic)
    beam_data, cosmic_data = mock_beam, mock_cosmic

print(f"Loaded Beam: {beam_data.shape} | Cosmic: {cosmic_data.shape}")


# ==========================================
# 2. DYNAMIC MIXING & GRAPH CREATION
# ==========================================
def create_mixed_dataset(beam_arr, cosmic_arr, cosmic_mix_fraction=0.5, seed=42):
    """
    Creates a dataset where 50% (by default) of beam frames are injected 
    with background cosmic rays.
    """
    np.random.seed(seed)
    num_beam_events = beam_arr.shape[0]
    num_cosmic_events = cosmic_arr.shape[0]

    graph_list = []

    # Determine which beam frames will get mixed with a cosmic track
    mix_mask = np.random.rand(num_beam_events) < cosmic_mix_fraction

    for i in range(num_beam_events):
        # 1. Extract and clean the base beam event
        beam_event = beam_arr[i]
        b_mask = beam_event[:, 0] != -999.0
        valid_beam_hits = beam_event[b_mask]

        if mix_mask[i]:
            # 2. Grab a random cosmic event to overlay
            rand_cosmic_idx = np.random.randint(0, num_cosmic_events)
            cosmic_event = cosmic_arr[rand_cosmic_idx]
            c_mask = cosmic_event[:, 0] != -999.0
            valid_cosmic_hits = cosmic_event[c_mask]

            if len(valid_cosmic_hits) >= 4:
                # 3. Merge the hits together into one single frame
                combined_hits = np.random.permutation(
                    np.vstack([valid_beam_hits, valid_cosmic_hits])
                )
                label = 1.0  # Contains Cosmic Overlay
            else:
                combined_hits = valid_beam_hits
                label = 0.0  # Pure Beam
        else:
            combined_hits = valid_beam_hits
            label = 0.0  # Pure Beam

        if len(combined_hits) == 0:
            continue

        # Convert to PyG tensors
        x_tensor = torch.tensor(combined_hits, dtype=torch.float)
        pos_tensor = x_tensor[:, :3]
        y_tensor = torch.tensor([label], dtype=torch.float)

        graph_list.append(Data(x=x_tensor, pos=pos_tensor, y=y_tensor))

    return graph_list

print("Generating mixed event frames...")
all_graphs = create_mixed_dataset(beam_data, cosmic_data, cosmic_mix_fraction=1)


# ==========================================
# 3. SPLIT AND BATCH DATA
# ==========================================
train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

plot_detector_hits_2d(beam_data, cosmic_data)

# ==========================================
# 4. DEPENDENCY-FREE NATIVE K-NN FUNCTION
# ==========================================
def native_knn_graph(pos, k, batch):
    neg_two_xy = -2 * torch.matmul(pos, pos.t())
    x_sq = torch.sum(pos**2, dim=-1, keepdim=True)
    dist = x_sq + neg_two_xy + x_sq.t()

    same_batch_mask = (batch.unsqueeze(0) == batch.unsqueeze(1))
    dist = torch.where(same_batch_mask, dist, torch.tensor(float('inf'), device=pos.device))

    # Clip k to prevent error if an event has fewer hits than requested neighbors
    actual_k = min(k, pos.size(0) - 1)

    _, indices = torch.topk(dist, k=actual_k+1, dim=-1, largest=False)
    knn_indices = indices[:, 1:] 

    node_idx = torch.arange(pos.size(0), device=pos.device).unsqueeze(1).repeat(1, actual_k)
    return torch.stack([knn_indices.flatten(), node_idx.flatten()], dim=0)


# ==========================================
# 5. GNN GRAPH CLASSIFICATION ARCHITECTURE
# ==========================================
class CosmicInBeamClassifier(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        
        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        
        edge_index = native_knn_graph(pos, k=8, batch=batch)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        graph_embeddings = global_mean_pool(x, batch)
        return self.classifier(graph_embeddings)


# ==========================================
# 6. TRAINING AND EVALUATION ENGINE
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = CosmicInBeamClassifier(in_channels=4, hidden_channels=64).to(device)
model = CosmicInBeamEdgeClassifier(in_channels=4, hidden_channels=128).to(device)

# Print total network parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Initialized GNN Model with {total_params:,} parameters.")

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch).squeeze(-1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def evaluate_with_auc(loader):
    model.eval()
    all_y_true = []
    all_y_scores = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).squeeze(-1)
        probs = torch.sigmoid(out)  # Get probability scores for AUC
        
        all_y_true.extend(batch.y.cpu().numpy())
        all_y_scores.extend(probs.cpu().numpy())
        
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)
    
    # Calculate traditional accuracy (Threshold = 0.5)
    preds = (all_y_scores > 0.5).astype(float)
    accuracy = np.mean(preds == all_y_true)
    
    # Calculate AUC (handles edge case where a split might lack one class)
    try:
        auc_score = roc_auc_score(all_y_true, all_y_scores)
    except ValueError:
        auc_score = 0.5  # Fallback to random guess baseline if calculation fails
        
    return accuracy, auc_score

print(f"Starting training loop on {device}...")
for epoch in range(1, 101):
    loss = train_epoch()
    train_acc, train_auc = evaluate_with_auc(train_loader)
    test_acc, test_auc = evaluate_with_auc(test_loader)

    # Clean, tabular print out
    print(
        f"Epoch: {epoch:02d} | "
        f"Loss: {loss:.4f} | "
        f"Train Acc/AUC: {train_acc:.2%} / {train_auc:.4f} | "
        f"Test Acc/AUC: {test_acc:.2%} / {test_auc:.4f}"
    )

plot_efficiency_vs_acceptance(model, test_loader, device)
plot_standard_roc_curve(model, test_loader, device)