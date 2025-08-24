import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax

from torch_geometric.data import Data, Batch
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Add custom path for model components
sys.path.append("/afs/desy.de/user/a/aulich/mu3e_trigger")
from torch_src.model.components import get_mlp, TransformerBlock, PoolerTransformerBlock
from torch_src.model.components.gnn import (
    WeightedMessagePassing,
    EdgeWeightGenerator,
    EdgeWeightUpdater,
)


# ===== CONFIGURATION =====
class Config:
    """Configuration settings for the training pipeline."""

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    ROOT_DIR = "/afs/desy.de/user/a/aulich/mu3e_trigger"
    DATA_DIR = "/data/dust/group/atlas/ttreco/mu3e_trigger_data"
    PLOTS_DIR = f"{ROOT_DIR}/plots_gnn"
    MODEL_DIR = f"{ROOT_DIR}/models_gnn"
    MODEL_NAME = "classification_single_seq"

    # Data files
    SIGNAL_PIXEL_FILE = f"{DATA_DIR}/sig_with_layer_pixel_spacetime.npy"
    BACKGROUND_PIXEL_FILE = f"{DATA_DIR}/bg_with_layer_pixel_spacetime.npy"
    SIGNAL_ONLY_PIXEL_FILE = f"{DATA_DIR}/sig_with_layer_only_pixel_spacetime.npy"

    # Model hyperparameters
    NUM_FEATURES = 3
    GNN_HIDDEN_DIM = 10
    GNN_EMBED_DIM = 16
    CLASSIFIER_HIDDEN_DIM = 16
    NUM_CLASSES = 2

    # Training hyperparameters
    DATA_MAX_SIZE = None
    BATCH_SIZE = 512
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    RANDOM_STATE = 42

    def __init__(self):
        # Create necessary directories
        os.makedirs(f"{self.MODEL_DIR}/{self.MODEL_NAME}", exist_ok=True)
        os.makedirs(self.PLOTS_DIR, exist_ok=True)

        # Print device info
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")


# ===== DATA UTILITIES =====
def load_and_prepare_data(config: Config):
    """Load and prepare the dataset."""
    print("Loading dataset...")

    bg_pixel_spacetime = np.load(config.BACKGROUND_PIXEL_FILE)
    sig_pixel_spacetime = np.load(config.SIGNAL_PIXEL_FILE)

    # Combine data
    X = np.concatenate([bg_pixel_spacetime, sig_pixel_spacetime], axis=0)
    y = np.concatenate(
        [
            np.zeros(len(bg_pixel_spacetime), dtype=int),
            np.ones(len(sig_pixel_spacetime), dtype=int),
        ],
        axis=0,
    )

    number_of_hits = (X != -1).all(axis=-1).sum(axis=-1)

    X = X[number_of_hits >= 9]
    y = y[number_of_hits >= 9]

    # Shuffle data
    np.random.seed(config.RANDOM_STATE)
    if config.DATA_MAX_SIZE is not None:
        shuffled_indices = np.random.permutation(len(X))[: config.DATA_MAX_SIZE]
    else:
        shuffled_indices = np.random.permutation(len(X))
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    print(f"Dataset loaded: {len(X)} samples")
    print(f"Background samples: {len(bg_pixel_spacetime)}")
    print(f"Signal samples: {len(sig_pixel_spacetime)}")

    return X, y


def batch_events_to_variable_graphs(events: torch.Tensor) -> Batch:
    """
    Convert a batch of events (padded) to a single PyG Batch object with variable number of graphs per event.
    Time is assumed to be at the last column (-1).

    Args:
        events (torch.Tensor): [num_events, num_hits, feature_dim] padded with -1
                               last column must be time

    Returns:
        Batch: PyG Batch object containing all graphs from all events
    """
    all_graphs = []
    event_indices = []

    for event_idx, event in enumerate(events):
        valid_hits = event[event[:, -1] != -1]

        times = valid_hits[:, -1]
        positions = valid_hits[:, :-2]
        layers = valid_hits[:, -2]  # Assuming layer is the second last feature

        unique_times = np.unique(times)
        for t in unique_times:
            mask = times == t
            masked_positions = positions[mask]
            masked_layers = layers[mask]
            num_nodes = masked_positions.size(0)

            edges = []
            for u, v in combinations(range(num_nodes), 2):
                if masked_layers[u] == masked_layers[v]:
                    continue
                if torch.abs(masked_layers[u] - masked_layers[v]) > 1:
                    continue
                edges.append([u, v])
                edges.append([v, u])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            nodes = masked_positions
            all_graphs.append(Data(x=nodes, edge_index=edge_index))
            event_indices.append(event_idx)

    # Batch all graphs together
    batch = Batch.from_data_list(all_graphs)
    batch.event_batch = torch.tensor(event_indices, dtype=torch.long)
    return batch


# ===== DATASET CLASSES =====
class EventDataset(Dataset):
    """Dataset for handling event data with class balancing."""

    def __init__(self, X, y, shuffle=True, num_classes=2):
        if shuffle:
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
        self.X = X
        self.y = y
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float),
            torch.nn.functional.one_hot(
                torch.tensor(self.y[idx], dtype=torch.long),
                num_classes=self.num_classes,
            ).float(),
        )

    def get_class_weights(self):
        """Calculate class weights for balancing."""
        labels = self.y
        class_sample_count = torch.tensor(
            [(labels == t).sum() for t in np.arange(self.num_classes)]
        )
        weight = 1.0 / class_sample_count.float()
        class_weights = weight / weight.sum()
        return class_weights


class GraphSetDataLoader(DataLoader):
    """Custom data loader for converting events to graph batches."""

    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[
                start_idx : min(start_idx + self.batch_size, len(self.dataset))
            ]
            batch_events = [self.dataset[i][0] for i in batch_indices]
            batch_labels = torch.stack([self.dataset[i][1] for i in batch_indices])

            padded_events = torch.nn.utils.rnn.pad_sequence(
                batch_events, batch_first=True, padding_value=-1
            )
            batch_graphs = batch_events_to_variable_graphs(padded_events)
            yield batch_graphs, batch_labels

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class GraphEmbedder(nn.Module):
    """Graph neural network for embedding individual graphs."""

    def __init__(self, in_dim, hidden_dim=64, emb_dim=128, add_self_loops_flag=True):
        super().__init__()
        self.edge_init = EdgeWeightGenerator(in_dim, hidden_dim)
        self.conv1 = WeightedMessagePassing(in_dim, hidden_dim)
        self.edge_updater = EdgeWeightUpdater(hidden_dim, hidden_dim)
        self.conv2 = WeightedMessagePassing(hidden_dim, hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, emb_dim)
        self.add_self_loops_flag = add_self_loops_flag

    def forward(self, x, edge_index, batch):
        num_nodes = x.size(0)

        if self.add_self_loops_flag:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Generate initial edge weights
        edge_weight = self.edge_init(x, edge_index)

        # First message passing
        x = F.relu(self.conv1(x, edge_index, edge_weight))

        # Update edge weights
        edge_weight = self.edge_updater(x, edge_index)

        # Second message passing with updated weights
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        # Pooling to fixed size embedding
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=-1)

        return self.fc(x)


# ===== SEQUENCE CLASSIFIER =====
class SequenceClassifier(nn.Module):
    """Transformer-based sequence classifier for graph embeddings."""

    def __init__(self, input_dim, hidden_dim=16, num_classes=2):
        super().__init__()
        self.embedding_dim = hidden_dim
        self.embedding_layer = get_mlp(input_dim, hidden_dim, num_layers=3)
        self.transformer = TransformerBlock(hidden_dim, num_heads=4)
        self.pooler = PoolerTransformerBlock(hidden_dim, num_heads=4)
        self.classifier = get_mlp(hidden_dim, num_classes, num_layers=3)

    def forward(self, graph_embeddings, event_batch):
        """
        Args:
            graph_embeddings: [num_graphs_total, input_dim]
            event_batch: [num_graphs_total] mapping graphs to events
        Returns:
            logits: [num_events, num_classes]
        """
        x = self.embedding_layer(graph_embeddings)
        x = self.transformer(x, event_batch)
        x = self.pooler(x, event_batch).view(-1, self.embedding_dim)
        logits = self.classifier(x)
        return nn.functional.softmax(logits, dim=-1)


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
        if not hasattr(batch, "event_batch"):
            raise ValueError("Batch must have 'event_batch' attribute")

        x, edge_index, graph_batch, event_batch = (
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.event_batch,
        )

        graph_embeddings = self.gnn(x, edge_index, graph_batch)
        classifier_outputs = self.classifier_module(graph_embeddings, event_batch)

        return classifier_outputs


# ===== TRAINING UTILITIES =====
class ModelTrainer:
    """Handles training, validation, and evaluation of graph set classifier models."""

    def __init__(self, model, device, model_dir="models", plots_dir="plots"):
        self.model = model
        self.device = device
        self.model_dir = model_dir
        self.plots_dir = plots_dir
        self.history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    def prepare_data(self, X, y, test_size=0.2, batch_size=512, random_state=42):
        """Split data and create data loaders."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        train_dataset = EventDataset(X_train, y_train, shuffle=True, num_classes=2)
        val_dataset = EventDataset(X_val, y_val, shuffle=False, num_classes=2)

        train_loader = GraphSetDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = GraphSetDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        print(f"Class weights: {train_dataset.get_class_weights()}")
        print(
            f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )

        return train_loader, val_loader, train_dataset, val_dataset

    def setup_training(self, train_dataset, lr=1e-4):
        """Initialize loss function, optimizer, and metrics."""
        class_weights = train_dataset.get_class_weights().to(self.device)
        loss_fn = nn.CrossEntropyLoss(class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        auc_metric = MulticlassAUROC(num_classes=2)
        acc_metric = MulticlassAccuracy(num_classes=2)

        return loss_fn, optimizer, auc_metric, acc_metric

    def train_epoch(self, train_loader, loss_fn, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_samples = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_graphs, batch_labels in pbar:
            batch_graphs = batch_graphs.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_graphs)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.size(0)
            num_samples += batch_labels.size(0)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_samples

    def validate_epoch(self, val_loader, loss_fn, auc_metric, acc_metric):
        """Validate for one epoch."""
        self.model.eval()
        total_val_loss = 0
        num_samples = 0

        auc_metric.reset()
        acc_metric.reset()

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            for batch_graphs, batch_labels in pbar:
                batch_graphs = batch_graphs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_graphs)
                loss = loss_fn(outputs, batch_labels)

                total_val_loss += loss.item() * batch_labels.size(0)
                num_samples += batch_labels.size(0)

                # Update metrics
                outputs_cpu = outputs.detach().cpu()
                labels_cpu = batch_labels.detach().cpu()
                labels_argmax = labels_cpu.argmax(dim=-1)

                auc_metric.update(outputs_cpu, labels_argmax)
                acc_metric.update(outputs_cpu, labels_argmax)

                # Update progress bar
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / num_samples
        val_auc = auc_metric.compute().item()
        val_acc = acc_metric.compute().item()

        return avg_val_loss, val_auc, val_acc

    def train(
        self,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        num_epochs=10,
        lr=1e-4,
        save_checkpoints=True,
    ):
        """Main training loop."""
        loss_fn, optimizer, auc_metric, acc_metric = self.setup_training(
            train_dataset, lr
        )

        print(f"Starting training for {num_epochs} epochs...")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        best_val_auc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            train_loss = self.train_epoch(train_loader, loss_fn, optimizer)
            self.history["train_loss"].append(train_loss)

            # Validation
            val_loss, val_auc, val_acc = self.validate_epoch(
                val_loader, loss_fn, auc_metric, acc_metric
            )

            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["val_acc"].append(val_acc)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_path = os.path.join(self.model_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with AUC: {best_val_auc:.4f}")

            # Save checkpoint
            if save_checkpoints:
                checkpoint_path = os.path.join(self.model_dir, f"epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

        # Save final model
        final_model_path = os.path.join(self.model_dir, "final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"\nTraining completed! Final model saved to {final_model_path}")

    def plot_training_history(self):
        """Plot training history with loss, AUC, and accuracy."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss plot
        axes[0].plot(
            self.history["train_loss"], label="Train Loss", color="blue", linewidth=2
        )
        axes[0].plot(
            self.history["val_loss"], label="Val Loss", color="red", linewidth=2
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # AUC plot
        axes[1].plot(
            self.history["val_auc"], label="Val AUC", color="orange", linewidth=2
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Validation AUC")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Accuracy plot
        axes[2].plot(
            self.history["val_acc"], label="Val Accuracy", color="green", linewidth=2
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Validation Accuracy")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        history_plot_path = os.path.join(self.plots_dir, "training_history.png")
        fig.savefig(history_plot_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to {history_plot_path}")
        plt.show()

    def plot_roc_curve(self, val_loader):
        """Generate and plot ROC curve."""
        self.model.eval()
        all_labels = []
        all_probs = []

        print("Generating ROC curve...")
        with torch.no_grad():
            for graph_set_batch, label_batch in tqdm(
                val_loader, desc="ROC computation"
            ):
                graph_set_batch = graph_set_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                outputs = self.model(graph_set_batch)
                probs = outputs[:, 1]  # Probability of positive class (already softmax)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(label_batch.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels).argmax(axis=-1)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            fpr,
            tpr,
            color="blue",
            linewidth=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="red",
            linestyle="--",
            linewidth=1,
            label="Random classifier",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        roc_plot_path = os.path.join(self.plots_dir, "roc_curve.png")
        fig.savefig(roc_plot_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve plot saved to {roc_plot_path}")
        plt.show()

        return roc_auc


# ===== MAIN PIPELINE =====
def create_model(config: Config):
    """Create the complete GNN classifier model."""
    gnn = GraphEmbedder(
        in_dim=config.NUM_FEATURES,
        hidden_dim=config.GNN_HIDDEN_DIM,
        emb_dim=config.GNN_EMBED_DIM,
    )

    classifier = SequenceClassifier(
        input_dim=config.GNN_EMBED_DIM,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
    )

    model = SequenceGNNClassifier(gnn=gnn, classifier_module=classifier)
    return model.to(config.DEVICE)


def main_training_pipeline(config: Config):
    """Complete training pipeline."""
    # Load data
    X, y = load_and_prepare_data(config)

    # Create model
    model = create_model(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {param_count:,} trainable parameters")

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        device=config.DEVICE,
        model_dir=f"{config.MODEL_DIR}/{config.MODEL_NAME}",
        plots_dir=config.PLOTS_DIR,
    )

    # Prepare data
    train_loader, val_loader, train_dataset, val_dataset = trainer.prepare_data(
        X, y, batch_size=config.BATCH_SIZE, random_state=config.RANDOM_STATE
    )

    # Train model
    trainer.train(
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
    )

    # Generate plots and evaluation
    trainer.plot_training_history()
    roc_auc = trainer.plot_roc_curve(val_loader)

    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Best Validation AUC: {max(trainer.history['val_auc']):.4f}")
    print(f"Best Validation Accuracy: {max(trainer.history['val_acc']):.4f}")
    print(f"Final ROC AUC: {roc_auc:.4f}")
    print(f"{'='*60}")

    return trainer, model


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Initialize configuration
    config = Config()

    # Run training pipeline
    trainer, model = main_training_pipeline(config)

    print("Training pipeline completed successfully!")
