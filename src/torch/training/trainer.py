from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
from typing import Optional, List, Dict, Any

from .loss import FocalLoss


def get_device(device: str = "auto"):
    """
    Get the appropriate device for training.

    Args:
        device (str): Device specification. Can be:
            - "auto": Automatically detect best available device
            - "cpu": Force CPU usage
            - "cuda": Force CUDA GPU usage
            - "mps": Force Apple Silicon GPU usage
            - Specific device like "cuda:0", "cuda:1", etc.

    Returns:
        torch.device: The device to use for training
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)
    print(f"Using device: {device}")

    return device


class Trainer:
    """
    A simple training loop for PyTorch models.
    Args:
        model (nn.Module): The PyTorch model to train
        loss_fn (callable): Loss function
        optimizer (torch.optim.Optimizer): Optimizer for training
        metrics (dict): Dictionary of metric functions to evaluate during training and validation
        device (str or torch.device): Device to use for training ("auto", "cpu", "cuda", "mps", or specific device)
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        metrics: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else {}
        self.device = device if device is not None else get_device("auto")
        self.model.to(self.device)

    def train_epoch(self, dataloader):
        """
        Train the model for one epoch.
        Args:
            dataloader (DataLoader): DataLoader for training data
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluate the model on validation or test data.
        Args:
            dataloader (DataLoader): DataLoader for validation/test data
        Returns:
            dict: Dictionary with average loss and metric results
        """
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.loss_fn(out, batch.y)
                total_loss += loss.item() * batch.num_graphs
                all_outputs.append(out.cpu())
                all_labels.append(batch.y.cpu())
        avg_loss = total_loss / len(dataloader.dataset)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        results = {"loss": avg_loss}
        for name, metric in self.metrics.items():
            results[name] = metric(all_outputs, all_labels)

        return results

    def fit(self, train_loader, val_loader, epochs, checkpoint_path=None):
        """
        Train the model for a specified number of epochs, with optional validation and checkpointing.
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            epochs (int): Number of epochs to train
            checkpoint_path (str, optional): Path to save the best model checkpoint based on validation loss
        Returns:
            dict: Training history with losses and metrics
        """
        best_val_loss = float("inf")
        history = {
            "train_loss": [],
            "val_loss": [],
            **{name: [] for name in self.metrics.keys()},
        }
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_results = self.evaluate(val_loader)
            val_loss = val_results["loss"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            for name in self.metrics.keys():
                history[name].append(val_results[name])
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for name, value in val_results.items():
                if name != "loss":
                    print(f"Val {name}: {value:.4f}")
            if checkpoint_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        print("Training complete.")
        return history
    
    def predict(self, dataloader):
        """
        Generate predictions for the given dataloader.
        Args:
            dataloader (DataLoader): DataLoader for data to predict on
        Returns:
            torch.Tensor: Concatenated predictions for all data in the dataloader
        """
        self.model.eval()
        all_outputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", leave=False):
                batch = batch.to(self.device)
                out = self.model(batch)
                all_outputs.append(out.cpu())
        all_outputs = torch.cat(all_outputs, dim=0)
        return all_outputs
