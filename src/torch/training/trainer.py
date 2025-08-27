from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

def train_set_graph_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    optimizer = None,
    criterion = None,
    model : nn.Module = None,
    MODEL_DIR: str = "./models",

):
    
    # Create datasets and loaders
    class FocalLoss(nn.Module):
        """
        Focal Loss for addressing class imbalance.
        """

        def __init__(self, alpha=0.5, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            p_t = torch.where(targets == 1, inputs, 1 - inputs)
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

            focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
            return focal_loss.mean()

    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    # Training loop
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []

    best_val_auc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            predictions = model(batch)
            loss = criterion(predictions, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(predictions.detach().cpu().numpy())
            train_labels.extend(batch.y.cpu().numpy())
        train_auc = roc_auc_score(train_labels, train_preds)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Training loss: {train_loss/len(train_loader):.4f}, AUC: {train_auc:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                predictions = model(batch)
                loss = criterion(predictions, batch.y)

                val_loss += loss.item()
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        # Calculate metrics
        val_auc = roc_auc_score(val_labels, val_preds)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Validation loss: {val_loss/len(val_loader):.4f}, AUC: {val_auc:.4f}"
        )

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_graph_classifier.pth")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    print(f"Best validation AUC: {best_val_auc:.4f}")
    return model, {"train_aucs": train_aucs, "val_aucs": val_aucs}