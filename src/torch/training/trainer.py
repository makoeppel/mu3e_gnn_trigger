from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

from .loss import FocalLoss

def train_set_graph_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model : nn.Module,
    num_epochs: int = 30,
    optimizer = None,
    scheduler = None,
    criterion = None,
    MODEL_DIR: str = "./models",

):
    # Create datasets and loaders
    
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    if criterion is None:
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

        train_preds = torch.tensor(train_preds)
        train_labels = torch.tensor(train_labels)
        if torch.std(train_preds)/(torch.mean(train_preds)+1e-6) < 0.01:
            print(f"Warning: Low variance in training predictions, possible mode collapse. Std: {torch.std(train_preds):.6f}, Mean: {torch.mean(train_preds):.6f}")
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


        val_preds = torch.tensor(val_preds)
        val_labels = torch.tensor(val_labels)
        if torch.std(val_preds)/(torch.mean(val_preds)+1e-6) < 0.01:
            print(f"Warning: Low variance in training predictions, possible mode collapse. Std: {torch.std(train_preds):.6f}, Mean: {torch.mean(train_preds):.6f}")
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

    print(f"Best validation AUC: {best_val_auc:.4f}")
    return model, {"train_aucs": train_aucs, "val_aucs": val_aucs}


def train_hetero_graph_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model : nn.Module,
    num_epochs: int = 30,
    optimizer = None,
    scheduler = None,
    criterion = None,
    MODEL_DIR: str = "./models",
    
):    # Create datasets and loaders
    
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters."
    )

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
    if criterion is None:
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
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_hetero_graph_classifier.pth")
            
    print(f"Best validation AUC: {best_val_auc:.4f}")
    return model, {"train_aucs": train_aucs, "val_aucs": val_aucs}