from util import EarlyStopping
import torch
import torch.nn as nn
import tqdm


def train_model(model, train_loader, val_loader, epochs=200, lr=1e-4, patience=20, path="checkpoint.pth", device="cpu"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, path=path)
    ckpt_epoch = epochs

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            ckpt_epoch = epoch - patience
            break

    return train_losses, val_losses, ckpt_epoch