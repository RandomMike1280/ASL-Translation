"""
train_mlp.py

Train an MLP on labeled hand keypoints distances.
Loads all NPZ dataset files, builds a configurable MLP, and trains it.
"""
import argparse
import os
import numpy as np
import string
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

def load_dataset(dataset_dir):
    distances_list = []
    labels_list = []
    for fname in os.listdir(dataset_dir):
        if fname.endswith('.npz'):
            path = os.path.join(dataset_dir, fname)
            data = np.load(path, allow_pickle=True)
            distances_list.append(data['distances'])
            labels_list.append(data['labels'])
    X = np.concatenate(distances_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

class HandKeypointsDataset(Dataset):
    def __init__(self, X, y, noise_std=0.01):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.noise_std = noise_std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        noise = torch.randn_like(x) * self.noise_std
        return x + noise, self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, layers, output_dim):
        super().__init__()
        modules = []
        prev = input_dim
        for size in layers:
            modules.append(nn.Linear(prev, size))
            modules.append(nn.ReLU())
            prev = size
        modules.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*modules)
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP on hand keypoints dataset')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to NPZ dataset directory')
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 64, 32], help='Hidden layer sizes')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Load and split dataset
    X, y = load_dataset(args.dataset_dir)
    classes = list(string.ascii_uppercase) + ['None']
    # Define class mapping explicitly to avoid index shifts
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([class_to_idx[label] for label in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2,
                                                       random_state=42,
                                                       stratify=y_enc)

    # Create datasets and loaders (1% input noise during training)
    train_ds = HandKeypointsDataset(X_train, y_train, noise_std=0.01)
    test_ds = HandKeypointsDataset(X_test, y_test, noise_std=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Build model
    input_dim = X.shape[1]
    model = MLP(input_dim, args.layers, output_dim=len(classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss and optimizer with L2 regularization (weight decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=args.learning_rate,
                     weight_decay=0.001)

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        correct = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
        print(f"Epoch {epoch}/{args.epochs} Loss: {total_loss/len(train_ds):.4f} "
              f"Acc: {correct/len(train_ds):.4f}")

    # Evaluation
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    print(f"Test Loss: {total_loss/len(test_ds):.4f} Accuracy: {correct/len(test_ds):.4f}")
    # Save model
    torch.save(model.state_dict(), 'mlp_model.pth')
    print('Model saved to mlp_model.pth')
