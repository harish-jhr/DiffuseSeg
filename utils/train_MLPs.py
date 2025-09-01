import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np
import sys
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 20
OUTPUT_DIR = "/dir/DiffuseSeg/extracted_ddpm_features/"
FEATURES_SAVE_PATH = os.path.join(OUTPUT_DIR, "ddpm_pixel_features_train.pt")
LABELS_SAVE_PATH = os.path.join(OUTPUT_DIR, "ddpm_pixel_labels_train.pt")

MLP_TRAINING_CONFIG = {
    'num_mlps': 10,
    'epochs_per_mlp': 10,
    'batch_size': 64,
    'learning_rate': 0.0001,
}


LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.txt")
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = Logger(LOG_FILE)


class PixelMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc_out(x)
        return x


def calculate_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    iou_per_class = []
    predictions = predictions.cpu()
    targets = targets.cpu()
    for class_id in range(num_classes):
        tp = ((predictions == class_id) & (targets == class_id)).sum().item()
        fp = ((predictions == class_id) & (targets != class_id)).sum().item()
        fn = ((predictions != class_id) & (targets == class_id)).sum().item()
        denom = tp + fp + fn
        if denom == 0:
            iou = float('nan')
        else:
            iou = tp / denom
        iou_per_class.append(iou)
    valid_iou = [iou for iou in iou_per_class if not np.isnan(iou)]
    if len(valid_iou) == 0:
        return 0.0, iou_per_class
    return sum(valid_iou) / len(valid_iou), iou_per_class

def train_mlp_ensemble(features: torch.Tensor, labels: torch.Tensor, mlp_config: dict, train_config: dict):
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels.cpu().numpy()
    )
    y_train = y_train.long()
    y_val = y_val.long()
    print(f"Total pixels: {len(features)}")
    print(f"Training pixels: {len(X_train)}")
    print(f"Validation pixels: {len(X_val)}")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    trained_mlps = []
    mlp_val_scores = []
    input_dim = features.shape[1]

    for i in range(train_config['num_mlps']):
        print(f"\n--- Training MLP {i+1}/{train_config['num_mlps']} ---")
        mlp = PixelMLP(input_dim, NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=train_config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
        best_val_mIoU = -1.0

        for epoch in range(train_config['epochs_per_mlp']):
            mlp.train()
            total_loss = 0
            for batch_features, batch_labels in tqdm(train_loader, desc=f"MLP {i+1} Epoch {epoch+1} (Train)"):
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                outputs = mlp(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
            wandb.log({f"MLP_{i+1}/train_loss": avg_train_loss, "epoch": epoch+1})

            mlp.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_features, batch_labels in tqdm(val_loader, desc=f"MLP {i+1} Epoch {epoch+1} (Val)"):
                    batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                    outputs = mlp(batch_features)
                    _, predicted = torch.max(outputs, 1)
                    val_predictions.append(predicted)
                    val_targets.append(batch_labels)
            val_predictions = torch.cat(val_predictions)
            val_targets = torch.cat(val_targets)
            current_val_mIoU, _ = calculate_iou(val_predictions, val_targets, NUM_CLASSES)
            print(f"  Epoch {epoch+1} Validation mIoU: {current_val_mIoU:.4f}")
            wandb.log({f"MLP_{i+1}/val_mIoU": current_val_mIoU, "epoch": epoch+1})

            if current_val_mIoU > best_val_mIoU:
                best_val_mIoU = current_val_mIoU
                mlp_save_path = os.path.join(OUTPUT_DIR, f"mlp_{i+1}_best.pt")
                torch.save(mlp.state_dict(), mlp_save_path)
                print(f"    New best mIoU for MLP {i+1}. Model saved to {mlp_save_path}")

        trained_mlps.append(mlp)
        mlp_val_scores.append(best_val_mIoU)
        print(f"MLP {i+1} finished. Best Validation mIoU: {best_val_mIoU:.4f}")
        wandb.log({f"MLP_{i+1}/best_val_mIoU": best_val_mIoU})

    return trained_mlps, mlp_val_scores

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wandb.init(project="DiffuseSeg-MLP-Logs", name="train-1")

    print("Loading features and labels...")
    features = torch.load(FEATURES_SAVE_PATH)
    labels = torch.load(LABELS_SAVE_PATH)
    print(f"Loaded features shape: {features.shape} (Dtype: {features.dtype})")
    print(f"Loaded labels shape: {labels.shape} (Dtype: {labels.dtype})")

    trained_mlps, val_scores = train_mlp_ensemble(features, labels, {}, MLP_TRAINING_CONFIG)

    print("\n--- Ensemble Training Summary ---")
    for i, score in enumerate(val_scores):
        print(f"MLP {i+1} Best Validation mIoU: {score:.4f}")
        wandb.log({f"MLP_{i+1}/summary_best_val_mIoU": score})

    avg_mIoU = sum(val_scores) / len(val_scores)
    print(f"\nAverage Best Validation mIoU across {MLP_TRAINING_CONFIG['num_mlps']} MLPs: {avg_mIoU:.4f}")
    wandb.log({"avg_best_val_mIoU": avg_mIoU})
