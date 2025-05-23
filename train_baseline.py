import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models.baseline_model import CRNN
from dataset.datasets import CCPDRecognitionDataset as PlateDataset
from config import *
import json
from tqdm import tqdm

def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), labels

def encode_labels(labels, char_to_idx):
    targets = []
    lengths = []
    for label in labels:
        encoded = [char_to_idx[c] for c in label]
        targets.extend(encoded)
        lengths.append(len(encoded))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

def train(train_json_path="dataset/train.json", val_json_path="dataset/val.json", image_root=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(train_json_path) as f:
        train_data = json.load(f)
    with open(val_json_path) as f:
        val_data = json.load(f)

    char_set = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-", "_"
    ]
    char_to_idx = {c: i for i, c in enumerate(char_set)}
    idx_to_char = {i: c for c, i in enumerate(char_set)}
    num_classes = len(char_set) + 1  # +1 for CTC blank

    train_dataset = PlateDataset(train_json_path, image_root=image_root)
    val_dataset = PlateDataset(val_json_path, image_root=image_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    model = CRNN(n_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=num_classes - 1, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(30):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            
            images = images.to(device)
            try:
                logits = model(images)
                # print("logits type:", type(logits))
                if logits is None or not isinstance(logits, torch.Tensor):
                    print("Model returned invalid logits:", logits)
                    continue
            except Exception as e:
                print("Model call failed:", e)
                continue
            targets, target_lengths = encode_labels(labels, char_to_idx)
            input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long)

            with autocast():
                loss = criterion(logits, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                targets, target_lengths = encode_labels(labels, char_to_idx)
                input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long)
                loss = criterion(logits, targets, input_lengths, target_lengths)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"/content/drive/MyDrive/cv_plate_recognition/crnn_best.pth")
            print(f"Saved new best model with val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    train()