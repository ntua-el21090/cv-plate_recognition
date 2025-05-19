import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = CRNN(n_classes=num_classes)
    criterion = nn.CTCLoss(blank=num_classes - 1, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if images is None:
                print("Batch contains None images!")
            else:
                print("images shape:", images.shape)
            try:
                logits = model(images)
            except Exception as e:
                print("Model call failed:", e)
                continue
            targets, target_lengths = encode_labels(labels, char_to_idx)
            input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long)

            loss = criterion(logits, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), f"crnn_epoch{epoch+1}.pth")

if __name__ == "__main__":
    train()