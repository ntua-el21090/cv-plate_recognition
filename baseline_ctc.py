
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

province_chars = "京津沪渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青蒙桂宁新藏港澳台学警领"
letters        = "ABCDEFGHJKLMNPQRSTUVWXYZ"
digits         = "0123456789"

ALPHABET    = province_chars + letters + digits
NUM_CLASSES = len(ALPHABET)
BLANK_IDX   = NUM_CLASSES

CHAR_TO_IDX: Dict[str,int] = {ch: i for i, ch in enumerate(ALPHABET)}
IDX_TO_CHAR: Dict[int,str] = {i: ch for ch, i in CHAR_TO_IDX.items()}

class PlateDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, img_size: Tuple[int,int],
                 alphabet: Dict[str,int], transforms_compose: transforms.Compose = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms_compose
        self.alphabet  = alphabet

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_image_paths: List[Path] = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in exts]
        )
        if len(all_image_paths) == 0:
            raise RuntimeError(f"No images found in {images_dir!r}.")

        self.samples: List[Tuple[Path,Path]] = []
        for img_path in all_image_paths:
            txt_path = self.labels_dir / (img_path.stem + ".txt")
            if not txt_path.exists():
                print(f"[WARN] skipping '{img_path.name}': no label '{txt_path.name}'")
                continue
            self.samples.append((img_path, txt_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No matching image/label pairs found in {images_dir!r} ↔ {labels_dir!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor,torch.Tensor]:
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        with open(lbl_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        if len(line) != 8:
            raise ValueError(f"Label '{line}' in {lbl_path!r} is not length=8.")

        label_idx = torch.zeros(8, dtype=torch.long)
        for i, ch in enumerate(line):
            if ch not in self.alphabet:
                raise ValueError(f"Character '{ch}' in {lbl_path!r} not in ALPHABET.")
            label_idx[i] = self.alphabet[ch]

        return image, label_idx

class BaselineCTC(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: bool = False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2,2)),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = features.squeeze(2).permute(0, 2, 1).contiguous()
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out)

def compute_char_and_plate_accuracy_ctc(preds, labels):
    with torch.no_grad():
        B, T, Vp1 = preds.size()
        argmaxed = preds.argmax(dim=2)
        all_char_correct = 0
        all_plate_correct = 0

        for b in range(B):
            seq = argmaxed[b].cpu().tolist()
            collapsed = []
            prev = None
            for idx in seq:
                if idx != prev and idx != BLANK_IDX:
                    collapsed.append(idx)
                prev = idx

            if len(collapsed) >= 8:
                final_idxs = collapsed[:8]
            else:
                final_idxs = [labels[b, 0].item()] * 8 if not collapsed else collapsed + [collapsed[-1]] * (8 - len(collapsed))

            pred_str = "".join(IDX_TO_CHAR[i] for i in final_idxs)
            gt_str = "".join(IDX_TO_CHAR[i] for i in labels[b].cpu().tolist())

            all_char_correct += sum(c1 == c2 for c1, c2 in zip(pred_str, gt_str))
            if pred_str == gt_str:
                all_plate_correct += 1

        return all_char_correct / (B * 8), all_plate_correct / B

def compute_f1_score_ctc_pytorch(preds, labels):
    with torch.no_grad():
        B, T, Vp1 = preds.size()
        argmaxed = preds.argmax(dim=2)

        tp = fp = fn = 0

        for b in range(B):
            seq = argmaxed[b].cpu().tolist()
            collapsed = []
            prev = None
            for idx in seq:
                if idx != prev and idx != BLANK_IDX:
                    collapsed.append(idx)
                prev = idx

            pred_idxs = collapsed[:8] if len(collapsed) >= 8 else ([labels[b, 0].item()] * 8 if not collapsed else collapsed + [collapsed[-1]] * (8 - len(collapsed)))
            true_idxs = labels[b].cpu().tolist()

            for p, t in zip(pred_idxs, true_idxs):
                if p == t:
                    tp += 1
                else:
                    fp += 1
                    fn += 1

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg.get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "cuda"
                          else "mps" if device_str == "mps" and torch.backends.mps.is_available()
                          else "cpu")

    train_transforms = transforms.Compose([
        transforms.Resize(cfg["recognition_image_size"]),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(3, fill=(0,0,0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(cfg["recognition_image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    train_dataset = PlateDataset(cfg["train_images"], cfg["train_labels"],
                                 cfg["recognition_image_size"], CHAR_TO_IDX, train_transforms)
    test_dataset  = PlateDataset(cfg["test_images"], cfg["test_labels"],
                                 cfg["recognition_image_size"], CHAR_TO_IDX, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers", 2), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg.get("num_workers", 2), pin_memory=True)

    model = BaselineCTC(NUM_CLASSES, False).to(device)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            B = labels.size(0)
            targets = labels.view(-1).to(device)
            target_lengths = torch.full((B,), 8, dtype=torch.long).to(device)

            logits = model(images)
            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
            input_lengths = torch.full((B,), log_probs.size(0), dtype=torch.long).to(device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}/{args.epochs}  Avg CTC Loss = {total_loss/len(train_loader):.4f}")

    print("\nRunning evaluation on test set...")
    model.eval()
    total_loss = total_char_acc = total_plate_acc = total_f1 = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            B = labels.size(0)
            targets = labels.view(-1).to(device)
            target_lengths = torch.full((B,), 8, dtype=torch.long).to(device)

            logits = model(images)
            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
            input_lengths = torch.full((B,), log_probs.size(0), dtype=torch.long).to(device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            ca, pa = compute_char_and_plate_accuracy_ctc(logits, labels)
            total_char_acc += ca
            total_plate_acc += pa
            total_f1 += compute_f1_score_ctc_pytorch(logits, labels)

    N = len(test_loader)
    print("\nBaseline Results on Test Set")
    print(f"CTC Loss        = {total_loss / N:.4f}")
    print(f"Character Acc   = {total_char_acc / N:.4f}")
    print(f"Plate Acc       = {total_plate_acc / N:.4f}")
    print(f"F1 Score        = {total_f1 / N:.4f}\n")

if __name__ == "__main__":
    main()
