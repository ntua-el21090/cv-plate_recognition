"""
Same as before, except we replace train_transforms with a much stronger
“glare/blur/skew” pipeline. Run ~10 epochs after this change.

"""
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

province_chars = "京津沪渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青蒙桂宁新藏港澳台学警领"
letters        = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # exclude I, O
digits         = "0123456789"

ALPHABET = province_chars + letters + digits
NUM_CLASSES = len(ALPHABET)

CHAR_TO_IDX: Dict[str, int] = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR: Dict[int, str] = {idx: char for char, idx in CHAR_TO_IDX.items()}

class PlateDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 labels_dir: str,
                 img_size: Tuple[int, int],
                 alphabet: Dict[str,int],
                 transforms_compose: transforms.Compose = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms_compose
        self.alphabet = alphabet

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths: List[Path] = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in exts]
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in {images_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in self.image_paths:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"Label file {label_path} not found for image {img_path}")
            self.samples.append((img_path, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        with open(label_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        if len(line) != 8:
            raise ValueError(f"Label '{line}' in {label_path} is not length=8")

        label_str = line
        label_idx = torch.zeros(8, dtype=torch.long)
        for i, ch in enumerate(label_str):
            if ch not in self.alphabet:
                raise ValueError(f"Character '{ch}' in label '{label_str}' not in ALPHABET")
            label_idx[i] = self.alphabet[ch]

        return image, label_idx

class PlateClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=pretrained_backbone)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        self.num_positions = 8
        self.in_channels = 512
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList([
            nn.Linear(self.in_channels, self.num_classes) for _ in range(self.num_positions)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)       # [B, 512, H', W']
        pooled   = self.adaptive_pool(features)    # [B, 512, 1, 8]
        pooled   = pooled.squeeze(2)               # [B, 512, 8]
        pooled   = pooled.permute(0, 2, 1).contiguous()  # [B, 8, 512]

        logits_list = []
        for pos in range(self.num_positions):
            vec   = pooled[:, pos, :]              # [B, 512]
            logit = self.classifiers[pos](vec)     # [B, num_classes]
            logits_list.append(logit.unsqueeze(1)) # [B, 1, num_classes]
        logits = torch.cat(logits_list, dim=1)     # [B, 8, num_classes]
        return logits

def compute_char_and_plate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        B = logits.size(0)
        preds = logits.argmax(dim=2)  # [B, 8]
        correct_chars = (preds == labels).sum().item()
        char_acc = correct_chars / (B * 8)
        correct_plates = (preds == labels).all(dim=1).sum().item()
        plate_acc = correct_plates / B
    return char_acc, plate_acc

def main():
    parser = argparse.ArgumentParser(description="Train with heavy glare/blur/skew augmentations")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_images_dir = cfg["train_images"]
    train_labels_dir = cfg["train_labels"]
    val_images_dir   = cfg["val_images"]
    val_labels_dir   = cfg["val_labels"]

    img_h, img_w = cfg["recognition_image_size"]
    seq_length   = cfg["seq_length"]

    batch_size   = cfg["batch_size"]
    num_epochs   = cfg["epochs"]
    base_lr      = cfg["lr"]

    pct_start       = cfg.get("pct_start", 0.3)
    div_factor      = cfg.get("div_factor", 10.0)
    final_div       = cfg.get("final_div", 10000.0)
    anneal_strategy = cfg.get("anneal_strategy", "cos")

    log_every_n     = cfg.get("log_every_n_steps", 50)
    device_str      = cfg.get("device", "cuda")
    device          = torch.device(device_str if torch.cuda.is_available() or device_str != "cuda" else "cpu")
    save_path       = cfg.get("save_path", "PDLPR/checkpoints/best_model.pth")
    num_workers     = cfg.get("num_workers", 2)

    train_transforms = transforms.Compose([
        transforms.Resize((img_h, img_w)),

        # VERY STRONG ColorJitter to mimic extreme glare/overexposure:
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.5, hue=0.1),

        # Slight random rotation (±10°) to mimic skew:
        transforms.RandomRotation(degrees=10, fill=(0,0,0)),

        # Very aggressive perspective warp (distortion_scale=0.4, p=0.8):
        transforms.RandomPerspective(distortion_scale=0.4, p=0.8),

        # Moderate Gaussian blur (kernel up to 5×5, sigma up to 2.0) at p=0.5:
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 2.0))
        ], p=0.5),

        # Random grayscale to simulate washed‐out lighting:
        transforms.RandomGrayscale(p=0.1),

        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    train_dataset = PlateDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        img_size=(img_h, img_w),
        alphabet=CHAR_TO_IDX,
        transforms_compose=train_transforms
    )
    val_dataset = PlateDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        img_size=(img_h, img_w),
        alphabet=CHAR_TO_IDX,
        transforms_compose=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = PlateClassifier(num_classes=NUM_CLASSES, pretrained_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)

    total_steps = num_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div,
        anneal_strategy=anneal_strategy
    )

    best_plate_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Train", leave=False), start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)  # [B,8,71]
            loss = 0.0
            for pos in range(8):
                loss += criterion(logits[:, pos, :], labels[:, pos])
            loss = loss / 8.0
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if batch_idx % log_every_n == 0 or batch_idx == len(train_loader):
                avg_loss_so_far = running_loss / batch_idx
                current_lr = optimizer.param_groups[0]["lr"]
                tqdm.write(f"  [Train] Step {batch_idx}/{len(train_loader)}  AvgLoss={avg_loss_so_far:.4f}  LR={current_lr:.2e}")

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        accum_char_acc = 0.0
        accum_plate_acc = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validate", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                loss = 0.0
                for pos in range(8):
                    loss += criterion(logits[:, pos, :], labels[:, pos])
                loss = loss / 8.0
                running_val_loss += loss.item()

                char_acc, plate_acc = compute_char_and_plate_accuracy(logits, labels)
                accum_char_acc += char_acc
                accum_plate_acc += plate_acc

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_char_acc = accum_char_acc / len(val_loader)
        avg_val_plate_acc = accum_plate_acc / len(val_loader)

        print(f"  → Train Loss      = {avg_train_loss:.4f}")
        print(f"  → Val   Loss      = {avg_val_loss:.4f}")
        print(f"  → Val   Char Acc  = {avg_val_char_acc:.4f}")
        print(f"  → Val   Plate Acc = {avg_val_plate_acc:.4f}")

        # Save best
        if avg_val_plate_acc > best_plate_acc:
            best_plate_acc = avg_val_plate_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_plate_acc": avg_val_plate_acc,
                "val_char_acc": avg_val_char_acc,
                "config": cfg
            }, save_path)
            print(f"  [+] Saved new best checkpoint to: {save_path}")

    print("\nTraining complete.")
if __name__ == "__main__":
    main()
