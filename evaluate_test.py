"""
This script supports two modes:
1. Evaluation of test set:
   python evaluate_test.py --mode eval --config config.yaml --checkpoint PDLPR/checkpoints/best_model.pth --show_samples

2. Inference on custom images using YOLO + PDLPR:
   python evaluate_test.py --mode infer --config config.yaml --checkpoint PDLPR/checkpoints/best_model.pth --det-weights yolov5s.pt --source input_folder
"""
import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import sys
import pathlib
from PIL import Image
import torchvision.transforms as T
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes as scale_coords
from yolov5.utils.augmentations import letterbox

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

province_chars = "京津沪渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青蒙桂宁新藏港澳台学警领"
letters        = "ABCDEFGHJKLMNPQRSTUVWXYZ"
digits         = "0123456789"
ALPHABET       = province_chars + letters + digits
NUM_CLASSES    = len(ALPHABET)

CHAR_TO_IDX: Dict[str,int] = {ch: i for i, ch in enumerate(ALPHABET)}
IDX_TO_CHAR: Dict[int,str] = {i: ch for ch, i in CHAR_TO_IDX.items()}

class PlateDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, img_size: Tuple[int,int],
                 alphabet: Dict[str,int], transforms_compose: transforms.Compose = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms_compose
        self.alphabet = alphabet

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in exts])
        if len(all_images) == 0:
            raise RuntimeError(f"No image files found in {images_dir!r}.")

        self.samples: List[Tuple[Path,Path]] = []
        for img_path in all_images:
            txt_path = self.labels_dir / (img_path.stem + ".txt")
            if not txt_path.exists():
                print(f"[WARN] skipping '{img_path.name}': no label '{txt_path.name}'")
                continue
            self.samples.append((img_path, txt_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No (image,label) pairs found in {images_dir!r} ↔ {labels_dir!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

class PlateClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=pretrained_backbone)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,8))
        self.num_positions = 8
        self.in_channels = 512
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList([
            nn.Linear(self.in_channels, self.num_classes) for _ in range(self.num_positions)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)
        p = self.adaptive_pool(f)
        p = p.squeeze(2).permute(0,2,1).contiguous()
        logits_list = [self.classifiers[pos](p[:, pos, :]).unsqueeze(1) for pos in range(self.num_positions)]
        return torch.cat(logits_list, dim=1)

def compute_char_and_plate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        B = logits.size(0)
        preds = logits.argmax(dim=2)
        correct_chars = (preds == labels).sum().item()
        char_acc = correct_chars / (B * 8)
        exact_plates = (preds == labels).all(dim=1).sum().item()
        plate_acc = exact_plates / B
    return char_acc, plate_acc

def compute_f1_score_pytorch(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=2)
        B = preds.size(0)
        tp = fp = fn = 0
        for b in range(B):
            pred_idxs = preds[b].tolist()
            true_idxs = labels[b].tolist()
            for p, t in zip(pred_idxs, true_idxs):
                if p == t:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)

def show_random_samples(model, dataset: PlateDataset, device: torch.device, cfg: dict, num_samples: int = 5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 2 * num_samples))
    shown = 0
    mean = np.array(cfg["mean"])
    std  = np.array(cfg["std"])
    with torch.no_grad():
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for idx in indices:
            img_tensor, label_idxs = dataset[idx]
            x = img_tensor.unsqueeze(0).to(device)
            logits = model(x)
            pred_idxs = logits.argmax(dim=2)[0].cpu().tolist()
            pred_str  = "".join(IDX_TO_CHAR[i] for i in pred_idxs)
            gt_str    = "".join(IDX_TO_CHAR[int(i)] for i in label_idxs)
            img_np = img_tensor.cpu().numpy().transpose(1,2,0)
            img_np = (img_np * std) + mean
            img_np = np.clip(img_np, 0, 1)
            ax = axes if num_samples == 1 else axes[shown]
            ax.imshow(img_np)
            ax.axis("off")
            ax.set_title(f"GT: {gt_str}   Pred: {pred_str}", fontsize=10)
            shown += 1
            if shown >= num_samples:
                break
    plt.tight_layout()
    plt.show()

def load_detector(weights, device):
    model = DetectMultiBackend(weights, device=device)
    return model, model.stride, model.names

def detect_plates(det_model, stride, img_path, conf_thres=0.25, iou_thres=0.45):
    img0_pil = Image.open(img_path).convert("RGB")
    im_np = np.array(img0_pil)  # H×W×3 RGB
    padded, ratio, (dw, dh) = letterbox(im_np, (640, 640), stride=stride)
    padded = padded[:, :, ::-1]
    img = padded.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(det_model.device)
    im = im.half() if det_model.fp16 else im.float()
    im /= 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)

    pred = det_model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    pad_h, pad_w = im.shape[2], im.shape[3]
    return pred, img0_pil, (pad_h, pad_w), (img0_pil.height, img0_pil.width)

def crop_and_recognize(pred, img0, pad_shape, orig_shape, recog_model, recog_tf, device):
    pad_h, pad_w = pad_shape
    h0, w0 = orig_shape
    plates = []
    for det in pred:
        if det is None or len(det) == 0:
            continue
        det[:, :4] = scale_coords((pad_h, pad_w), det[:, :4], (h0, w0)).round()
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = img0.crop((x1, y1, x2, y2)).resize((256, 32), Image.BILINEAR)
            x = recog_tf(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = recog_model(x)
                idxs = logits.argmax(dim=2)[0].cpu().tolist()
            plate_str = "".join(IDX_TO_CHAR[i] for i in idxs)
            plates.append((plate_str, (x1, y1, x2, y2), float(conf)))
    return plates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--show_samples", action="store_true")
    parser.add_argument("--mode", choices=["eval", "infer"], default="eval",
                        help="Choose whether to evaluate test set or run inference on images")
    parser.add_argument("--det-weights", type=str, help="Path to YOLOv5 weights (for inference)")
    parser.add_argument("--source", type=str, help="Path to folder of input images (for inference)")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg.get("device", "cuda")
    if torch.cuda.is_available() and device_str == "cuda":
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}\n")

    img_h, img_w = cfg["recognition_image_size"]
    batch_size = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 2)
    test_transforms = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    print("Loading model checkpoint…")
    model = PlateClassifier(num_classes=NUM_CLASSES, pretrained_backbone=False).to(device)
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if args.mode == "infer":
        if not args.det_weights or not args.source:
            raise ValueError("Inference mode requires --det-weights and --source.")
        print("[INFO] Running inference mode...")
        det_model, stride, _ = load_detector(args.det_weights, device)
        for img_path in Path(args.source).glob("*.[jp][pn]g"):
            pred, img0, pad_shape, orig_shape = detect_plates(
                det_model, stride, img_path, args.conf_thres, args.iou_thres)
            recog_tf = T.Compose([
                T.Resize((img_h, img_w)),
                T.ToTensor(),
                T.Normalize(mean=cfg["mean"], std=cfg["std"]),
            ])
            plates = crop_and_recognize(
                pred, img0, pad_shape, orig_shape, model, recog_tf, device)
            print(f"\nResults for {img_path.name}:")
            for plate, box, conf in plates:
                print(f"  {plate:>8s}  @ {box}  ({conf:.2f})")
        return

    print("Building test‐set DataLoader…")
    test_dataset = PlateDataset(cfg["test_images"], cfg["test_labels"], (img_h, img_w), CHAR_TO_IDX, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    print(f"Number of test samples: {len(test_dataset)}\n")

    print("Evaluating test set…")
    criterion = nn.CrossEntropyLoss()
    total_loss = total_char_acc = total_plate_acc = total_f1_score = 0.0
    total_batches = len(test_loader)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = sum(criterion(logits[:, pos, :], labels[:, pos]) for pos in range(8)) / 8.0
            total_loss += loss.item()
            char_acc, plate_acc = compute_char_and_plate_accuracy(logits, labels)
            total_char_acc += char_acc
            total_plate_acc += plate_acc
            total_f1_score += compute_f1_score_pytorch(logits, labels)

    print("\n=== Test Set Results ===")
    print(f"Test Loss            = {total_loss / total_batches:.4f}")
    print(f"Test Character Acc   = {total_char_acc / total_batches:.4f}")
    print(f"Test Exact‐Plate Acc = {total_plate_acc / total_batches:.4f}")
    print(f"Test F1 Score        = {total_f1_score / total_batches:.4f}\n")

    if args.show_samples:
        print("Displaying sample predictions…")
        show_random_samples(model, test_dataset, device, cfg, num_samples=5)

if __name__ == "__main__":
    main()
