"""
Save the best checkpoint (by character accuracy) and finally evaluate on the test set.

Usage:
  python train_pdlpr.py --config config.yaml
"""

import os
import math
import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import yaml
import torch.nn.functional as F

def fixed_length_decode_ctc(logp: torch.Tensor, blank_idx: int, seq_len: int):
    T, B, C = logp.shape
    assert T == seq_len, f"T ({T}) must equal seq_len ({seq_len})"
    argmaxed = logp.argmax(dim=-1).cpu().tolist() 

    out = [[] for _ in range(B)]
    for t in range(T):
        for b in range(B):
            out[b].append(argmaxed[t][b])

    final_seqs = []
    for b in range(B):
        seq = out[b]
        collapsed = []
        prev = None
        for c in seq:
            if c != prev:
                collapsed.append(c)
            prev = c
        no_blanks = [c for c in collapsed if c != blank_idx]
        if len(no_blanks) < seq_len:
            no_blanks += [blank_idx] * (seq_len - len(no_blanks))
        else:
            no_blanks = no_blanks[:seq_len]
        final_seqs.append(no_blanks)
    return final_seqs  # [[8‐indices], …] length B


def beam_search_ctc(logp: torch.Tensor, beam_width: int, blank_idx: int):
    T, B, C = logp.shape
    logp_cpu = logp.detach().cpu()
    all_beams = []

    for b in range(B):
        beam = [(0.0, [])]
        for t in range(T):
            new_beam = {}
            step = logp_cpu[t, b]
            topk_vals, topk_idxs = torch.topk(step, k=min(C, beam_width))
            for (prefix_score, prefix_seq) in beam:
                for k in range(len(topk_idxs)):
                    idx_c = int(topk_idxs[k].item())
                    new_score = prefix_score + float(topk_vals[k].item())
                    new_seq = prefix_seq + [idx_c]
                    key = tuple(new_seq)
                    if key not in new_beam or new_score > new_beam[key]:
                        new_beam[key] = new_score
            beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            beam = [(s, list(seq)) for (seq, s) in beam]
        beam_sorted = sorted(beam, key=lambda x: x[0], reverse=True)
        seqs_only = [seq for (s, seq) in beam_sorted]
        all_beams.append(seqs_only[:beam_width])
    return all_beams


def compute_char_plate_accuracy(preds: list, targets: list):
    assert len(preds) == len(targets)
    total_chars = 0
    correct_chars = 0
    total_plates = 0
    correct_plates = 0

    for p_seq, g_seq in zip(preds, targets):
        total_chars += len(g_seq)
        for i in range(len(g_seq)):
            if i < len(p_seq) and p_seq[i] == g_seq[i]:
                correct_chars += 1
        total_plates += 1
        if p_seq == g_seq:
            correct_plates += 1

    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    plate_acc = correct_plates / total_plates if total_plates > 0 else 0.0
    return char_acc, plate_acc

class CRNN_CTC(nn.Module):
    def __init__(self, num_chars: int, seq_len: int, img_h: int, img_w: int):
        super().__init__()
        # 0) Spatial Transformer Net (STN)
        self.stn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.Conv2d(8,10, kernel_size=5, padding=2),
            nn.MaxPool2d(2), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(10 * (img_h//4) * (img_w//4), 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )
        # Initialize STN to identity
        self.stn[-1].weight.data.zero_()
        self.stn[-1].bias.data.copy_(torch.tensor([1,0,0, 0,1,0], dtype=torch.float))

        # Pretrained ResNet‐34 backbone (drop avgpool + fc)
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B,512,H',W')

        # 1×1 projection + BN + ReLU
        self.proj = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Squeeze‐and‐Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512//16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512//16, 512, kernel_size=1),
            nn.Sigmoid(),
        )
        # Pool height→1, width→seq_len
        self.pool = nn.AdaptiveAvgPool2d((1, seq_len))
        # BiLSTM (3 layers, bidirectional)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.classifier = nn.Linear(512 * 2, num_chars + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        theta = self.stn(x).view(-1, 2, 3)
        grid  = F.affine_grid(theta, x.size(), align_corners=False)
        x     = F.grid_sample(x, grid, align_corners=False)

        f = self.backbone(x)     # (B,512,H',W')
        p = self.proj(f)         # (B,512,H',W')
        w = self.se(p)           # (B,512,1,1)
        p = p * w                # channel‐wise reweight

        p = self.pool(p)         # (B,512,1,seq_len)
        t = p.squeeze(2).permute(0,2,1)  # (B, seq_len, 512)

        out, _ = self.lstm(t)    # (B, seq_len, 1024)

        logits = self.classifier(out)  # (B, seq_len, num_chars+1)
        return logits

class PlateDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 lbl_dir: str,
                 c2i: dict,
                 mean: list,
                 std: list,
                 img_h: int,
                 img_w: int,
                 augment: bool = False,
                 aug_brightness_contrast: bool = False,
                 aug_random_rotation: bool = False,
                 aug_random_perspective: bool = False):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.files = sorted(self.img_dir.glob("*.jpg"))
        assert self.files, f"No .jpg files found in {img_dir}"

        self.labels = {}
        for img_path in self.files:
            txt_path = self.lbl_dir / f"{img_path.stem}.txt"
            assert txt_path.exists(), f"Missing label file {txt_path}"
            plate_str = txt_path.read_text(encoding="utf-8").strip()
            assert len(plate_str) == 8, (
                f"Label '{plate_str}' for {img_path.stem} has length {len(plate_str)}, expected 8"
            )
            self.labels[img_path.stem] = plate_str

        self.c2i = c2i
        self.augment = augment
        self.aug_bc = aug_brightness_contrast
        self.aug_rot = aug_random_rotation
        self.aug_persp = aug_random_perspective

        tf_list = []
        if self.augment:
            if self.aug_bc:
                tf_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
            if self.aug_rot:
                tf_list.append(transforms.RandomRotation(degrees=5))
            if self.aug_persp:
                tf_list.append(transforms.RandomPerspective(distortion_scale=0.1, p=0.5))
        tf_list += [
            transforms.Resize((img_h, img_w), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.tf = transforms.Compose(tf_list)

        print(f"\n[DEBUG] First 3 items in PlateDataset (augment={self.augment}):")
        for i in range(min(3, len(self.files))):
            stem = self.files[i].stem
            plate_str = self.labels[stem]
            idx_list = [self.c2i[c] for c in plate_str]
            print(f"  {i:02d}: {stem} → '{plate_str}'  indices={idx_list}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        plate_str = self.labels[img_path.stem]  # length=8
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        tgt_idx = torch.tensor([self.c2i[c] for c in plate_str], dtype=torch.long)
        tgt_len = 8
        return x, tgt_idx, tgt_len

class CCPD_Dataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 gt_path: str,
                 c2i: dict,
                 mean: list,
                 std: list,
                 img_h: int,
                 img_w: int,
                 augment: bool = False,
                 aug_brightness_contrast: bool = False,
                 aug_random_rotation: bool = False,
                 aug_random_perspective: bool = False):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.gt_path = Path(gt_path)
        self.c2i = c2i
        self.augment = augment
        self.aug_bc = aug_brightness_contrast
        self.aug_rot = aug_random_rotation
        self.aug_persp = aug_random_perspective

        tf_list = []
        if self.augment:
            if self.aug_bc:
                tf_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
            if self.aug_rot:
                tf_list.append(transforms.RandomRotation(degrees=5))
            if self.aug_persp:
                tf_list.append(transforms.RandomPerspective(distortion_scale=0.1, p=0.5))
        tf_list += [
            transforms.Resize((img_h, img_w), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.tf = transforms.Compose(tf_list)

        self.entries = []
        with open(self.gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                img_name = parts[0]
                plate_str = parts[1]
                if len(plate_str) != 8:
                    continue
                self.entries.append((img_name, plate_str))
        assert self.entries, f"No valid entries in {gt_path}"

        print(f"\n[DEBUG] First 3 CCPD entries (augment={self.augment}):")
        for i in range(min(3, len(self.entries))):
            img_name, plate_str = self.entries[i]
            idx_list = [self.c2i[c] for c in plate_str]
            print(f"  {i:02d}: {img_name} → '{plate_str}'  indices={idx_list}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_name, plate_str = self.entries[idx]
        img_path = self.img_dir / img_name
        assert img_path.exists(), f"Missing image {img_path}"
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        tgt_idx = torch.tensor([self.c2i[c] for c in plate_str], dtype=torch.long)
        tgt_len = len(plate_str)  # should be 8
        return x, tgt_idx, tgt_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg.get('device') == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif cfg.get('device') == 'mps' and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device('mps')
        torch.set_float32_matmul_precision('high')
    else:
        device = torch.device('cpu')
    print("DEVICE:", device)

    FULL_CHINESE_PROVINCES = [
        "京","津","沪","渝","冀","晋","辽","吉","黑","苏","浙","皖",
        "闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云",
        "陕","甘","青","蒙","藏","宁","新","使","港","澳"
    ]
    DIGITS  = [str(i) for i in range(10)]
    LETTERS = list(__import__('string').ascii_uppercase)  # A–Z

    chars = FULL_CHINESE_PROVINCES + DIGITS + LETTERS
    chars.append("_")
    num_chars_cfg = len(chars) - 1
    blank_idx = num_chars_cfg

    c2i = {c: i for i, c in enumerate(chars)}

    bcounts = defaultdict(lambda: defaultdict(lambda: 1e-6))
    for split in ('train', 'val', 'test'):
        lbl_path = Path(cfg[f"{split}_labels"])
        if lbl_path.is_dir():
            for fpath in lbl_path.glob("*.txt"):
                s = fpath.read_text(encoding="utf-8").strip()
                for a, b in zip(s, s[1:]):
                    bcounts[a][b] += 1
        elif lbl_path.is_file():
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                    s = parts[1]
                    for a, b in zip(s, s[1:]):
                        bcounts[a][b] += 1
    bigram_log_probs = {
        a: {b: math.log(cnt / sum(nxt.values())) for b, cnt in nxt.items()}
        for a, nxt in bcounts.items()
    }

    img_h, img_w = cfg['recognition_image_size']
    seq_len      = cfg['seq_length']
    assert seq_len == 8, "seq_length must be 8 for CCPD"
    batch_size   = cfg['batch_size']
    epochs       = cfg['epochs']
    lr           = cfg['lr']
    pct_start    = cfg['pct_start']
    div_factor   = cfg['div_factor']
    final_div    = cfg['final_div']
    anneal_strategy = cfg['anneal_strategy']

    train_images = cfg['train_images']
    train_labels = cfg['train_labels']
    val_images   = cfg['val_images']
    val_labels   = cfg['val_labels']
    test_images  = cfg['test_images']
    test_labels  = cfg['test_labels']

    mean = cfg['mean']
    std  = cfg['std']
    num_workers = cfg['num_workers']
    save_path   = Path(cfg['save_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if Path(train_labels).is_dir():
        # “one txt per image” → PlateDataset
        train_ds = PlateDataset(
            img_dir=train_images,
            lbl_dir=train_labels,
            c2i=c2i,
            mean=mean,
            std=std,
            img_h=img_h,
            img_w=img_w,
            augment=cfg['aug_enable'],
            aug_brightness_contrast=cfg['aug_brightness_contrast'],
            aug_random_rotation=cfg['aug_random_rotation'],
            aug_random_perspective=cfg['aug_random_perspective']
        )
        val_ds = PlateDataset(
            img_dir=val_images,
            lbl_dir=val_labels,
            c2i=c2i,
            mean=mean,
            std=std,
            img_h=img_h,
            img_w=img_w,
            augment=False
        )
        test_ds = PlateDataset(
            img_dir=test_images,
            lbl_dir=test_labels,
            c2i=c2i,
            mean=mean,
            std=std,
            img_h=img_h,
            img_w=img_w,
            augment=False
        )
    else:
        # CCPD single gt.txt → CCPD_Dataset
        train_ds = CCPD_Dataset(
            img_dir=train_images,
            gt_path=train_labels,
            c2i=c2i,
            mean=mean,
            std=std,
            img_h=img_h,
            img_w=img_w,
            augment=cfg['aug_enable'],
            aug_brightness_contrast=cfg['aug_brightness_contrast'],
            aug_random_rotation=cfg['aug_random_rotation'],
            aug_random_perspective=cfg['aug_random_perspective']
        )
        val_ds = CCPD_Dataset(
            img_dir=val_images,
            gt_path=val_labels,
            c2i=c2i,
            mean=mean,
            std=std,
            img_h=img_h,
            img_w=img_w,
            augment=False
        )
        test_ds = CCPD_Dataset(
            img_dir=test_images,
            gt_path=test_labels,
            c2i=c2i,
            mean=mean,
            std=std,
            img_h=img_h,
            img_w=img_w,
            augment=False
        )

    print(f"\n[CHECK] train: {len(train_ds)} samples")
    print(f"[CHECK] val:   {len(val_ds)} samples")
    print(f"[CHECK] test:  {len(test_ds)} samples\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )

    print("\n[OVERFIT TEST] Overfitting 4 samples for 50 iters...\n")
    temp_model = CRNN_CTC(num_chars=num_chars_cfg, seq_len=seq_len,
                          img_h=img_h, img_w=img_w).to(device)
    temp_model.train()
    sample_batch = next(iter(train_loader))
    imgs0, tgts0, _ = sample_batch
    imgs0 = imgs0[:4].to(device)
    tgts0 = tgts0[:4].to(device)
    tgt_lens0 = torch.full((4,), seq_len, dtype=torch.long, device=device)

    optim_of = optim.AdamW(temp_model.parameters(), lr=1e-4)
    ctc_of   = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    for i in range(1, 51):
        optim_of.zero_grad()
        logits_of = temp_model(imgs0)                         # (4,8,C)
        logp_of   = logits_of.log_softmax(-1).permute(1, 0, 2) # (T=8,B=4,C)
        input_lens_of = torch.full((4,), seq_len, dtype=torch.long, device=device)
        flat_tgts_of = tgts0.view(-1)  # (4*8,)
        loss_of = ctc_of(logp_of, flat_tgts_of, input_lens_of, tgt_lens0)
        loss_of.backward()
        optim_of.step()
        if i % 10 == 0:
            print(f"[OVERFIT] Iter {i:02d}   loss={loss_of.item():.4f}")
    print("[OVERFIT TEST] Done\n")

    # Build model + freeze backbone+STN on epoch 1
    model = CRNN_CTC(num_chars=num_chars_cfg, seq_len=seq_len,
                     img_h=img_h, img_w=img_w).to(device)
    print("\nModel created:\n", model)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.stn.parameters():
        p.requires_grad = False

    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [{"params": head_params, "lr": lr}],
        weight_decay=cfg.get('weight_decay', 0.0)
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div,
        anneal_strategy=anneal_strategy
    )

    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    best_char_acc = 0.0
    epochs_since_improve = 0

    for ep in range(1, epochs + 1):
        print(f"\n=== Epoch {ep}/{epochs} ===")

        # Unfreeze backbone+STN at epoch 2
        if ep == 2:
            print("[INFO] Unfreezing backbone + STN at epoch 2.")
            for p in model.backbone.parameters():
                p.requires_grad = True
            for p in model.stn.parameters():
                p.requires_grad = True

            backbone_params = []
            head_params     = []
            for name, param in model.named_parameters():
                if "lstm" in name or "classifier" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr * 0.1},
                    {"params": head_params,     "lr": lr}
                ],
                weight_decay=cfg.get('weight_decay', 0.0)
            )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[lr * 0.1, lr],
                steps_per_epoch=len(train_loader),
                epochs=epochs - 1,
                pct_start=0.1,
                div_factor=div_factor,
                final_div_factor=final_div,
                anneal_strategy=anneal_strategy
            )

        model.train()
        running_loss = 0.0
        t0 = time.time()

        for step, (images, targets, tgt_lens) in enumerate(train_loader, start=1):
            images  = images.to(device)    # (B,3,32,256)
            targets = targets.to(device)   # (B,8)
            B = images.size(0)
            tgt_lens_tensor = torch.full((B,), seq_len, dtype=torch.long, device=device)

            logits = model(images)         # (B,8,num_chars+1)
            logp   = logits.log_softmax(-1).permute(1, 0, 2)  # (T=8,B,C)

            input_lens = torch.full((B,), seq_len, dtype=torch.long, device=device)
            flat_targets = targets.view(-1)  # (B*8,)

            loss = ctc_loss(logp, flat_targets, input_lens, tgt_lens_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            running_loss += loss.item()
            if step % cfg.get('log_every_n_steps', 50) == 0:
                avg_loss = running_loss / cfg['log_every_n_steps']
                lrs = [group['lr'] for group in optimizer.param_groups]
                elapsed = time.time() - t0
                print(f"[Train] Epoch {ep}/{epochs} | Step {step}/{len(train_loader)} "
                      f"| Avg Loss={avg_loss:.4f} | LR={lrs} | Elapsed={elapsed:.1f}s")
                running_loss = 0.0
                t0 = time.time()

        model.eval()
        total_gt_chars = 0
        total_correct_chars = 0
        total_exact = 0
        total_samples = 0

        with torch.no_grad():
            for idx, (images, targets, _) in enumerate(val_loader):
                images = images.to(device)    # (1,3,32,256)
                logits = model(images)        # (1,8,C)
                logp = logits.log_softmax(-1).permute(1, 0, 2)  # (T=8,B=1,C)

                # Greedy CTC decode (collapse + drop blanks)
                greedy_idx = fixed_length_decode_ctc(logp, blank_idx, seq_len)[0]

                # Naïve 2‐beam search (raw sequences, length=8)
                beams = beam_search_ctc(logp, beam_width=2, blank_idx=blank_idx)[0]

                gt8 = targets[0].tolist()
                for i in range(seq_len):
                    total_gt_chars += 1
                    if greedy_idx[i] == gt8[i]:
                        total_correct_chars += 1
                if greedy_idx == gt8:
                    total_exact += 1
                total_samples += 1

                if idx < 5:
                    gt_str     = "".join(chars[i] for i in gt8)
                    first_str  = "".join(chars[i] for i in greedy_idx)
                    second_str = "".join(chars[i] for i in beams[1])
                    print(f"[VAL] Sample {idx:02d}  GT='{gt_str}'  1st='{first_str}'  2nd='{second_str}'")

        char_acc = total_correct_chars / total_gt_chars if total_gt_chars > 0 else 0.0
        plate_acc = total_exact / total_samples if total_samples > 0 else 0.0
        mark = " ✓" if char_acc > best_char_acc else ""
        print(f"→ Val Char Acc = {char_acc:.4f}{mark}, Val Plate Acc = {plate_acc:.4f}")

        if char_acc > best_char_acc:
            best_char_acc = char_acc
            torch.save(model.state_dict(), save_path)
            print(f"[+] Saved new best checkpoint: {save_path}")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= 5:
            print(f"No improvement in {epochs_since_improve} epochs → stopping early.")
            break

    print("\nFinished training. Best val char‐acc:", best_char_acc)

    # 10) Test set evaluation (just top‐1)
    print("\n[TEST SET EVALUATION]")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    total_gt_chars = 0
    total_correct_chars = 0
    total_exact = 0
    total_samples = 0

    with torch.no_grad():
        for idx, (images, targets, _) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            logp = logits.log_softmax(-1).permute(1, 0, 2)

            greedy_idx = fixed_length_decode_ctc(logp, blank_idx, seq_len)[0]
            gt8 = targets[0].tolist()

            for i in range(seq_len):
                total_gt_chars += 1
                if greedy_idx[i] == gt8[i]:
                    total_correct_chars += 1
            if greedy_idx == gt8:
                total_exact += 1
            total_samples += 1

            if idx < 5:
                gt_str    = "".join(chars[i] for i in gt8)
                first_str = "".join(chars[i] for i in greedy_idx)
                print(f"[TEST] Sample {idx:02d}  GT='{gt_str}'  PRED='{first_str}'")

    char_acc = total_correct_chars / total_gt_chars if total_gt_chars > 0 else 0.0
    plate_acc = total_exact / total_samples if total_samples > 0 else 0.0
    print(f"→ Test Char Acc = {char_acc:.4f}, Test Plate Acc = {plate_acc:.4f}")


if __name__ == "__main__":
    main()
