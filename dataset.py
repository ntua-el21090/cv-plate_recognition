import os
import glob
import warnings
import yaml
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def load_alphabet(path):
    with open(path, 'r', encoding='utf-8') as f:
        chars = list(f.read().strip())
    return chars

class PlateDataset(Dataset):


    def __init__(self, images_dir, labels_dir, seq_len, alphabet_path,
                 augment=False, aug_brightness_contrast=False,
                 aug_random_rotation=False, aug_random_perspective=False,
                 config_path="config.yaml"):
        super().__init__()

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.seq_len = seq_len

        self.chars = load_alphabet(alphabet_path)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.num_chars = len(self.chars)
        self.blank_idx = self.num_chars 

        self.augment = augment
        self.aug_brightness_contrast = aug_brightness_contrast
        self.aug_random_rotation = aug_random_rotation
        self.aug_random_perspective = aug_random_perspective

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if "recognition_image_size" not in cfg:
            raise KeyError("'recognition_image_size' not found in config.yaml")
        img_h, img_w = cfg["recognition_image_size"]

        if "mean" not in cfg or "std" not in cfg:
            raise KeyError("'mean' or 'std' missing in config.yaml")
        mean = tuple(cfg["mean"])
        std  = tuple(cfg["std"])

        self.img_h = img_h
        self.img_w = img_w
        self.mean = mean
        self.std  = std

        all_jpgs = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))
        if not all_jpgs:
            raise RuntimeError(f"No .jpg files found in {self.images_dir!r}.")
        self.samples = []
        for img_path in all_jpgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.labels_dir, f"{stem}.txt")
            if not os.path.isfile(txt_path):
                warnings.warn(f"[Warning] skipping '{os.path.basename(img_path)}' because '{txt_path}' not found.")
                continue

            with open(txt_path, "r", encoding="utf-8") as tf:
                plate_str = tf.readline().strip()
            self.samples.append((img_path, plate_str))

        if not self.samples:
            raise RuntimeError(f"No valid image/label pairs found in {self.images_dir!r} ↔ {self.labels_dir!r}.")

        self._build_transforms()

    def _build_transforms(self):

        tf_list = []

        if self.augment:
            if self.aug_brightness_contrast:
                tf_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
            if self.aug_random_rotation:
                tf_list.append(transforms.RandomRotation(degrees=5, fill=(0, 0, 0)))
            if self.aug_random_perspective:
                tf_list.append(transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=0))

        print(f"[DEBUG] Dataset will resize to: (img_h={self.img_h}, img_w={self.img_w})")
        tf_list.append(transforms.Resize((self.img_h, self.img_w), interpolation=Image.BILINEAR))

        tf_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]

        self.transforms = transforms.Compose(tf_list)

    def __len__(self):
        return len(self.samples)

    def _encode_label(self, plate_str):
        encoded = []
        for ch in plate_str:
            if ch not in self.char_to_idx:
                raise ValueError(f"Character '{ch}' not in vocabulary.")
            idx = self.char_to_idx[ch]
            
            if idx >= self.blank_idx:
                raise ValueError(f"Character '{ch}' maps to index {idx}, but blank_idx={self.blank_idx}.")
            encoded.append(idx)

        if len(encoded) > self.seq_len:
            encoded = encoded[: self.seq_len]
        while len(encoded) < self.seq_len:
         
            encoded.append(self.blank_idx)

        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        img_path, plate_str = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        x_tensor = self.transforms(img)

        label_indices = self._encode_label(plate_str)

        return x_tensor, label_indices
