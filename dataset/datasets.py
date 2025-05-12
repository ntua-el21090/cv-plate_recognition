import json
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Shared image transform
def get_transform(img_size=640):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

class CCPDDetectionDataset(Dataset):
    """
    For object detection: returns full image + YOLO-format bbox
    """
    def __init__(self, annotation_file, transform=None):
        from tqdm import tqdm
        with open(annotation_file, 'r') as f:
            raw_data = json.load(f)
            self.data = [entry for entry in tqdm(raw_data, desc="Loading detection data")]
        self.transform = transform if transform else get_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        print(f"[INFO] Loading image {idx}: {entry['path']}")
        img = cv2.imread(entry['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        x1, y1, x2, y2 = entry['bbox']
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        target = torch.tensor([0, cx, cy, bw, bh], dtype=torch.float32)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, target


class CCPDRecognitionDataset(Dataset):
    """
    For plate recognition: returns cropped plate + text
    """
    def __init__(self, annotation_file, transform=None):
        from tqdm import tqdm
        with open(annotation_file, 'r') as f:
            raw_data = json.load(f)
            self.data = [entry for entry in tqdm(raw_data, desc="Loading recognition data")]
        self.transform = transform if transform else get_transform(128)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img = cv2.imread(entry['path'])
        x1, y1, x2, y2 = entry['bbox']
        plate = entry['plate']

        cropped = img[y1:y2, x1:x2]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        if self.transform:
            cropped = self.transform(image=cropped)['image']

        return cropped, plate