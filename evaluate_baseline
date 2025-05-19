import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from dataset.datasets import CCPDRecognitionDataset as PlateDataset
from models.baseline_model import CRNN
from config import *
import json
from tqdm import tqdm

char_set = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-", "_"
]
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

def greedy_decode(logits):
    probs = log_softmax(logits, dim=2)
    max_indices = torch.argmax(probs, dim=2)
    max_indices = max_indices.permute(1, 0)  
    decoded_strings = []
    for indices in max_indices:
        decoded = []
        prev_idx = -1
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != len(char_set):  
                decoded.append(idx_to_char.get(idx, ""))
            prev_idx = idx
        decoded_strings.append("".join(decoded))
    return decoded_strings

def evaluate(model_path, json_path, image_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PlateDataset(json_path, image_root=image_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CRNN(n_classes=len(char_set) + 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            decoded = greedy_decode(logits)
            print(f"GT: {labels[0]}, Pred: {decoded[0]}")
            if decoded[0] == labels[0]:
                correct += 1
            total += 1

    print(f"Plate-Level Accuracy: {correct}/{total} = {correct / total:.4f}")

if __name__ == "__main__":
    evaluate(
        model_path="/content/drive/MyDrive/cv_plate_recognition/crnn_best.pth",
        json_path="dataset/val.json",
        image_root="/content/drive/MyDrive/cv_plate_recognition/ccpd_dataset"
    )