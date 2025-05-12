# config.py

# === General Config ===
SEED = 42

# === File Paths ===
TRAIN_JSON = "dataset/train.json"
VAL_JSON = "dataset/val.json"
TEST_JSON = "dataset/test.json"

# === Image Settings ===
DETECTION_IMAGE_SIZE = 640        # For YOLOv5 training
RECOGNITION_IMAGE_SIZE = (128, 384)  # For plate crops (H, W)

# === Training Parameters ===
BATCH_SIZE = 16
NUM_WORKERS = 4

# === Classes ===
NUM_CLASSES = 1   # Only 1 class: 'license_plate'

# === CCPD Alphabet Mapping ===
CCPD_ALPHABET = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]