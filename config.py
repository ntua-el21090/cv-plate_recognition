import torch
import yaml
import os

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
    "cpu"
)

CCPD_ALPHABET = (
    "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学O"
    + "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
)
CCPD_NUM_CLASSES = len(CCPD_ALPHABET) + 1

RECOGNITION_IMAGE_SIZE = (32, 128)  # (h, w)
AUG_PROB = 0.25
NUM_WORKERS = 2
SEED = 42

class Config:
    def __init__(self, yaml_path):
        # Load YAML
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Config YAML file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            params = yaml.safe_load(f)
        for k, v in params.items():
            setattr(self, k, v)

        self.device = getattr(self, "device", DEVICE)
        # Alphabet fallback
        self.alphabet = getattr(self, "alphabet", CCPD_ALPHABET)
        self.num_classes = getattr(self, "num_classes", CCPD_NUM_CLASSES)

        # Handle image size, aliases for h/w.
        # Accept both recognition_image_size and individual img_h/img_w
        if hasattr(self, "recognition_image_size"):
            self.img_h, self.img_w = self.recognition_image_size
        elif hasattr(self, "img_h") and hasattr(self, "img_w"):
            self.recognition_image_size = (self.img_h, self.img_w)
        elif hasattr(self, "img_size") and isinstance(self.img_size, (list, tuple)) and len(self.img_size) == 2:
            self.img_h, self.img_w = self.img_size
            self.recognition_image_size = tuple(self.img_size)
        else:
            self.img_h, self.img_w = RECOGNITION_IMAGE_SIZE
            self.recognition_image_size = RECOGNITION_IMAGE_SIZE

        self.aug_enable = getattr(self, "aug_enable", False)
        self.aug_brightness_contrast = getattr(self, "aug_brightness_contrast", False)
        self.aug_random_rotation = getattr(self, "aug_random_rotation", False)
        self.aug_random_perspective = getattr(self, "aug_random_perspective", False)

        self.num_workers = getattr(self, "num_workers", NUM_WORKERS)

        for subset in ["train", "val", "test"]:
            img_key = f"{subset}_images"
            lbl_key = f"{subset}_labels"
            img_dir_key = f"{subset}_images_dir"
            lbl_dir_key = f"{subset}_labels_dir"
            if hasattr(self, img_key):
                setattr(self, img_dir_key, getattr(self, img_key))
            if hasattr(self, lbl_key):
                setattr(self, lbl_dir_key, getattr(self, lbl_key))
            # Backwards
            if hasattr(self, img_dir_key):
                setattr(self, img_key, getattr(self, img_dir_key))
            if hasattr(self, lbl_dir_key):
                setattr(self, lbl_key, getattr(self, lbl_dir_key))

        # Sequence length alias
        if hasattr(self, "recognition_image_size"):
            self.img_h, self.img_w = self.recognition_image_size
        if hasattr(self, "seq_length"):
            self.seq_len = self.seq_length
        elif not hasattr(self, "seq_len"):
            self.seq_len = 10

        self.batch_size = getattr(self, "batch_size", 32)
        self.epochs = getattr(self, "epochs", 10)
        self.lr = getattr(self, "lr", 0.001)
        self.pct_start = getattr(self, "pct_start", 0.2)
        self.div_factor = getattr(self, "div_factor", 10.0)
        self.final_div = getattr(self, "final_div", 10000.0)
        self.anneal_strategy = getattr(self, "anneal_strategy", "cos")
        self.save_path = getattr(self, "save_path", "checkpoints/best_model.pth")
        self.mean = getattr(self, "mean", [0.485, 0.456, 0.406])
        self.std = getattr(self, "std", [0.229, 0.224, 0.225])

        # Random seed fallback
        self.seed = getattr(self, "seed", SEED)

    def __str__(self):
        return str(self.__dict__)


