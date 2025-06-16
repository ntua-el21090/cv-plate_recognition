import os
import json
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


SPLITS = ["train", "val", "test"]

def build_yolo_labels(ann_dir, yolo_dir):
    print("[1/3] Building YOLO label files …")
    for split in SPLITS:
        ann_file = Path(ann_dir) / f"{split}_annotations.json"
        out_dir = Path(yolo_dir) / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(ann_file) as f:
            anns = json.load(f)

        print(f"[+] Converting {len(anns):,} annotations → {out_dir}")

        for ann in tqdm(anns, desc=f"[{split}]"):
            name = ann["filename"]
            x1, y1, x2, y2 = ann["bbox"]
            # CCPD images are all 1160×720, so we can skip opening the file
            w, h = 1160, 720

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            label_file = out_dir / f"{name.replace('.jpg', '.txt')}"
            with open(label_file, 'a') as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

def copy_images(images_dir, yolo_dir, link=False):
    print("[2/3] Copying or linking images …")
    for split in SPLITS:
        src_dir = Path(images_dir) / split
        dst_dir = Path(yolo_dir) / "images" / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[WARNING] Missing split directory: {src_dir}")
            continue

        for file in os.listdir(src_dir):
            if not file.endswith(".jpg"):
                continue
            src = src_dir / file
            dst = dst_dir / file
            if link:
                if dst.exists():
                    continue
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)

def clean_dataset(yolo_dir):
    print("[3/3] Cleaning corrupted or unmatched image/label files …")
    for split in SPLITS:
        img_dir = Path(yolo_dir) / "images" / split
        lbl_dir = Path(yolo_dir) / "labels" / split

        for file in os.listdir(img_dir):
            if not file.endswith(".jpg"): continue
            img_path = img_dir / file
            lbl_path = lbl_dir / file.replace(".jpg", ".txt")

            if not img_path.exists() or not lbl_path.exists() or lbl_path.stat().st_size == 0:
                img_path.unlink(missing_ok=True)
                lbl_path.unlink(missing_ok=True)
    print("[DONE] YOLO dataset ready")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', required=True, help='ccpd_annotations/')
    parser.add_argument('--images_dir', required=True, help='ccpd_dataset/')
    parser.add_argument('--yolo_dir', required=True, help='Output folder yolo_dataset/')
    parser.add_argument('--link_images', action='store_true', help='Use symlinks instead of copying')
    args = parser.parse_args()

    build_yolo_labels(args.ann_dir, args.yolo_dir)
    copy_images(args.images_dir, args.yolo_dir, link=args.link_images)
    clean_dataset(args.yolo_dir)

if __name__ == '__main__':
    main()