import os
import json
import cv2
from tqdm import tqdm

def convert_bbox_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [0, cx, cy, bw, bh]  # class_id = 0 (license plate)

# Loop over all splits
splits = ['train', 'val', 'test']
for SPLIT in splits:
    print(f"Processing {SPLIT} split...")
    IMG_INPUT_DIR = f'dataset/{SPLIT}'
    OUT_IMG_DIR = f'yolo_dataset/images/{SPLIT}'
    OUT_LABEL_DIR = f'yolo_dataset/labels/{SPLIT}'
    ANNOTATION_FILE = f'dataset/{SPLIT}.json'

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)

    if not os.path.exists(ANNOTATION_FILE):
        print(f"Warning: Annotation file {ANNOTATION_FILE} not found, skipping {SPLIT}.")
        continue

    with open(ANNOTATION_FILE, 'r') as f:
        annotations = json.load(f)

    log_file = os.path.join(f'detection/skipped_{SPLIT}.log')
    skipped = []

    for entry in tqdm(annotations, desc=f"Converting {SPLIT} set"):
        img_path = entry['path']
        bbox = entry['bbox']
        img_name = os.path.basename(img_path)
        label_name = img_name.replace('.jpg', '.txt')

        # Validate bbox
        if not bbox or len(bbox) != 4:
            skipped.append((img_path, 'Invalid bbox'))
            continue

        # Read and validate image
        img = cv2.imread(img_path)
        if img is None:
            skipped.append((img_path, 'Image not readable'))
            continue

        img_h, img_w = img.shape[:2]
        out_img_path = os.path.join(OUT_IMG_DIR, img_name)
        cv2.imwrite(out_img_path, img)

        # Convert bbox and write label file
        yolo_bbox = convert_bbox_to_yolo(bbox, img_w, img_h)
        label_path = os.path.join(OUT_LABEL_DIR, label_name)
        with open(label_path, 'w') as f_label:
            f_label.write(' '.join([str(round(x, 6)) for x in yolo_bbox]))

    # Save log of skipped files
    if skipped:
        with open(log_file, 'w') as logf:
            for path, reason in skipped:
                logf.write(f"{path} - {reason}\n")
        print(f"Skipped {len(skipped)} entries. Details logged in {log_file}")