import os
import argparse
import json
import cv2
from tqdm import tqdm

def crop_manual_from_annotations_main(annotations_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for annot_file in os.listdir(annotations_dir):
        if annot_file.endswith(".json"):
            with open(os.path.join(annotations_dir, annot_file)) as f:
                data = json.load(f)
            img_path = data['image_path']
            coords = data['bbox']
            img = cv2.imread(img_path)
            crop = img[coords[1]:coords[3], coords[0]:coords[2]]
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, crop)

def crop_detected_plates_main(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            img_file = os.path.splitext(label_file)[0] + ".jpg"
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    cls, cx, cy, bw, bh = parts[:5] 
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    crop = img[y1:y2, x1:x2]
                    out_path = os.path.join(output_dir, img_file)
                    cv2.imwrite(out_path, crop)

def crop_detected_plates_fix_main(image_dir, label_dir, output_dir, failed_list_path, fix_list_path):
    os.makedirs(output_dir, exist_ok=True)
    with open(failed_list_path) as f:
        failed_images = set(line.strip() for line in f)
    with open(fix_list_path) as f:
        fix_images = set(line.strip() for line in f)
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            img_file = os.path.splitext(label_file)[0] + ".jpg"
            if img_file not in fix_images or img_file in failed_images:
                continue
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.strip().split())
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    crop = img[y1:y2, x1:x2]
                    out_path = os.path.join(output_dir, img_file)
                    cv2.imwrite(out_path, crop)


def crop_ccpd_test_plates_main(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(os.listdir(image_dir)):
        if file.endswith(".jpg"):
            parts = file.split("-")[2].split("_")[0].split("&")
            x1, y1, x2, y2 = map(int, parts)
            img = cv2.imread(os.path.join(image_dir, file))
            crop = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(output_dir, file), crop)


def main():
    parser = argparse.ArgumentParser(description="Unified Plate Cropping Utility")
    parser.add_argument('--method', type=str, required=True, choices=['manual', 'detected', 'detected_fix', 'ccpd_test'],
                        help="Which cropping method to use.")
    parser.add_argument('--input', type=str, required=True, help="Path to input images/annotations.")
    parser.add_argument('--output', type=str, required=True, help="Directory to save cropped plates.")
    parser.add_argument('--labels', type=str, default=None, help="(Optional) Path to YOLO labels if needed.")
    parser.add_argument('--detections_txt', type=str, default=None, help="(Optional) Path to yolo_no_detections.txt if needed.")
    parser.add_argument('--filter_failures', type=str, default=None, help="(Optional) fix_failures_no_detection.txt path")

    args = parser.parse_args()

    if args.method == 'manual':
        crop_manual_from_annotations_main(args.input, args.output)

    elif args.method == 'detected':
        if not args.labels:
            raise ValueError("--labels is required for detected method")
        crop_detected_plates_main(args.input, args.labels, args.output)

    elif args.method == 'detected_fix':
        if not args.labels or not args.detections_txt or not args.filter_failures:
            raise ValueError("--labels, --detections_txt and --filter_failures are required for detected_fix method")
        crop_detected_plates_fix_main(args.input, args.labels, args.output, args.detections_txt, args.filter_failures)

    elif args.method == 'ccpd_test':
        crop_ccpd_test_plates_main(args.input, args.output)

    print("Done.")


if __name__ == '__main__':
    main()