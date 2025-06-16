import os
import json
from tqdm import tqdm

def parse_filename(filename):
    name = filename[:-4]
    parts = name.split('-')

    plate = parts[0]
    coords = parts[2].split('_')
    points = [list(map(int, pt.split('&'))) for pt in coords]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox = [min(xs), min(ys), max(xs), max(ys)]

    return {
        'filename': filename,
        'plate': plate,
        'points': points,
        'bbox': bbox
    }

def parse_subset(folder_path, subset_name):
    full_path = os.path.join(folder_path, subset_name)
    annotations = []

    for fname in tqdm(os.listdir(full_path)):
        if not fname.endswith('.jpg'):
            continue
        img_path = os.path.join(full_path, fname)
        if not os.path.exists(img_path):
            continue
        try:
            parsed = parse_filename(fname)
            parsed['path'] = os.path.relpath(img_path, start='.')
            annotations.append(parsed)
        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")

    os.makedirs("ccpd_annotations", exist_ok=True)
    out_file = f"ccpd_annotations/{subset_name}_annotations.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(annotations)} annotations to {out_file}")

def main():
    print("STARTING")
    root = "ccpd_dataset"
    for subset in ['train', 'val', 'test']:
        parse_subset(root, subset)

if __name__ == "__main__":
    main()