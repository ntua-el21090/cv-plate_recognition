import os
from pathlib import Path

# CCPD alphabet mappings
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def decode_plate_from_filename(filename):
    try:
        stem = Path(filename).stem
        index_str = stem.split('-')[-3]
        indices = list(map(int, index_str.split('_')))
        assert len(indices) == 8
        plate = (
            provinces[indices[0]] +
            alphabets[indices[1]] +
            ''.join(ads[i] for i in indices[2:])
        )
        return plate
    except Exception as e:
        print(f"[WARN] Cannot decode {filename}: {e}")
        return None

def build_labels(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    for img_path in Path(image_dir).glob("*.jpg"):
        plate = decode_plate_from_filename(img_path.name)
        if plate:
            lbl_path = label_dir / (img_path.stem + ".txt")
            lbl_path.write_text(plate, encoding="utf-8")

def main():
    base = Path("cropped_plates_detected")
    splits = {
        "train": base / "train" ,
        "val": base / "val" ,
        "test": base / "test" 
    }
    for split, img_dir in splits.items():
        label_dir = base / split / "labels_text"
        print(f"\nProcessing split: {split}")
        print(f" → Looking for images in: {img_dir}")
        print(f" → Saving labels to: {label_dir}")
        build_labels(img_dir, label_dir)
    print("\nAll labels generated from filenames.")

if __name__ == "__main__":
    main()
