import os
import json
from tqdm import tqdm

# CCPD alphabet used to decode license plates
CCPD_ALPHABET = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-", "_"
]

def decode_plate(label_code: str) -> str:
    try:
       print(f"[DEBUG] Province index: {province_idx}")
       print(f"[DEBUG] Alphabet index: {alphabet_idx}")
       print(f"[DEBUG] ADS indices: {ads_idx}")
       plate = (
          provinces[province_idx[0]] +
          alphabets[alphabet_idx[0]] +
          ''.join(ads[i] for i in ads_idx)
          )

    except Exception as e:
       print(f"[ERROR decoding plate]: {e}")
       plate = "INVALID"
 
def parse_filename(file_name):
    try:
        print(f"\n[DEBUG] Filename: {file_name}")
        parts = file_name.split('-')
        if len(parts) < 7:
            return None

        # Extract bounding box
        x1, y1 = map(int, parts[2].split('_')[0].split('&'))
        x2, y2 = map(int, parts[2].split('_')[1].split('&'))

        # Correctly decode plate from single label field
        label_code = parts[4]
        indices = list(map(int, label_code.split('_')))
        province_idx = indices[0]
        alphabet_idx = indices[1]
        ads_idx = indices[2:]

        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

        try:
            print(f"[DEBUG] Province index: {province_idx}")
            print(f"[DEBUG] Alphabet index: {alphabet_idx}")
            print(f"[DEBUG] ADS indices: {ads_idx}")

            province_part = provinces[province_idx] if 0 <= province_idx < len(provinces) else "?"
            alphabet_part = alphabets[alphabet_idx] if 0 <= alphabet_idx < len(alphabets) else "?"
            ads_part = ''.join(ads[i] if 0 <= i < len(ads) else "?" for i in ads_idx)

            plate = province_part + alphabet_part + ads_part
        except Exception as e:
            print(f"[ERROR decoding plate]: {e}")
            plate = "INVALID"

        print(f"[DEBUG] Decoded plate: {plate}")
        print(f"[DEBUG] All parts: {parts}")

        return {
            'file_name': file_name,
            'bbox': [x1, y1, x2, y2],
            'plate': plate
        }
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_name}: {e}")
        return None

def process_split(split_name):
    folder_path = os.path.join('dataset', split_name)
    data = []

    for file_name in tqdm(os.listdir(folder_path), desc=f"Parsing {split_name}"):
        if not file_name.endswith('.jpg'):
            continue
        parsed = parse_filename(file_name)
        if parsed:
            parsed['path'] = os.path.join(folder_path, file_name)
            data.append(parsed)

    with open(f'dataset/{split_name}.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} samples to dataset/{split_name}.json")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        process_split(split)