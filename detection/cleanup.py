import os

splits = ['train', 'val', 'test']
for split in splits:
    image_dir = f"yolo_dataset/images/{split}"
    label_dir = f"yolo_dataset/labels/{split}"

    removed_img = 0
    removed_lbl = 0

    # Remove label files with no matching image
    for label_file in os.listdir(label_dir):
        base = os.path.splitext(label_file)[0]
        img_path = os.path.join(image_dir, base + ".jpg")
        if not os.path.exists(img_path):
            os.remove(os.path.join(label_dir, label_file))
            removed_lbl += 1

    # Remove image files with no matching label
    for img_file in os.listdir(image_dir):
        base = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(label_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            os.remove(os.path.join(image_dir, img_file))
            removed_img += 1

    print(f"[{split}] Removed {removed_lbl} orphan labels, {removed_img} orphan images.")