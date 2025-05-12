<<<<<<< Updated upstream
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.datasets import CCPDDetectionDataset, CCPDRecognitionDataset
import torchvision

def show_detection_batch():
=======
import cv2
print(">>> Starting test_loader.py")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.datasets import CCPDDetectionDataset
from dataset.datasets import CCPDRecognitionDataset
import torchvision.transforms.functional as F
import numpy as np

def unnormalize(img_tensor):
    # Undo normalization from Albumentations
    img_tensor = img_tensor * 0.5 + 0.5
    return img_tensor

def show_bboxes_on_image(img_tensor, bbox_tensor):
    img = unnormalize(img_tensor).permute(1, 2, 0).numpy() * 255
    img = img.astype(np.uint8).copy()
    h, w, _ = img.shape

    class_id, cx, cy, bw, bh = bbox_tensor.numpy()
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def show_detection_bboxes():
>>>>>>> Stashed changes
    dataset = CCPDDetectionDataset("dataset/train.json")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for imgs, targets in loader:
<<<<<<< Updated upstream
        img_grid = torchvision.utils.make_grid(imgs, nrow=2)
        plt.imshow(img_grid.permute(1, 2, 0).numpy())
        plt.title("Detection Sample Batch (YOLO Format)")
        plt.axis('off')
        plt.show()
        break

def show_recognition_samples():
    dataset = CCPDRecognitionDataset("dataset/train.json")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for crops, plates in loader:
        for i in range(len(crops)):
            img = crops[i].permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.title(f"Plate: {plates[i]}")
=======
        for i in range(len(imgs)):
            img = show_bboxes_on_image(imgs[i], targets[i])
            plt.imshow(img)
            plt.title("Detection Sample with BBox")
>>>>>>> Stashed changes
            plt.axis('off')
            plt.show()
        break

<<<<<<< Updated upstream
if __name__ == "__main__":
    print("🔍 Showing sample detection images...")
    show_detection_batch()

    print("🔍 Showing cropped license plate images and text...")
    show_recognition_samples()
=======
def show_recognition_side_by_side():
    det_dataset = CCPDDetectionDataset("dataset/train.json")
    rec_dataset = CCPDRecognitionDataset("dataset/train.json")

    for i in range(4):  # Show 4 samples
        entry = det_dataset.data[i]
        img = cv2.imread(entry['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = entry['bbox']
        plate = entry['plate']

        # Draw bounding box
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop plate
        plate_crop = img[y1:y2, x1:x2]

        # Plot side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(img_with_box)
        axs[0].set_title("Full Image with BBox")
        axs[0].axis('off')

        axs[1].imshow(plate_crop)
        print("Plate:", plate)
        axs[1].set_title("Cropped Plate")
        axs[1].axis('off')
        print(f"Decoded plate string: {plate}")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    show_detection_bboxes()
    show_recognition_side_by_side()
>>>>>>> Stashed changes
