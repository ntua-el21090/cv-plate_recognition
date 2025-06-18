# CV Project 4 — License Plate Detection & Recognition

This project implements a two-stage pipeline for license plate detection and recognition using the CCPD dataset. It uses YOLOv5 for detection and a CRNN-CTC model for recognition, and includes a baseline for comparison.

---

## Dataset Setup

This project uses the [CCPD dataset](https://github.com/detectRecog/CCPD).

### Environment Setup

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset Preparation

### 1. Download & Structure

Download the CCPD dataset and extract it to the following structure:

```
ccpd_dataset/
├── train/
├── val/
├── test/
```

Each subfolder must contain `.jpg` images only.

---

### 2. Parse the CCPD Annotations

```bash
python parse_ccpd_green.py
```

This script converts raw CCPD filenames into JSON annotations:
- `ccpd_annotations/train_annotations.json`
- `ccpd_annotations/val_annotations.json`
- `ccpd_annotations/test_annotations.json`

---

### 3. Build YOLO Dataset (Detection)

Convert JSON annotations to YOLO format for training:

```bash
python build_yolo_dataset.py \
    --ann_dir ccpd_annotations \
    --images_dir ccpd_dataset \
    --yolo_dir yolo_dataset \
    --link_images        
```

It creates a directory:

```
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
```

---

### 4. Train YOLOv5 Detection Model

First, clone the yolov5 repository: https://github.com/ultralytics/yolov5 in the root of the project.

```bash
git clone https://github.com/ultralytics/yolov5
```
Then proceed to train the model:

```bash
cd yolov5
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data ../ccpd_data.yaml \
  --weights yolov5s.pt \
  --project ../checkpoints/yolo \
  --name plate_detector

```

---

## Plate Cropping

### 1. Crop Detected Plates and Ground Truth
run this for train,test,val:

```bash
python yolov5/detect.py \
  --weights yolov5/runs/*test or train or val*/ccpd-plates3/weights/best.pt \
  --source ccpd_dataset/*test or train or val* \
  --img 640 \
  --device mps \
  --save-txt --save-conf \
  --nosave \
  --project runs/detect \
  --name ccpd_*test or train or val*_detect \
  --exist-ok
```
And after, run this also for train,test,val:

```bash
python crop_plates.py \
  --method detected \
  --input ccpd_dataset/*test or train or val* \
  --labels runs/detect/ccpd_*test or train or val*_detect/labels \
  --output cropped_plates_detected/*test or train or val*          

```

This will generate cropped plates and label files in:

```
cropped_plates_detected/
├── train/
├── val/
├── test/
```

---

### 2. Build CRNN Training Dataset

```bash
python build_recognition_dataset.py
```

This creates cropped plate images and `.txt` labels for each set (train, val, test).

---


## Recognition Models

### 1. Train Baseline CTC (for comparison)

```bash
python baseline_ctc.py --config config.yaml
```

Used for benchmarking performance against the CRNN model.

---

### 2. Train PDLPR Model

```bash
python train_pdlpr.py --config config.yaml
```

This trains the CRNN-CTC model on the dataset built in the previous step, using config-controlled augmentation and evaluation.

---

### 3. Train Final Model

```bash
python train_plate_classifier.py --config config.yaml
```

This will add to the directrory PDLPR/checkpoints/ the best model which we are going to use for evaluation.
---

## End-to-End Evaluation

### Evaluate the Entire Pipeline

export PYTHONPATH=$(pwd):$(pwd)/yolov5

```bash
python evaluate_test.py \
  --mode eval \
  --config config.yaml \
  --checkpoint PDLPR/checkpoints/best_model.pth \
  --show_samples
```
---
### Evaluate on your Images

```bash
python evaluate_test.py \
  --mode infer \
  --config config.yaml \
  --checkpoint plate_classifier/checkpoints/best_model.pth \
  --det-weights yolov5/runs/train/ccpd-plates3/weights/best.pt \
  --source your_image_directory
```
---
## Evaluation Metrics

- **Detection:** IOU > 0.7 for plate match.
- **Recognition:** Full 8-character plate string match.
- Outputs character-level and plate-level accuracy.

---

## Project Structure

```
.
├── yolov5/                   # YOLOv5 source
├── PDLPR/                    # CRNN model and support code
├── utils/                    # Utility and data tools
├── dataset/                  # Raw CCPD images
├── yolo_dataset/             # Converted YOLO format data
├── cropped_plates_detected/  # Cropped license plates
├── config.yaml
├── build_yolo_dataset.py
├── build_recognition_dataset.py
├── crop_plates.py
├── dataset_tools.py
├── train_pdlpr.py
├── baseline_ctc.py
├── evaluate_test.py
├── parse_ccpd_green.py
├── requirements.txt
├── README.md
```

---

## Notes

- The YOLOv5 and CRNN training can be performed locally or using Google Colab with slight adjustments to paths.
- The configuration is driven through `config.yaml` for consistent parameters.

## Sources

1. Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). A Real-Time License Plate Detection and Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791 
2. Xu, Z.; Yang, W.; Meng, A.; Lu, N.; Huang, H.; Ying, C.; Huang, L. Towards end-to-end license plate detection and recognition: A large dataset and baseline. In Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 8–14 September 2018. 
3. R. K. Prajapati, Y. Bhardwaj, R. K. Jain and D. Kamal Kant Hiran, ”A Review Paper on Automatic Num ber Plate Recognition using Machine Learning : An In-Depth Analysis of Machine Learning Techniques in Automatic Number Plate Recognition: Opportunities and Limitations,” 2023 International Conference on Computational Intelligence, Communication Technology and Networking (CICTN), Ghaziabad, India, 2023, pp. 527-532 
4. https://github.com/detectRecog/CCPD
5. https://github.com/ultralytics/yolov5
