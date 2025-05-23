## Dataset Setup

This project uses the [CCPD dataset](https://github.com/detectRecog/CCPD).

### Environment Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

### Dataset Download and Structure

3. Download the dataset from the official source (e.g., Google Drive or Baidu link).

4. Extract and organize it as follows:

```
dataset/
├── train/
├── val/
├── test/
```

Each folder should contain CCPD `.jpg` images.

### Parse Annotations

5. Run the dataset parser to generate annotation files:

```bash
python dataset/parse_ccpd.py
```

This will create:
- `dataset/train.json`
- `dataset/val.json`
- `dataset/test.json`

## Google Colab Setup for Training

To train efficiently using GPU, we provide a Colab notebook located at:
https://colab.research.google.com/drive/1PXdbbHXBq3CkclZEWiC_nRwBm2KXbTkO?usp=drive_link


This notebook:

- Mounts Google Drive
- Clones this GitHub repository
- Installs dependencies
- Trains Baseline model and YOLOv5 using the dataset stored on Drive

### A. Upload the dataset to Google Drive

There is a google drive folder with the appropriate ccpd_dataset and yolo_dataset available on: 
https://drive.google.com/drive/folders/1bfURR4furfjVZu4IqXHxOJQ41b3m0hhc?usp=drive_link


If you want to use your own generated yolo_dataset:

Upload the entire `yolo_dataset/` folder to:

```
/MyDrive/cv_plate_recognition/ccpd_data/yolo_dataset/
```

It should contain:

```
yolo_dataset/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
```

```
ccpd_dataset/
├── train/
├── val/
├── test/
```
## Baseline Training and Evaluation (Phase 2)

### Train the Baseline CRNN Model

6. Run the baseline training script to train a CRNN model with CTC loss:

```bash
python train_baseline.py
```

This will:
- Load the dataset and annotations from `dataset/train.json` and `dataset/val.json`
- Train the model using a ResNet feature extractor + BiLSTM + CTC loss
- Save the best-performing model to:

```
/MyDrive/cv_plate_recognition/crnn_best.pth
```

### Evaluate the Baseline Model

7. Run the evaluation script to compute plate-level recognition accuracy:

```bash
python evaluate_baseline.py
```

Make sure your `evaluate_baseline.py` uses the correct model and paths:
```python
evaluate(
    model_path="/content/drive/MyDrive/cv_plate_recognition/crnn_best.pth",
    json_path="dataset/val.json",
    image_root="/content/drive/MyDrive/cv_plate_recognition/ccpd_dataset"
)
```

This will output the number of license plates predicted with 100% character accuracy.

---


## YOLOv5 Training Setup (Phase 3)

### Convert Annotations to YOLO Format

8. Convert parsed CCPD data to YOLOv5 format:

```bash
python detection/yolo_label_converter.py
```

This creates YOLO-style `images/` and `labels/` folders under `yolo_dataset/`.

### Clean the Dataset

9. Remove image-label mismatches and invalid files:

```bash
python detection/cleanup.py
```

This ensures only valid `.jpg` images with matching `.txt` labels remain.

### Run the notebook

Open `https://colab.research.google.com/drive/1PXdbbHXBq3CkclZEWiC_nRwBm2KXbTkO?usp=drive_link` in Google Colab and run to complete training.
