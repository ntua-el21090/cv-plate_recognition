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

## YOLOv5 Training Setup (Phase 2)

### Convert Annotations to YOLO Format

6. Convert parsed CCPD data to YOLOv5 format:

```bash
python detection/yolo_label_converter.py
```

This creates YOLO-style `images/` and `labels/` folders under `yolo_dataset/`.

### Clean the Dataset

7. Remove image-label mismatches and invalid files:

```bash
python detection/cleanup.py
```

This ensures only valid `.jpg` images with matching `.txt` labels remain.

---

## Google Colab Setup for Training

To train efficiently using GPU, we provide a Colab notebook located at:
https://colab.research.google.com/drive/1PXdbbHXBq3CkclZEWiC_nRwBm2KXbTkO?usp=drive_link


This notebook:

- Mounts Google Drive
- Clones this GitHub repository
- Installs dependencies
- Trains YOLOv5 using the dataset stored on Drive

### A. Upload the dataset to Google Drive

There is a google drive folder with the appropriate yolo_dataset available on: 
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


### B. Run the notebook

Open `https://colab.research.google.com/drive/1PXdbbHXBq3CkclZEWiC_nRwBm2KXbTkO?usp=drive_link` in Google Colab and run to complete training.
