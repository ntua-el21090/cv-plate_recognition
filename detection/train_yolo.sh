#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# OPTIONAL: Run dataset cleanup to remove bad files before training
python3 detection/cleanup.py

# Navigate to YOLOv5 directory
cd yolov5

# Launch training
python3 train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --data ../detection/ccpd.yaml \
  --weights yolov5s.pt \
  --name ccpd_yolov5 \
  --workers 4
  