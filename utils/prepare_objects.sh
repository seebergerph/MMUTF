#!/bin/bash

DATA_DIR="data/M2E2/image/image"

python -m utils.objects.detection \
--image_dir $DATA_DIR \
--yolo_model models/yolov8l.pt \
--threshold 0. \
--output_dir data/objects \
--output_file m2e2.pkl