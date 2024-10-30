#!/bin/bash

DATA_DIR="data/M2E2/annotations"

python -m utils.m2e2.prepare \
 --text_only_file $DATA_DIR/text_only_event.json \
 --text_multi_file $DATA_DIR/text_multimedia_event.json \
 --image_only_file $DATA_DIR/image_only_event.json \
 --image_multi_file $DATA_DIR/image_multimedia_event.json \
 --text_out_file test_text.json \
 --image_out_file test_image.json \
 --image_dir $DATA_DIR/../image/image \
 --output_dir data/M2E2/prepared