#!/bin/bash

python mmutf.py \
--device "cuda" \
--experiment "mmutf" \
--output_dir "experiments/mmutf" \
--do_train \
--do_eval \
--context_text_model "google-t5/t5-base" \
--context_vision_model "openai/clip-vit-base-patch16" \
--context_vision_projection_flag \
--context_vision_projection_size 768 \
--query_text_model "google-t5/t5-base" \
--train_epochs 5 \
--train_epochs_warmup 0 \
--train_learning_rate 0.00003 \
--train_weight_decay 0.001 \
--clip_grad_max_norm 5.0 \
--txt_train_batch_size 32 \
--img_train_batch-size 16 \
--txt_eval_batch_size 32 \
--img_eval_batch-size 32 \
--add_ace_task \
--add_swig_task \
--evaluation_key "ACE" \
--evaluation_metric "argument_cls" \
--evaluation_score "f1" \
--ace_train_file "data/ACE/train_and_dev.json" \
--ace_dev_file "data/ACE/test.json" \
--ace_prompts_file "data/prompts/ace.json" \
--swig_train_file "data/SWIG/train_and_dev.json" \
--swig_dev_file "data/SWIG/test.json" \
--swig_img_dir "data/SWIG/images_512" \
--swig_space_file "data/SWIG/imsitu_space.json" \
--swig_prompts_file "data/prompts/ace.json" \
--m2e2_txt_file "data/M2E2/prepared/test_text.json" \
--m2e2_img_file "data/M2E2/prepared/test_image.json" \
--m2e2_prompts_file "data/prompts/m2e2.json" \
--m2e2_img_objects_file "data/objects/m2e2.pkl" \
--m2e2_txt_triggers_file "data/triggers/reproduced_text_triggers.json" \
--m2e2_img_triggers_file "data/triggers/reproduced_img_triggers.json" \
--m2e2_img_dir "data/M2E2/image/image" \
--swig2ace_mapping_file "data/M2E2/ace_sr_mapping.txt"
# --context_freeze_vision_encoder