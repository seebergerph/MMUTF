import cv2
import math
import torch
import collections
import numpy as np
import albumentations as A


def get_patches_anchor(coordinate, img_size, patch_size):
    [x1, y1, x2, y2] = coordinate
    patch_num = int(img_size / patch_size)

    def get_patch_num(c1, c2):
        num_x = min(math.floor(c1 / patch_size), patch_num - 1)
        num_y = min(math.floor(c2 / patch_size), patch_num - 1) * patch_num
        return num_x + num_y

    x_a = (x1 + x2) / 2
    y_a = (y1 + y2) / 2

    num1 = get_patch_num(x1, y1) + 1
    num2 = get_patch_num(x_a, y_a) + 1 
    num3 = get_patch_num(x2, y2) + 1

    return torch.tensor([num1, num2, num3])


def build_transform_fn(img_size, preprocessor):
    train_fn= A.Compose([
        A.Resize(img_size, img_size, cv2.INTER_CUBIC, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=(0.8, 1), contrast=(0.8, 1), saturation=(0.8, 1), p=0.4),
        A.Normalize(mean=preprocessor.image_mean, std=preprocessor.image_std)
    ], bbox_params=A.BboxParams(format="pascal_voc"))

    test_fn = A.Compose([
        A.Resize(img_size, img_size, cv2.INTER_CUBIC, always_apply=True),
        A.Normalize(mean=preprocessor.image_mean, std=preprocessor.image_std)
    ], bbox_params=A.BboxParams(format="pascal_voc"))
    
    def transform_fn(input, bboxes=[], train=False):
        input = np.asarray(input)
        if train:
            preprocessed = train_fn(image=input, bboxes=bboxes)
        preprocessed = test_fn(image=input, bboxes=bboxes)
        pp_results = {"pixel_values": torch.tensor(preprocessed["image"]).permute((2, 0, 1)).unsqueeze(0)}
        return {"pixel_values": pp_results["pixel_values"], "bboxes": preprocessed["bboxes"]}
    return transform_fn


def create_object_candidates(image, objects, orig_objects, arguments, stoi, img_size, patch_size):
    ground_truth = collections.defaultdict(int)
    for argument in arguments:
        bbox = argument[:4]
        arg_coord = tuple(bbox)
        arg_type = argument[-1]
        ground_truth[arg_coord] = stoi[arg_type]

    candidate_spans = []
    candidate_labels = []
    for object, orig_object in zip(objects, orig_objects):
        bbox = object[:4]
        anchors = get_patches_anchor(bbox, img_size, patch_size)
        candidate_spans.append(tuple(anchors))
        candidate_labels.append(ground_truth[tuple(orig_object[:4])])
    return candidate_spans, candidate_labels