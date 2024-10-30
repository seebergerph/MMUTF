import os
import cv2
import pickle
import argparse
import collections
import PIL as pillow
from ultralytics import YOLO


def main(args):
    model = YOLO(args.yolo_model)
    results = collections.defaultdict(list)

    for idx, image_file in enumerate(os.listdir(args.image_dir)):
        image_path = os.path.join(args.image_dir, image_file)

        image = pillow.Image.open(image_path)
        preds = model.predict(source=image)

        bboxs = preds[0].boxes.xyxy.cpu().numpy().tolist()
        confs = preds[0].boxes.conf.cpu().numpy().tolist()
        classes = preds[0].boxes.cls.cpu().numpy().tolist()
        classes = [preds[0].names[p] for p in classes]

        candidates = [
            {"bbox": bbox,"class": cls,"conf": conf} 
            for bbox, cls, conf in zip(bboxs, classes, confs)
            if conf >= args.threshold
        ]

        results[image_file] = candidates

        if args.plots_dir:
            if idx % args.plots_mod == 0:
                os.makedirs(args.plots_dir, exist_ok=True)
                plot_file = os.path.join(args.plots_dir, image_file)
                cv2.imwrite(plot_file, preds[0].plot())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)
    with open(output_file, "wb") as fs:
        pickle.dump(results, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--yolo_model", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--plots_dir", type=str, required=False)
    parser.add_argument("--plots_mod", type=int, default=10)
    args = parser.parse_args()
    main(args)