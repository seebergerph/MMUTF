import os
import sys
import json
import tqdm
import time
import wandb
import pickle
import logging
import datetime
import argparse

import src.eval_image as eval_image
import src.eval_text as eval_text

from src.utils_data import prepare_swig2ace_mapping
from src.datasets import ACEDatasetEAE
from src.datasets import SwigDatasetEAE
from src.datasets import M2E2TextDatasetED
from src.datasets import M2E2ImageDatasetED
from src.datasets import M2E2TextDatasetEAE
from src.datasets import M2E2ImageDatasetEAE
from src.trainer import TrainerJoint

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    # Context model
    parser.add_argument("--context_text_model", type=str)
    parser.add_argument("--context_vision_model", type=str)
    parser.add_argument("--context_vision_projection_flag", action="store_true")
    parser.add_argument("--context_vision_projection_bias", action="store_true")
    parser.add_argument("--context_vision_projection_size", type=int)
    parser.add_argument("--context_freeze_text_encoder", action="store_true")
    parser.add_argument("--context_freeze_vision_encoder", action="store_true")
    parser.add_argument("--context_freeze_text_all", action="store_true")
    parser.add_argument("--context_freeze_vision_all", action="store_true")

    # Query model
    parser.add_argument("--query_text_model", type=str)
    parser.add_argument("--query_freeze_text_encoder", action="store_true")
    parser.add_argument("--query_freeze_text_all", action="store_true")

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--train_epochs_warmup", type=int, default=0)
    parser.add_argument("--train_learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_weight_decay", type=float, default=1e-3)
    parser.add_argument("--clip_grad_max_norm", type=float, default=5.0)
    parser.add_argument("--txt_train_batch_size", type=int, default=12)
    parser.add_argument("--img_train_batch-size", type=int, default=12)
    parser.add_argument("--txt_eval_batch_size", type=int, default=12)
    parser.add_argument("--img_eval_batch-size", type=int, default=12)
    parser.add_argument("--add_ace_task", action="store_true")
    parser.add_argument("--add_swig_task", action="store_true")

    # Evaluation
    parser.add_argument("--evaluation_key", type=str, default="ACE")
    parser.add_argument("--evaluation_metric", type=str, default="argument_cls")
    parser.add_argument("--evaluation_score", type=str, default="f1")

    # Training data
    parser.add_argument("--ace_train_file", type=str, required=True)
    parser.add_argument("--ace_dev_file", type=str, required=True)
    parser.add_argument("--ace_prompts_file", type=str, required=True)
    parser.add_argument("--swig_train_file", type=str, required=True)
    parser.add_argument("--swig_dev_file", type=str, required=True)
    parser.add_argument("--swig_img_dir", type=str, required=True)
    parser.add_argument("--swig_space_file", type=str, required=True)
    parser.add_argument("--swig_prompts_file", type=str, required=True)

    # Testing data
    parser.add_argument("--m2e2_txt_file", type=str, required=True)
    parser.add_argument("--m2e2_img_file", type=str, required=True)
    parser.add_argument("--m2e2_prompts_file", type=str, required=True)
    parser.add_argument("--m2e2_img_objects_file", type=str, required=True)
    parser.add_argument("--m2e2_txt_triggers_file", type=str, required=True)
    parser.add_argument("--m2e2_img_triggers_file", type=str, required=True)
    parser.add_argument("--m2e2_img_dir", type=str, required=True)
    parser.add_argument("--swig2ace_mapping_file", type=str, required=True)
    return parser.parse_args()


def prepare_pred_datasets(args):
    # Text Predicted Triggers
    with open(args.m2e2_txt_triggers_file, "r", encoding="utf-8") as fs:
        lines = fs.readlines()
    m2e2_txt_pred_triggers = [json.loads(line) for line in lines]

    # Image Predicted Triggers
    with open(args.m2e2_img_triggers_file, "r", encoding="utf-8") as fs:
        lines = fs.readlines()
    m2e2_img_pred_triggers = [json.loads(line) for line in lines]

    # Image Predicted Objects
    with open(args.m2e2_img_objects_file, "rb") as fs: 
        m2e2_img_pred_objects = pickle.load(fs)

    return {
        "m2e2_txt_pred_triggers": m2e2_txt_pred_triggers,
        "m2e2_img_pred_triggers": m2e2_img_pred_triggers,
        "m2e2_img_pred_objects": m2e2_img_pred_objects
    }


def prepare_ed_datasets(args):
    # Text Event Detection
    m2e2_txt_ed_dataset = M2E2TextDatasetED(
        path=args.m2e2_txt_file
    )

    # Image Event Detection
    m2e2_img_ed_dataset = M2E2ImageDatasetED(
        path=args.m2e2_img_file, 
        image_dir=args.m2e2_img_dir
    )

    return {
        "m2e2_txt_ed_dataset": m2e2_txt_ed_dataset, 
        "m2e2_img_ed_dataset": m2e2_img_ed_dataset
    }


def prepare_eae_datasets(args, preds_dict):
    # Text Event Argument Extraction (Gold Triggers)
    m2e2_txt_eae_gt_dataset = M2E2TextDatasetEAE(
        path=args.m2e2_txt_file,
        prompts_path=args.m2e2_prompts_file,
        events=None
    )

    # Text Event Argument Extraction (Pred Triggers)
    m2e2_txt_eae_pt_dataset = M2E2TextDatasetEAE(
        path=args.m2e2_txt_file,
        prompts_path=args.m2e2_prompts_file,
        events=preds_dict["m2e2_txt_pred_triggers"]
    )

    # Image Event Argument Extraction (Gold Triggers)
    m2e2_img_eae_gt_dataset = M2E2ImageDatasetEAE(
        path=args.m2e2_img_file, 
        image_dir=args.m2e2_img_dir,
        prompts_path=args.m2e2_prompts_file,
        objects=preds_dict["m2e2_img_pred_objects"],
        events=None
    )

    # Image Event Argument Extraction (Pred Triggers)
    m2e2_img_eae_pt_dataset = M2E2ImageDatasetEAE(
        path=args.m2e2_img_file, 
        image_dir=args.m2e2_img_dir,
        prompts_path=args.m2e2_prompts_file,
        objects=preds_dict["m2e2_img_pred_objects"],
        events=preds_dict["m2e2_img_pred_triggers"]
    )

    return {
        "m2e2_txt_eae_gt_dataset": m2e2_txt_eae_gt_dataset,
        "m2e2_img_eae_gt_dataset": m2e2_img_eae_gt_dataset,
        "m2e2_txt_eae_pt_dataset": m2e2_txt_eae_pt_dataset,
        "m2e2_img_eae_pt_dataset": m2e2_img_eae_pt_dataset
    }


def prepare_train_datasets(args, m2e2_type_set, swig2ace_mapping):
    # ACE train dataset
    train_txt_dataset = ACEDatasetEAE(
        path=args.ace_train_file, 
        filter_type_set=m2e2_type_set,
        prompts_path=args.ace_prompts_file
    )

    # ACE development dataset
    dev_txt_dataset = ACEDatasetEAE(
        path=args.ace_dev_file, 
        filter_type_set=m2e2_type_set,
        prompts_path=args.ace_prompts_file
    )

    # SWiG train dataset
    train_img_dataset = SwigDatasetEAE(
        path=args.swig_train_file,
        space_path=args.swig_space_file,
        image_dir=args.swig_img_dir,
        mapping=swig2ace_mapping,
        prompts_path=args.swig_prompts_file
    )

    # SWiG development dataset
    dev_img_dataset = SwigDatasetEAE(
        path=args.swig_dev_file,
        space_path=args.swig_space_file,
        image_dir=args.swig_img_dir,
        mapping=swig2ace_mapping,
        prompts_path=args.swig_prompts_file
    )

    return {
        "train_txt_dataset": train_txt_dataset,
        "dev_txt_dataset": dev_txt_dataset,
        "train_img_dataset": train_img_dataset,
        "dev_img_dataset": dev_img_dataset
    }


def prepare_txt_event_detection_golds(dataset):
    # Just for ED evaluation
    from torch.utils.data import DataLoader
    samples = []

    dataloader = DataLoader(
        dataset, batch_size=12,
        collate_fn=dataset.collate_fn, shuffle=False
    )

    for batch in dataloader:
        iterator = zip(
                batch.doc_ids, batch.wnd_ids, batch.tokens, 
                batch.triggers
        )
        for doc_id, wnd_id, tokens, triggers in iterator:
            samples.append({
                "doc_id": doc_id, "wnd_id": wnd_id, "tokens": tokens,
                "triggers": triggers
            })
    return samples


def prepare_txt_event_detection_preds(dataset):
    # Just for ED evaluation
    results = {}
    for pred in dataset:
        if pred["wnd_id"] not in results:
            results[pred["wnd_id"]] = {
                "doc_id": pred["doc_id"], "wnd_id": pred["wnd_id"],
                "triggers": []
            }
        results[pred["wnd_id"]]["triggers"].append(
            [pred["start"], pred["end"], pred["event_type"], pred["text"]]
        )
    return list(results.values())


def prepare_img_event_detection_golds(dataset):
    # Just for ED evaluation
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset, batch_size=12,
        collate_fn=dataset.collate_fn, shuffle=False
    )
    data_iterator = iter(dataloader)

    ground_truths = []
    for iteration in tqdm.trange(len(dataloader), desc="Gold"):
        batch = next(data_iterator)

        iterator = zip(
            batch.doc_ids, batch.wnd_ids, batch.triggers
        )
        for doc_id, wnd_id, trigger in iterator:
            if trigger != "O":
                ground_truths.append({
                    "doc_id": doc_id, "wnd_id": wnd_id, "trigger": trigger
                })
    return ground_truths


def main(args):
    ### Datasets & Schema mapping

    m2e2_preds_dict = prepare_pred_datasets(args)
    m2e2_ed_datasets = prepare_ed_datasets(args)
    m2e2_eae_datasets = prepare_eae_datasets(args, m2e2_preds_dict)

    m2e2_type_set = {}
    m2e2_type_set["event_type_set"] = m2e2_eae_datasets["m2e2_txt_eae_gt_dataset"].event_type_set
    m2e2_type_set["argument_type_set"] = m2e2_eae_datasets["m2e2_txt_eae_gt_dataset"].argument_type_set


    swig2ace_mapping_file = args.swig2ace_mapping_file
    swig2ace_mapping = prepare_swig2ace_mapping(swig2ace_mapping_file)
    train_datasets = prepare_train_datasets(args, m2e2_type_set, swig2ace_mapping)

    for key in train_datasets["train_txt_dataset"].argument_type_set:
        if not key in m2e2_type_set["argument_type_set"]:
            m2e2_type_set["argument_type_set"].add(key)
    for key in train_datasets["train_img_dataset"].argument_type_set:
        if not key in m2e2_type_set["argument_type_set"]:
            m2e2_type_set["argument_type_set"].add(key)
    m2e2_type_set["argument_type_set"].add("Beneficiary")
    args.type_set = m2e2_type_set

    ###

    ### Model & Training

    os.makedirs(args.output_dir, exist_ok=True)
    trainer = TrainerJoint(args, type_set=m2e2_type_set)
    trainer.load_model(args.checkpoint)

    if args.do_train:
        train_objectives = []
        dev_objectives = []

        if args.add_ace_task:
            train_objectives.append(
                {"key": "ACE", "dataset": train_datasets["train_txt_dataset"], 
                "batch_size": args.txt_train_batch_size, "modality": "text"}
            )
            dev_objectives.append(
                {"key": "ACE", "dataset": train_datasets["dev_txt_dataset"],
                "batch_size": args.txt_eval_batch_size, "modality": "text"},
            )

        if args.add_swig_task:
            train_objectives.append(
                {"key": "SWIG", "dataset": train_datasets["train_img_dataset"],
                "batch_size": args.img_train_batch_size, "modality": "vision"}
            )
            dev_objectives.append(
                {"key": "SWIG", "dataset": train_datasets["dev_img_dataset"],
                "batch_size": args.img_eval_batch_size, "modality": "vision"}
            )

        trainer.train(train_objectives=train_objectives, dev_objectives=dev_objectives)

    ###

    if args.do_eval:
        ### Evaluate text ED

        print("### EVENT DETECTION (TEXT) ###")
        txt_gold_triggers = prepare_txt_event_detection_golds(m2e2_ed_datasets["m2e2_txt_ed_dataset"])
        txt_pred_triggers = prepare_txt_event_detection_preds(m2e2_preds_dict["m2e2_txt_pred_triggers"])
        txt_ed_scores = eval_text.compute_ED_scores(preds=txt_pred_triggers, golds=txt_gold_triggers)
        eval_text.print_scores(txt_ed_scores)

        ###

        ### Evaluate vision ED

        print("### EVENT DETECTION (VISION)")
        img_gold_triggers = prepare_img_event_detection_golds(m2e2_ed_datasets["m2e2_img_ed_dataset"])
        img_pred_triggers = m2e2_preds_dict["m2e2_img_pred_triggers"]
        img_ed_scores = eval_image.compute_ED_scores(preds=img_pred_triggers, golds=img_gold_triggers)
        eval_image.print_scores(img_ed_scores)

        ###

        ### Evalute text EAE

        print("### EVENT ARGUMENTS (TEXT) ###")

        txt_eae_golds = trainer.prepare_golds(dataset=m2e2_eae_datasets["m2e2_txt_eae_gt_dataset"], 
                                              batch_size=args.txt_eval_batch_size, modality="text")

        print("### Gold Triggers")

        txt_eae_gt_preds = trainer.predict(dataset=m2e2_eae_datasets["m2e2_txt_eae_gt_dataset"], 
                                        batch_size=args.txt_eval_batch_size, modality="text")
        txt_eae_gt_scores = eval_text.compute_EAE_scores(txt_eae_gt_preds, txt_eae_golds)
        eval_text.print_scores(txt_eae_gt_scores)

        txt_eae_gt_preds_file = os.path.join(args.output_dir, "txt_eae_gt_preds.json")
        txt_eae_gt_scores_file = os.path.join(args.output_dir, "txt_eae_gt_results.json")
        with open(txt_eae_gt_preds_file, "w") as fp: json.dump(txt_eae_gt_preds, fp)
        with open(txt_eae_gt_scores_file, "w") as fp: json.dump(txt_eae_gt_scores, fp)

        print("### Prediction Triggers")

        txt_eae_preds = trainer.predict(dataset=m2e2_eae_datasets["m2e2_txt_eae_pt_dataset"], 
                                        batch_size=args.txt_eval_batch_size, modality="text")
        txt_eae_scores = eval_text.compute_EAE_scores(txt_eae_preds, txt_eae_golds)
        eval_text.print_scores(txt_eae_scores)

        txt_eae_preds_file = os.path.join(args.output_dir, "txt_eae_preds.json")
        txt_eae_scores_file = os.path.join(args.output_dir, "txt_eae_results.json")
        with open(txt_eae_preds_file, "w") as fp: json.dump(txt_eae_preds, fp)
        with open(txt_eae_scores_file, "w") as fp: json.dump(txt_eae_scores, fp)

        ###

        ### Evaluate vision EAE

        print("\n### EVENT ARGUMENTS (VISION)")

        img_eae_golds = trainer.prepare_golds(dataset=m2e2_eae_datasets["m2e2_img_eae_gt_dataset"], 
                                              batch_size=args.img_eval_batch_size, modality="vision")

        print("### Gold Triggers")

        img_eae_gt_preds = trainer.predict(dataset=m2e2_eae_datasets["m2e2_img_eae_gt_dataset"], 
                                              batch_size=args.img_eval_batch_size, modality="vision")
        img_eae_gt_scores = eval_image.compute_EAE_scores(img_eae_gt_preds, img_eae_golds)
        eval_image.print_scores(img_eae_gt_scores)

        img_eae_gt_preds_file = os.path.join(args.output_dir, "img_eae_gt_preds.json")
        img_eae_gt_scores_file = os.path.join(args.output_dir, "img_eae_gt_results.json")
        with open(img_eae_gt_preds_file, "w") as fp: json.dump(img_eae_gt_preds, fp)
        with open(img_eae_gt_scores_file, "w") as fp: json.dump(img_eae_gt_scores, fp)

        print("### Prediction Triggers")

        img_eae_preds = trainer.predict(dataset=m2e2_eae_datasets["m2e2_img_eae_pt_dataset"], 
                                              batch_size=args.img_eval_batch_size, modality="vision")
        img_eae_scores = eval_image.compute_EAE_scores(img_eae_preds, img_eae_golds)
        eval_image.print_scores(img_eae_scores)

        img_eae_preds_file = os.path.join(args.output_dir, "img_eae_preds.json")
        img_eae_scores_file = os.path.join(args.output_dir, "img_eae_results.json")
        with open(img_eae_preds_file, "w") as fp: json.dump(img_eae_preds, fp)
        with open(img_eae_scores_file, "w") as fp: json.dump(img_eae_scores, fp)

        ###


if __name__ == "__main__":
    args = parse_args()
    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts)
    dt = dt.strftime('%Y-%m-%d %H:%M')
    wandb.init(
        project="MMUTF", 
        name=f"{args.experiment} ({dt})", 
        config=args
    )
    main(args)