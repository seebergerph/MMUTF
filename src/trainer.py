import os
import json
import wandb
import torch
import logging
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from tqdm.autonotebook import trange
from transformers import get_linear_schedule_with_warmup

import src.eval_text as eval_text
import src.eval_image as eval_image
from src.model import MultimediaExtractor


EvaluationMetricFNs = {
    "text": eval_text.compute_EAE_scores,
    "vision": eval_image.compute_EAE_scores
}

EvaluationMetricPrints = {
    "text": eval_text.print_scores,
    "vision": eval_image.print_scores
}


class TrainerJoint:
    def __init__(self, config, type_set=None):
        self.config = config
        self.type_set = type_set


    def load_model(self, checkpoint=None):
        if checkpoint:
            logging.info(f"Loading model from {checkpoint}")
            state = torch.load(checkpoint, map_location=torch.device("cpu"))
            self.type_set = state["type_set"]
            self.model = MultimediaExtractor(
                self.config, role_set=self.type_set["argument_type_set"]
            )
            self.model.load_state_dict(state["model"], strict=False)
            self.model = self.model.to(self.config.device)
        else:
            logging.info(f"Loading model from {self.config.context_text_model}")
            self.model = MultimediaExtractor(
                self.config, role_set=self.type_set["argument_type_set"]
            )
            self.model = self.model.to(self.config.device)


    def train(self, train_objectives, dev_objectives):
        eval_key = self.config.evaluation_key
        eval_type = self.config.evaluation_metric
        eval_score = self.config.evaluation_score

        ### prepare dataloaders
        train_dataloaders = []
        for train_objective in train_objectives:
            dataset = train_objective["dataset"]
            dataloader = DataLoader(dataset, batch_size=train_objective["batch_size"], 
                                    collate_fn=dataset.collate_fn, shuffle=True,
                                    num_workers=0)
            train_dataloaders.append(dataloader)


        ### prepare optimizer & scheduler
        train_steps_per_epoch = max([len(dataloader) for dataloader in train_dataloaders])
        train_steps_all_epochs = train_steps_per_epoch * self.config.train_epochs
        train_steps_warmup = train_steps_all_epochs * self.config.train_epochs_warmup

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.train_weight_decay, 
                "lr": self.config.train_learning_rate
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                            if any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0, 
                "lr": self.config.train_learning_rate
            }
        ]
        optimizer = optim.AdamW(grouped_parameters)

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_training_steps=train_steps_all_epochs, 
            num_warmup_steps=train_steps_warmup
        )

        scaler = torch.GradScaler(self.config.device)

        ### train and eval model
        best_epoch = 0
        best_scores = None
        train_iterators = [iter(dataloader) for dataloader in train_dataloaders]

        for epoch in trange(self.config.train_epochs, desc="Epoch"):
            cum_loss = []
            self.model.train()
            optimizer.zero_grad()

            # train for epoch
            for iteration in trange(train_steps_per_epoch, desc="Train"):
                for train_idx in range(len(train_objectives)):
                    optimizer.zero_grad()
                    train_iterator = train_iterators[train_idx]

                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_dataloaders[train_idx])
                        train_iterators[train_idx] = train_iterator
                        batch = next(train_iterator)

                    modality = train_objectives[train_idx]["modality"]

                    with torch.autocast(device_type=self.config.device, dtype=torch.bfloat16):
                        loss = self.model(batch, modality=modality)
                    cum_loss.append(loss.item())

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    wandb.log({f"Train Loss ({modality})": loss.item()})

                    clip_grad_norm_(
                        parameters=self.model.parameters(), 
                        max_norm=self.config.clip_grad_max_norm
                    )

                    scaler.step(optimizer)
                    scaler.update()

                scheduler.step()
            logging.info(f"Average training loss: {np.mean(cum_loss)}")

            # eval for epoch
            dev_scores = {}
            for dev_idx in range(len(dev_objectives)):
                dev_key = dev_objectives[dev_idx]["key"]
                dev_modality = dev_objectives[dev_idx]["modality"]
                dev_dataset = dev_objectives[dev_idx]["dataset"]
                dev_batchsize = dev_objectives[dev_idx]["batch_size"]

                ground_truths = self.prepare_golds(dev_dataset, batch_size=dev_batchsize, modality=dev_modality)
                predictions = self.internal_predict(dev_dataset, batch_size=dev_batchsize, modality=dev_modality, 
                                                    split=f"Dev ({dev_key})")

                dev_scores_modality = EvaluationMetricFNs[dev_modality](predictions, ground_truths)
                dev_scores[dev_key] = dev_scores_modality

                wandb.log({
                    f"F1 - Argument CLS ({dev_key})": dev_scores_modality["argument_cls"]["f1"],
                    f"Recall - Argument CLS ({dev_key})": dev_scores_modality["argument_cls"]["recall"],
                    f"Precision - Argument CLS ({dev_key})": dev_scores_modality["argument_cls"]["precision"],
                })
            print(dev_scores)

            save_model = False
            if best_scores is not None:
                if (dev_scores[eval_key][eval_type][eval_score] >= 
                    best_scores[eval_key][eval_type][eval_score]):
                    save_model = True
            else:
                save_model = True

            if save_model:
                logging.info("Saving best model")
                state = dict(model=self.model.state_dict(), type_set=self.type_set)
                torch.save(state, os.path.join(self.config.output_dir, "best_model.state"))
                best_epoch = epoch + 1
                best_scores = dev_scores
                best_scores["epoch"] = best_epoch
                best_scores_file = os.path.join(self.config.output_dir, "best_dev_scores.json")
                json.dump(best_scores, open(best_scores_file, "w"))

            # print scores
            for dev_idx in range(len(dev_objectives)):
                dev_key = dev_objectives[dev_idx]["key"]
                dev_modality = dev_objectives[dev_idx]["modality"]

                print(f"Dev epoch: {epoch + 1}")
                EvaluationMetricPrints[dev_modality](dev_scores[dev_key])
                print()
                print(f"Best epoch: {best_epoch}")
                EvaluationMetricPrints[dev_modality](best_scores[dev_key])

        logging.info("Saving last model")
        state = dict(model=self.model.state_dict(), type_set=self.type_set)
        torch.save(state, os.path.join(self.config.output_dir, "last_model.state"))
        last_epoch = epoch + 1
        last_scores = dev_scores
        last_scores["epoch"] = last_epoch
        last_scores_file = os.path.join(self.config.output_dir, "last_dev_scores.json")
        json.dump(best_scores, open(last_scores_file, "w"))
        

    def prepare_golds(self, dataset, batch_size, modality):
        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            collate_fn=dataset.collate_fn, shuffle=False
        )
        data_iterator = iter(dataloader)

        ground_truths = []
        for iteration in trange(len(dataloader), desc="Gold"):
            batch = next(data_iterator)

            if modality == "text":
                iterator = zip(
                    batch.doc_ids, batch.wnd_ids, batch.tokens, 
                    batch.triggers, batch.arguments
                )
                for doc_id, wnd_id, tokens, trigger, arguments in iterator:
                    ground_truths.append({
                        "doc_id": doc_id, "wnd_id": wnd_id, "tokens": tokens,
                        "trigger": trigger, "arguments": arguments
                    })
            elif modality == "vision":
                iterator = zip(
                    batch.doc_ids, batch.wnd_ids, batch.triggers, 
                    batch.arguments
                )
                for doc_id, wnd_id, trigger, arguments in iterator:
                    ground_truths.append({
                        "doc_id": doc_id, "wnd_id": wnd_id, "trigger": trigger, 
                        "arguments": arguments
                    })

        return ground_truths
                        

    def internal_predict(self, dataset, batch_size, modality, threshold=0.5, split="Dev"):
        self.model.eval()

        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            collate_fn=dataset.collate_fn, shuffle=False
        )
        data_iterator = iter(dataloader)

        predictions = []
        for iteration in trange(len(dataloader), desc=split):
            batch = next(data_iterator)

            if modality == "text":
                batch_pred_arguments = self.model.predict(batch, modality, threshold=threshold)
                pred_iterator = zip(
                    batch.doc_ids, batch.wnd_ids, batch.tokens, 
                    batch.triggers, batch_pred_arguments
                )
    
                for doc_id, wnd_id, tokens, trigger, pred_arguments in pred_iterator:
                    prediction = {
                        "doc_id": doc_id,  "wnd_id": wnd_id, "tokens": tokens, 
                        "trigger": trigger, "arguments": pred_arguments
                    }
                    predictions.append(prediction)
            elif modality == "vision":
                batch_pred_arguments = self.model.predict(batch, modality, threshold=threshold)
                pred_iterator = zip(
                    batch.doc_ids, batch.wnd_ids, batch.triggers, batch_pred_arguments
                )
    
                for doc_id, wnd_id, trigger, pred_arguments in pred_iterator:
                    prediction = {
                        "doc_id": doc_id,  "wnd_id": wnd_id, "trigger": trigger, 
                        "arguments": pred_arguments
                    }
                    predictions.append(prediction)

        return predictions
    

    def predict(self, dataset, batch_size, modality, threshold=0.5):
        return self.internal_predict(dataset=dataset, batch_size=batch_size, threshold=threshold, 
                                     modality=modality, split="Test")