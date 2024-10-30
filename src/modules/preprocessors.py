import torch
import torch.nn.utils.rnn as rnn

import src.utils_text as utils_text
import src.utils_vision as utils_vision


class TextQueryProcessor:
    def __init__(self, tokenizer, device, span_pad=0):
        self.text_tokenizer = tokenizer
        self.device = device
        self.span_pad = span_pad


    def __call__(self, batch):
        batch_inputs = {"input_ids": [], "attention_mask": []}
        iterator = zip(batch.prompt_tokens, batch.prompt_slots)

        batch_slot_stoi = []
        batch_slot_itos = []
        batch_token_nums = []
        batch_slot_spans = []
        for tokens, slots in iterator:
            inputs = self.text_tokenizer(tokens, is_split_into_words=True)
            starts = [inputs.word_to_tokens(i).start for i, _ in enumerate(tokens)]
            ends = [inputs.word_to_tokens(i).end for i, _ in enumerate(tokens)]
            tok2subtok = {i: (s, e) for i, (s, e) in enumerate(zip(starts, ends))}
            slot_spans = [(slot["start"], slot["end"]) for slot in slots.values()]
            slot_spans = utils_text.tokens_to_subtokens(slot_spans, tok2subtok)

            batch_inputs["attention_mask"].append(inputs["attention_mask"])
            batch_inputs["input_ids"].append(inputs["input_ids"])

            slot_stoi = {s: i for i, s in enumerate(slots.keys(), 1)}
            slot_itos = {i: s for s, i in slot_stoi.items()}

            batch_slot_stoi.append(slot_stoi)
            batch_slot_itos.append(slot_itos)
            batch_token_nums.append(len(slot_spans))
            batch_slot_spans.append(torch.LongTensor(slot_spans))

        batch_inputs = self.text_tokenizer.pad(
            batch_inputs, padding=True, max_length=None, return_tensors="pt"
        )

        batch_slot_spans = rnn.pad_sequence(
            batch_slot_spans, batch_first=True, padding_value=self.span_pad
        )

        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        batch_token_nums = torch.LongTensor(batch_token_nums).to(self.device)
        batch_slot_spans = batch_slot_spans.to(self.device)

        return {"inputs": batch_inputs, "slot_spans": batch_slot_spans, "slot_stoi": batch_slot_stoi, 
                "slot_itos": batch_slot_itos, "token_nums": batch_token_nums}


class TextCandidatesProcessor:
    def __init__(self, tokenizer, processor, device, span_pad=0, label_pad=-1):
        self.text_tokenizer = tokenizer
        self.vision_processor = processor
        self.device = device
        self.span_pad = span_pad
        self.label_pad = label_pad


    def __call__(self, batch, batch_stoi, train):
        batch_inputs = {"input_ids": [], "attention_mask": []}
        iterator = zip(batch.tokens, batch.entities, batch.arguments, batch_stoi)

        batch_token_nums = []
        batch_word_spans = []
        batch_cand_spans = []
        batch_cand_labels = []
        for tokens, candidates, arguments, stoi in iterator:
            inputs = self.text_tokenizer(tokens, is_split_into_words=True)
            starts = [inputs.word_to_tokens(i).start for i, _ in enumerate(tokens)]
            ends = [inputs.word_to_tokens(i).end for i, _ in enumerate(tokens)]
            tok2subtok = {i: (s, e) for i, (s, e) in enumerate(zip(starts, ends))}

            cand_spans, cand_labels = utils_text.create_entity_candidates(
                entities=candidates, arguments=arguments, stoi=stoi
            )
            batch_word_spans.append(cand_spans)
            cand_spans = utils_text.tokens_to_subtokens(cand_spans, tok2subtok)

            batch_inputs["attention_mask"].append(inputs["attention_mask"])
            batch_inputs["input_ids"].append(inputs["input_ids"])

            if len(cand_spans) == 0: cand_spans = [(0, 0)]
            if len(cand_labels) == 0: cand_labels = [-1]
            batch_cand_spans.append(torch.LongTensor(cand_spans))
            batch_cand_labels.append(torch.LongTensor(cand_labels))
            batch_token_nums.append(len(cand_spans))

        batch_inputs = self.text_tokenizer.pad(
            batch_inputs, padding=True, max_length=None, return_tensors="pt"
        )
        
        batch_cand_spans = rnn.pad_sequence(
            batch_cand_spans, batch_first=True, padding_value=self.span_pad
        )

        batch_cand_labels = rnn.pad_sequence(
            batch_cand_labels, batch_first=True, padding_value=self.label_pad
        )

        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        batch_token_nums = torch.LongTensor(batch_token_nums).to(self.device)
        batch_cand_spans = batch_cand_spans.to(self.device)
        batch_cand_labels = batch_cand_labels.to(self.device)

        return {"inputs": batch_inputs, "cand_spans": batch_cand_spans, "cand_labels": batch_cand_labels, 
                "word_spans": batch_word_spans, "token_nums": batch_token_nums, 
                "triggers": batch.triggers}
    

class VisionCandidatesProcessor:
    def __init__(self, processor, tokenizer, device, img_size, patch_size, span_pad=0, label_pad=-1):
        self.vision_processor = processor
        self.text_tokenizer = tokenizer
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size
        self.span_pad = span_pad
        self.label_pad = label_pad


    def __call__(self, batch, batch_stoi, train):
        batch_inputs = {"pixel_values": []}
        iterator = zip(batch.images, batch.objects, batch.arguments, batch_stoi)

        batch_cand_bboxes = []
        batch_cand_spans = []
        batch_cand_labels = []
        batch_object_nums = []
        for image, candidates, arguments, stoi in iterator:
            preprocessed = self.vision_processor(image, bboxes=candidates, train=train)
            #inputs_preprocessed = torch.tensor(preprocessed["image"]).permute((2, 0, 1)).unsqueeze(0)
            inputs_preprocessed = preprocessed["pixel_values"]
            candidates_preprocessed = preprocessed["bboxes"]
            
            cand_spans, cand_labels = utils_vision.create_object_candidates(
                image=image, 
                objects=candidates_preprocessed, 
                orig_objects=candidates,
                arguments=arguments,
                stoi=stoi, img_size=self.img_size, 
                patch_size=self.patch_size
            )

            batch_inputs["pixel_values"].append(inputs_preprocessed)

            if len(cand_spans) == 0: cand_spans = [(0, 0, 0)]
            if len(cand_labels) == 0: cand_labels = [-1]
            batch_cand_spans.append(torch.LongTensor(cand_spans))
            batch_cand_labels.append(torch.LongTensor(cand_labels))
            batch_object_nums.append(len(candidates))

            cand_bboxes = [bbox[:4] for bbox in candidates_preprocessed]
            if len(cand_bboxes) == 0: cand_bboxes = [(0, 0, 0, 0)]
            batch_cand_bboxes.append(torch.LongTensor(cand_bboxes))

        batch_inputs["pixel_values"] = torch.concat(batch_inputs["pixel_values"])

        batch_cand_spans = rnn.pad_sequence(
            batch_cand_spans, batch_first=True, padding_value=self.span_pad
        )

        batch_cand_labels = rnn.pad_sequence(
            batch_cand_labels, batch_first=True, padding_value=self.label_pad
        )

        batch_cand_bboxes = rnn.pad_sequence(
            batch_cand_spans, batch_first=True, padding_value=0
        )

        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        batch_cand_spans = batch_cand_spans.to(self.device)
        batch_cand_labels = batch_cand_labels.to(self.device)
        batch_cand_bboxes = batch_cand_bboxes.to(self.device)

        return {"inputs": batch_inputs, "cand_spans": batch_cand_spans, "cand_labels": batch_cand_labels, 
                "cand_bboxes": batch_cand_bboxes, "object_nums": batch_object_nums, "triggers": batch.triggers}