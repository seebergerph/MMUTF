import torch
import torch.nn as nn

from transformers import AutoImageProcessor
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import CLIPModel
from transformers import ViTModel

import src.utils_text as utils_text
import src.utils_vision as utils_vision
from src.modules.preprocessors import TextCandidatesProcessor
from src.modules.preprocessors import VisionCandidatesProcessor


def create_mapping_network(hidden_size, dropout, out_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim)
    )


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._prepare_model(config)
        self._prepare_projection(config)
        self._prepare_embeddings(config)
        self._freeze_model(config)
                

    def _prepare_model(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.context_text_model,
            use_fast=True, add_prefix_space=True 
        )

        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config.context_text_model
        )

        if ("bart" in config.context_text_model.lower() or
            "t5" in config.context_text_model.lower()):
            self.model = self.model.encoder
        else:
            raise ValueError(f"Text model {config.context_text_model} not supported!")

        print(f"TextEncoder (Context): Loaded {config.context_text_model}")


    def _prepare_projection(self, config):
        model_config = self.model.config
        hidden_size = model_config.hidden_size
        self.hidden_size = hidden_size
        self.projection = nn.Identity()
        self.cand_mlp = create_mapping_network(
            hidden_size*2, dropout=0.4, out_dim=hidden_size
        )


    def _prepare_embeddings(self, config):
        eventtype_num = len(config.type_set["event_type_set"])
        self.eventtype_stoi = {et:i for i, et in enumerate(config.type_set["event_type_set"])}
        self.eventtype_emb = nn.Embedding(eventtype_num, embedding_dim=100)


    def prepare_preprocessor(self, processor):
        self.preprocessor = TextCandidatesProcessor(
            tokenizer=self.tokenizer, processor=processor, device=self.config.device
        )


    def forward(self, inputs):
        att_mask = inputs["attention_mask"]
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        features = self.projection(features)
        return {"hidden_states": features, "attention_mask": att_mask}
    

    def embed_candidates(self, batch_c_inputs):
        c_inputs, c_triggers, cand_spans, cand_labels, word_spans, c_token_nums = (
            batch_c_inputs["inputs"], batch_c_inputs["triggers"], batch_c_inputs["cand_spans"],
            batch_c_inputs["cand_labels"], batch_c_inputs["word_spans"],
            batch_c_inputs["token_nums"]
        )

        # compute context embeddings
        context_embeddings = self.forward(c_inputs)

        # compute candidate embeddings
        cand_embeddings = utils_text.span_mean_pooling(
            encodings=context_embeddings["hidden_states"], 
            token_spans=cand_spans, 
            token_nums=c_token_nums
        )

        # add trigger embeddings
        trigger_spans = [[(t[0],t[1])] for t in c_triggers]
        trigger_token_nums = [1 for _ in range(len(trigger_spans))]

        trigger_embeddings = utils_text.span_mean_pooling(
            encodings=context_embeddings["hidden_states"], 
            token_spans=trigger_spans, 
            token_nums=trigger_token_nums
        )

        trigger_embeddings = trigger_embeddings.repeat(1, cand_embeddings.size(1), 1)
        cand_embeddings = torch.cat((cand_embeddings, trigger_embeddings), dim=-1)

        # compute final embeddings
        cand_embeddings = self.cand_mlp(cand_embeddings)

        # (batch_size, tokens, hidden), (batch_size, cands, hidden)
        return context_embeddings, cand_embeddings


    def _freeze_model(self, config):
        if config.context_freeze_text_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            print("TextEncoder (Context): Freezed context text encoder parameters!")
        if config.context_freeze_text_all:
            for param in self.parameters():
                param.requires_grad = False
            print("TextEncoder (Context): Freezed context text model parameters!")
    

class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._prepare_model(config)
        self._prepare_projection(config)
        self._prepare_embeddings(config)
        self._prepare_patch_info()
        self._freeze_model(config)


    def _prepare_model(self, config):
        if "openai/clip-vit" in config.context_vision_model.lower():
            pp_processor= AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config.context_vision_model
            )
            self.model = CLIPModel.from_pretrained(
                pretrained_model_name_or_path=config.context_vision_model
            )
            self.model = self.model.vision_model

        elif "google/vit" in config.context_vision_model.lower():
            pp_processor = AutoImageProcessor.from_pretrained(
                pretrained_model_name_or_path=config.context_vision_model
            )
            self.model = ViTModel.from_pretrained(
                pretrained_model_name_or_path=config.context_vision_model
            )

        else:
            raise ValueError(f"Vision model {config.context_vision_model} not supported!")

        model_config = self.model.config
        img_size = model_config.image_size
        self.processor = utils_vision.build_transform_fn(
            img_size=img_size, preprocessor=pp_processor
        )

        print(f"VisionEncoder (Context): Loaded {config.context_vision_model}")


    def _prepare_projection(self, config):
        model_config = self.model.config

        if config.context_vision_projection_flag:
            bias = config.context_vision_projection_bias
            proj_size = config.context_vision_projection_size
            self.projection = nn.Linear(model_config.hidden_size, proj_size, bias=bias)
            self.hidden_size = proj_size
        else:
            self.projection = nn.Identity()
            self.hidden_size = model_config.hidden_size

        self.cand_mlp = create_mapping_network(
            self.hidden_size*2, dropout=0.4, out_dim=self.hidden_size
        )


    def _prepare_embeddings(self, config):
        eventtype_num = len(config.type_set["event_type_set"])
        self.eventtype_stoi = {et:i for i, et in enumerate(config.type_set["event_type_set"])}
        self.eventtype_emb = nn.Embedding(eventtype_num, embedding_dim=100)


    def _prepare_patch_info(self):
        model_config = self.model.config
        self.image_size = model_config.image_size
        self.patch_size = model_config.patch_size
        self.patch_num = int(self.image_size / self.patch_size)


    def prepare_preprocessor(self, tokenizer):
        self.preprocessor = VisionCandidatesProcessor(
            processor=self.processor, tokenizer=tokenizer, device=self.config.device, 
            img_size=self.image_size, patch_size=self.patch_size
        )


    def forward(self, inputs):
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        features = self.projection(features)
        att_mask = torch.ones((features.size(0), self.patch_num * self.patch_num + 1))
        att_mask = att_mask.to(features.device)
        return {"hidden_states": features, "attention_mask": att_mask}
    

    def embed_candidates(self, batch_c_inputs):
        c_inputs, c_triggers, cand_spans, cand_bboxes, cand_labels = (
            batch_c_inputs["inputs"], batch_c_inputs["triggers"], batch_c_inputs["cand_spans"],
            batch_c_inputs["cand_bboxes"], batch_c_inputs["cand_labels"]
        )

        # compute context embeddings
        context_embeddings = self.forward(c_inputs)

        # compute candidate embeddings
        cand_embeddings = []
        for idx, cand_span in enumerate(cand_spans):
            _cand_embeddings = []
            for cand_idxs in cand_span:
                # max pooling
                _cand_encoding = context_embeddings["hidden_states"][idx][
                    cand_idxs[0]:cand_idxs[-1]+1
                ]
                _cand_encoding = torch.max(_cand_encoding, dim=0)[0]
                _cand_embeddings.append(_cand_encoding.unsqueeze(0))

            _cand_embeddings = torch.concat(_cand_embeddings)
            cand_embeddings.append(_cand_embeddings.unsqueeze(0))
        cand_embeddings = torch.concat(cand_embeddings)

        # add trigger embeddings
        trigger_embeddings = context_embeddings["hidden_states"][:,0,:]
        trigger_embeddings = trigger_embeddings.unsqueeze(1).repeat(1, cand_embeddings.size(1), 1)
        cand_embeddings = torch.cat((cand_embeddings, trigger_embeddings), dim=-1)

        # compute final embeddings
        cand_embeddings = self.cand_mlp(cand_embeddings)

        # (batch_size, tokens, hidden), (batch_size, cands, hidden)
        return context_embeddings, cand_embeddings


    def _freeze_model(self, config):
        if config.context_freeze_vision_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            print("VisionEncoder (Context): Freezed context vision encoder parameters!")
        if config.context_freeze_vision_all:
            for param in self.parameters():
                param.requires_grad = False
            print("VisionEncoder (Context): Freezed context vision model parameters!")