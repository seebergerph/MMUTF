import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModel

import src.utils_text as utils_text
from src.modules.preprocessors import TextQueryProcessor


def create_mapping_network(hidden_size, dropout, out_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim)
    )


class QueryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._prepare_model(config)
        self._prepare_projection(config)
        self._prepare_preprocessor(config)
        self._freeze_model(config)


    def _prepare_model(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.query_text_model,
            use_fast=True, add_prefix_space=True
        )

        if ("bart" in config.query_text_model.lower() or
            "t5" in config.query_text_model.lower()):
            print("QueryEncoder: Use decoder model!")
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=config.query_text_model
            )
            self.model = self.model.decoder
        else:
            raise ValueError(f"Query model {config.query_text_model} not supported!")

        print(f"QueryEncoder: Loaded {config.query_text_model}")


    def _prepare_projection(self, config):
        model_config = self.model.config
        hidden_size = model_config.hidden_size
        self.projection = create_mapping_network(
            hidden_size, dropout=0.4, out_dim=hidden_size
        )


    def _prepare_preprocessor(self, config):
        self.preprocessor = TextQueryProcessor(
            self.tokenizer, config.device
        )


    def forward(
        self, dec_prompt_ids, dec_prompt_mask, enc_context_states, 
        enc_context_mask, dec_slot_spans, dec_token_nums
    ):
        # compute query embeddings
        prompt_outputs = self.model(
            input_ids=dec_prompt_ids,
            attention_mask=dec_prompt_mask,
            encoder_hidden_states=enc_context_states,
            encoder_attention_mask=enc_context_mask
        )

        # (batch_size, tokens, hidden)
        prompt_outputs = prompt_outputs.last_hidden_state

        # compute slot embeddings
        slot_encodings = utils_text.span_mean_pooling(
            encodings=prompt_outputs, 
            token_spans=dec_slot_spans, 
            token_nums=dec_token_nums
        )

        # (batch_size, slots, hidden)
        return self.projection(slot_encodings)
    

    def embed_slots(self, batch_q_inputs, batch_c_embeddings):
        q_inputs, slot_spans, slot_stoi, slot_itos, q_token_nums = (
            batch_q_inputs["inputs"], batch_q_inputs["slot_spans"], 
            batch_q_inputs["slot_stoi"],batch_q_inputs["slot_itos"], 
            batch_q_inputs["token_nums"]
        )

        # compute slot embeddings
        slot_embeddings = self.forward(
            dec_prompt_ids=q_inputs["input_ids"],
            dec_prompt_mask=q_inputs["attention_mask"],
            enc_context_states=batch_c_embeddings["hidden_states"],
            enc_context_mask=batch_c_embeddings["attention_mask"],
            dec_slot_spans=slot_spans, 
            dec_token_nums=q_token_nums
        )

        # (batch_size, slots, hidden)   
        return slot_embeddings


    def _freeze_model(self, config):
        if config.query_freeze_text_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            print("QueryEncoder: Freezed query text encoder parameters!")
        if config.query_freeze_text_all:
            for param in self.parameters():
                param.requires_grad = False
            print("QueryEncoder: Freezed query text model parameters!")