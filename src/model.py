import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.context import TextEncoder
from src.modules.context import VisionEncoder
from src.modules.query import QueryEncoder


ModalityType = types.SimpleNamespace(
    VISION="vision", TEXT="text"
)


class MultimediaExtractor(nn.Module):
    def __init__(self, config, role_set=None):
        super().__init__()
        self.config = config

        self.encoders = nn.ModuleDict(
            {
                ModalityType.TEXT: TextEncoder(config),
                ModalityType.VISION: VisionEncoder(config)
            }
        )

        self.encoders[ModalityType.VISION].prepare_preprocessor(
            self.encoders[ModalityType.TEXT].tokenizer
        )
        self.encoders[ModalityType.TEXT].prepare_preprocessor(
            self.encoders[ModalityType.VISION].processor
        )

        self.postprocessors = {
            ModalityType.TEXT: self._postprocess_text,
            ModalityType.VISION: self._postprocess_vision
        }

        self.query_model = QueryEncoder(config)


    def forward(self, inputs, modality):
        # prepare batch
        q_inputs = self.query_model.preprocessor(inputs)
        c_inputs = self.encoders[modality].preprocessor(inputs, q_inputs["slot_stoi"], train=True)

        # compute logits
        logits = self.compute_logits(q_inputs, c_inputs, modality)

        # compute loss
        cand_labels, q_token_nums = c_inputs["cand_labels"], q_inputs["token_nums"]
        loss = self.compute_loss(q_token_nums, logits, cand_labels)

        return loss


    def compute_logits(self, batch_q_inputs, batch_c_inputs, modality):
        # (batch_size, tokens, hidden), (batch_size, cands, hidden)
        encoder_fn = self.encoders[modality].embed_candidates
        all_embeddings, cand_embeddings = encoder_fn(batch_c_inputs)
                
        # (batch_size, slots, hidden)
        query_fn = self.query_model.embed_slots
        slot_embeddings = query_fn(batch_q_inputs, all_embeddings)

        # compute slot logits
        logits = torch.einsum("BCD,BSD->BCS", cand_embeddings, slot_embeddings)

        # (batch_size, cands, slots)
        return logits
    

    def compute_loss(self, q_token_nums, logits, labels):
        device = logits.device

        # mask for padded slots
        batch_size = len(q_token_nums)
        max_num_slots = torch.max(q_token_nums)
        slots_mask = torch.arange(max_num_slots)
        slots_mask = slots_mask.to(device)
        slots_mask = slots_mask.unsqueeze(0).expand(batch_size, -1)
        slots_mask = slots_mask < q_token_nums.unsqueeze(-1)

        # mask for padded candidates
        cand_labels = labels.view(-1)
        cand_mask = cand_labels != -1
        cand_labels.masked_fill_(~cand_mask, 0)

        # one-hot encoding
        sample_size = cand_labels.size(0)
        labels_one_hot = torch.zeros(sample_size, max_num_slots+1).float()
        labels_one_hot = labels_one_hot.to(device)
        labels_one_hot.scatter_(1, cand_labels.unsqueeze(1), 1)
        labels_one_hot = labels_one_hot[:, 1:] # remove padding label 0
        labels_one_hot = labels_one_hot

        # compute loss
        all_losses = F.binary_cross_entropy_with_logits(
            input=logits.view(-1, max_num_slots), target=labels_one_hot, reduction="none"
        )

        # padded slots masking
        all_losses = all_losses.view(batch_size, -1, max_num_slots)
        all_losses = all_losses * slots_mask.unsqueeze(1)
        all_losses = all_losses.view(-1, max_num_slots)

        # padded candidates masking
        cand_mask = cand_mask.unsqueeze(-1).expand_as(all_losses)
        final_loss = all_losses * cand_mask.float()
        return final_loss.sum()
    

    @torch.no_grad()
    def predict(self, inputs, modality, threshold=0.5):
        self.eval()

        # prepare batch
        q_inputs = self.query_model.preprocessor(inputs)
        c_inputs = self.encoders[modality].preprocessor(inputs, q_inputs["slot_stoi"], train=False)

        # compute probs
        logits = self.compute_logits(q_inputs, c_inputs, modality)
        probs = torch.sigmoid(logits)

        # get predictions
        predictions = self.postprocessors[modality](
            probs, inputs, q_inputs, c_inputs, threshold
        )
        return predictions
    

    def _postprocess_text(self, probs, inputs, q_inputs, c_inputs, threshold):
        cand_word_spans = c_inputs["word_spans"]
        slot_itos = q_inputs["slot_itos"]

        # get predictions
        predictions = []
        for i, scores in enumerate(probs):
            candidates_i = [it.tolist() for it in torch.where(scores >= threshold)]

            predictions_tmp = []
            for cand_idx, slot_idx in zip(*candidates_i):
                if cand_idx < len(cand_word_spans[i]):
                    span_start = cand_word_spans[i][cand_idx][0]
                    span_end = cand_word_spans[i][cand_idx][1]
                    if slot_itos[i].get(slot_idx+1):
                        score = scores[cand_idx][slot_idx]
                        span_role = slot_itos[i].get(slot_idx+1)
                        span_text = inputs.tokens[i][span_start:span_end]
                        pred_item = ([span_start, span_end, span_role, span_text], score)
                        predictions_tmp.append(pred_item)

            # select max score slot
            cands_seen = []
            predictions_i = []
            predictions_tmp.sort(key=lambda i: (i[0][:2], i[1]*-1))
            for tmp in predictions_tmp:
                if not tmp[0][:2] in cands_seen:
                    cands_seen.append(tmp[0][:2])
                    predictions_i.append(tuple(tmp[0]))

            predictions.append(predictions_i)
        return predictions
    

    def _postprocess_vision(self, probs, inputs, q_inputs, c_inputs, threshold):
        object_nums = c_inputs["object_nums"]
        slot_itos = q_inputs["slot_itos"]

        # get predictions
        predictions = []
        for i, scores in enumerate(probs):
            candidates_i = [it.tolist() for it in torch.where(scores >= threshold)]

            predictions_tmp = []
            for cand_idx, slot_idx in zip(*candidates_i):
                if cand_idx < object_nums[i]:
                    if slot_itos[i].get(slot_idx+1):
                        score = scores[cand_idx][slot_idx]
                        bbox = inputs.objects[i][cand_idx][:4]
                        span_role = slot_itos[i].get(slot_idx+1)
                        pred_item = (bbox + [span_role], score)
                        predictions_tmp.append(pred_item)

            # select max score slot
            cands_seen = []
            predictions_i = []
            predictions_tmp.sort(key=lambda i: (i[0][:4], i[1]*-1))
            for tmp in predictions_tmp:
                if not tmp[0][:4] in cands_seen:
                    cands_seen.append(tmp[0][:4])
                    predictions_i.append(tuple(tmp[0]))

            predictions.append(predictions_i)
        return predictions