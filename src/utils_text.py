import torch
import collections


def span_first_subtoken(encodings, token_spans, token_nums):
    batch_size, n_subtokens, hidden_size = encodings.size()

    all_token_embeddings = torch.zeros(
        batch_size, max(token_nums), hidden_size
    ).to(encodings.device)

    for i in torch.arange(batch_size):
        for j in torch.arange(token_nums[i]):
            token_span = token_spans[i][j]
            token_start = token_span[0]
            token_enconding = encodings[i, token_start, :]
            all_token_embeddings[i, j, :] = token_enconding
    return all_token_embeddings


def span_mean_pooling(encodings, token_spans, token_nums):
    batch_size, n_subtokens, hidden_size = encodings.size()

    all_token_embeddings = torch.zeros(
        batch_size, max(token_nums), hidden_size
    ).to(encodings.device)

    for i in torch.arange(batch_size):
        for j in torch.arange(token_nums[i]):
            token_span = token_spans[i][j]
            token_start = token_span[0]
            token_end = token_span[1]
            token_ids = torch.arange(token_start, token_end)
            all_token_embeddings[i, j, :] = torch.nan_to_num(
                encodings[i, token_ids, :].mean(dim=0)
            )
    return all_token_embeddings


def create_entity_candidates(entities, arguments, stoi):
    candidate_spans = []
    for entity in entities:
        entity_start = entity[0]
        entity_end = entity[1]
        entity_span = (entity_start, entity_end)
        candidate_spans.append(entity_span)

    ground_truth = collections.defaultdict(int)
    for argument in arguments:
        arg_start = argument[0]
        arg_end = argument[1]
        arg_type = argument[2]
        arg_span = (arg_start, arg_end)
        ground_truth[arg_span] = stoi[arg_type]

    candidate_labels = [
        ground_truth[span] for span in candidate_spans
    ]
    return candidate_spans, candidate_labels


def tokens_to_subtokens(token_spans, tokenmap):
    new_token_spans = []
    for token_span in token_spans:
        token_start = token_span[0]
        token_end = token_span[1]
        new_token_start = tokenmap[token_start][0]
        new_token_end = tokenmap[token_end - 1][1]
        new_token_span = (new_token_start, new_token_end)
        new_token_spans.append(new_token_span)
    return new_token_spans