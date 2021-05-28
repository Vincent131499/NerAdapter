#-*- coding:utf-8 -*-
"""
    desc:
    author:MeteorMan
    datetime:2021/5/27
"""
import os
import torch
import json

from tokenizers import BertWordPieceTokenizer
from finetuning_argparse import get_argparse
from models.bert_query_ner import BertQueryNER
from models.query_ner_config import BertQueryNerConfig

def pad(lst, value=0, max_length=128):
    max_length = max_length
    while len(lst) < max_length:
        lst.append(value)
    return lst

def gen_input(query, context, tokenizer, max_length, pad_to_maxlen=False):
    """
    Args:
        item: int, idx
    Returns:
        tokens: tokens of query + context, [seq_len]
        token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
        start_labels: start labels of NER in tokens, [seq_len]
        end_labels: end labelsof NER in tokens, [seq_len]
        label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
        match_labels: match labels, [seq_len, seq_len]
        sample_idx: sample id
        label_idx: label id

    """
    # data = self.all_data[item]
    # tokenizer = self.tokenzier

    # qas_id = data.get("qas_id", "0.0")
    # sample_idx, label_idx = qas_id.split(".")
    # sample_idx = torch.LongTensor([int(sample_idx)])
    # label_idx = torch.LongTensor([int(label_idx)])

    # query = item["query"]
    # context = item["context"]

    tokens_str = ['[CLS]'] + list(query) + ['[SEP]'] + list(context) + ['[SEP]']
    tokens = []
    type_ids = [0] * (len(query) + 2) + [1] * (len(context) + 1)
    for token in tokens_str:
        if tokenizer.token_to_id(token) is None:
            tokens.append(tokenizer.token_to_id('[UNK]'))
        else:
            tokens.append(tokenizer.token_to_id(token))

    assert len(query) + len(context) + 3 == len(tokens)

    label_mask = [
        (0 if type_ids[token_idx] == 0 else 1)
        for token_idx in range(len(tokens) - 1)
    ]
    label_mask.append(0)

    # truncate
    tokens = tokens[:max_length]
    type_ids = type_ids[:max_length]
    label_mask = label_mask[:max_length]

    # make sure last token is [SEP]
    sep_token = tokenizer.token_to_id("[SEP]")
    if tokens[-1] != sep_token:
        assert len(tokens) == max_length
        tokens = tokens[: -1] + [sep_token]
        label_mask[-1] = 0

    if pad_to_maxlen:
        tokens = pad(tokens, 0, max_length)
        type_ids = pad(type_ids, 1, max_length)
        label_mask = pad(label_mask, value=0, max_length=max_length)

    assert len(label_mask) == len(tokens) == len(type_ids)

    return [
        torch.LongTensor([tokens]),
        torch.LongTensor([type_ids]),
        torch.LongTensor([label_mask]),
        # sample_idx,
        # label_idx
    ]


def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, tag, now_tokens_str):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of tuple (start, end)
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2)]
    """
    json_d = {}
    entities = []

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start + 1, tmp_end):
                    bmes_labels[i] = f"M-{tag}"
                # entities.append([tag, tmp_start - 11, tmp_end- 11, "".join(now_tokens_str[tmp_start:tmp_end + 1])])
                entities.append([tag, tmp_start - 11, tmp_end - 11])
            else:
                bmes_labels[tmp_end] = f"S-{tag}"
                # entities.append([tag, tmp_start - 11, tmp_end - 11, "".join(now_tokens_str[tmp_start:tmp_end + 1])])
                entities.append([tag, tmp_start - 11, tmp_end - 11])
    # json_d['id'] = id
    # json_d['entities'] = entities
    # return json_d
    return entities

if __name__ == '__main__':
    args = get_argparse().parse_args()
    bert_path = args.bert_config_dir
    json_path = args.data_dir
    is_chinese = True
    max_length = args.max_length
    query_json_file = './data/queries/ccf_ner.json'
    # query_json_file = './data/queries/msra_ner.json'
    tag2query = json.load(open(query_json_file, 'r', encoding='utf-8'))
    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join('./output', "best_f1_checkpoint")
    bert_config = BertQueryNerConfig.from_pretrained(output_dir,
                                                     hidden_dropout_prob=args.bert_dropout,
                                                     attention_probs_dropout_prob=args.bert_dropout,
                                                     mrc_dropout=args.mrc_dropout)
    model = BertQueryNER.from_pretrained(output_dir, config=bert_config).to(device)
    model.eval()

    # context = '中储粮在黑龙江收储的粮食达到1700多万吨，这里面包括大豆、玉米和水稻等。'
    # context = '从此在长滩岛旅游除了可以在沙滩收集贝壳，还可以来这里收集一张用真枪弹药射击的靶纸，博物馆应把握历史发展的规律，积极参与社会的变迁，充满活力地参与创造一个新的未来。'

    while True:
        context = input('text：')

        extract_info = []
        for tag in tag2query:
            query = tag2query[tag]
            print(tag)
            [tokens, token_type_ids, label_mask] = gen_input(query, context, tokenizer, max_length, pad_to_maxlen=False)
            attention_mask = (tokens != 0).long()
            tokens, attention_mask, token_type_ids, label_mask = tokens.to(device), attention_mask.to(
                    device), token_type_ids.to(device), label_mask.to(device)

            start_logits, end_logits, span_logits = model(input_ids=tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)

            start_logits, end_logits, span_logits = (start_logits > 0).long(), (end_logits > 0).long(), (span_logits > 0).long()

            start_logits, end_logits, span_logits, label_mask = start_logits.squeeze(dim=0).tolist(), \
                                                                    end_logits.squeeze(dim=0).tolist(), \
                                                                    span_logits.squeeze(dim=0).tolist(), \
                                                                    label_mask.squeeze(dim=0).tolist()
            seq_len = len(start_logits)
            now_tokens_str = ['[CLS]'] + list(query) + ['[SEP]'] + list(context) + ['[SEP]']
            now_tokens_str = now_tokens_str[:seq_len]
            now_tokens_str[-1] = ['[SEP]']

            assert len(now_tokens_str) == len(start_logits)

            entities = extract_flat_spans(start_logits, end_logits, span_logits, label_mask, tag, now_tokens_str)
            # print(entities)
            entity_list = []
            for item in entities:
                if len(item) != 0:
                    start = int(item[1])
                    end = int(item[2]) + 1
                    entity_list.append(context[start:end])
            extract_info.append({tag: entity_list})
        print(extract_info)


