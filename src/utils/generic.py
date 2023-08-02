# -*- encoding: utf-8 -*-
'''
@create_time: 2023/08/01 17:46:02
@author: lichunyu
'''
import argparse
from typing import Dict, List

import jieba
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from transformers import PreTrainedTokenizer


def get_gen_kwargs(inputs: dict, args: argparse.ArgumentParser) -> dict:
    gen_kwargs = {}
    gen_kwargs["num_beams"] = args.num_beams
    gen_kwargs["synced_gpus"] = args.synced_gpus
    gen_kwargs["input_ids"] = inputs["input_ids"]
    gen_kwargs["attention_mask"] = inputs["attention_mask"]
    gen_kwargs["position_ids"] = inputs["position_ids"]
    gen_kwargs["top_p"] = args.top_p
    gen_kwargs["do_sample"] = args.do_sample
    gen_kwargs["temperature"] = args.temperature
    gen_kwargs["max_length"] = args.max_length
    return gen_kwargs


def get_eval_metrics(source: List[str], target: List[str]) -> Dict[str, float]:
    scores = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu_4": []}
    for s, t in zip(source, target):
        hypothesis = list(jieba.cut(s))
        reference = list(jieba.cut(t))
        rouge = Rouge()
        result = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
        for k, v in result.items():
            scores[k].append(v["f"])
        bleu_score = sentence_bleu([list(t)], list(s), smoothing_function=SmoothingFunction().method3)
        scores["bleu_4"].append(bleu_score)
    for k, v in scores.items():
        scores[k] = float(np.mean(v))
    return scores


def tensor_pad(tensor: torch.Tensor, tokenizer: PreTrainedTokenizer, max_length: int):
    padded_tensor = tokenizer.pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :tensor.shape[-1]] = tensor
    return padded_tensor
