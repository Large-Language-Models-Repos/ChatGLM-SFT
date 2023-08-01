# -*- encoding: utf-8 -*-
'''
@create_time: 2023/07/25 10:04:06
@author: lichunyu
'''
import argparse
from dataclasses import dataclass, field

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
)

from utils.data import get_data
from utils.generic import get_eval_metrics, get_gen_kwargs


class CausalDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_q_length=100,
        max_a_length=50,
    ) -> None:
        super().__init__()
        self.data = get_data(file_path=file_path)
        self.tokenizer = tokenizer
        self.max_q_length = max_q_length
        self.max_a_length = max_a_length
        self.max_seq_length = max_q_length + max_a_length + 1

    def __getitem__(self, index) -> T_co:
        q = self.data[index]["q"]
        a = self.data[index]["a"]
        history = self.data[index]["history"]
        prompt = self.tokenizer.build_prompt(q, history)
        q_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True, max_length=self.max_q_length)
        a_ids = self.tokenizer.encode(text=a, add_special_tokens=False, truncation=True, max_length=self.max_a_length)
        context_length = len(q_ids)
        input_ids = q_ids + a_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + a_ids + [self.tokenizer.eos_token_id]
        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(i if i != self.tokenizer.pad_token_id else -100) for i in labels]

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.data)


class GenerationDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_q_length=100,
        max_a_length=50,
    ) -> None:
        self.data = get_data(file_path=file_path)
        self.tokenizer = tokenizer
        self.max_q_length = max_q_length
        self.max_a_length = max_a_length
        self.max_seq_length = max_q_length + max_a_length + 1

    def __getitem__(self, index):
        q = self.data[index]["q"]
        a = self.data[index]["a"]
        history = self.data[index]["history"]
        prompt = self.tokenizer.build_prompt(q, history)
        inputs = self.tokenizer(prompt, max_length=self.max_q_length, truncation=True, padding=True)
        labels = self.tokenizer(text_target=a, max_length=self.max_a_length, truncation=True)
        labels["input_ids"] = [i if i != self.tokenizer.pad_token_id else -100 for i in labels["input_ids"]]
        inputs["labels"] = labels["input_ids"]
        return inputs

    def __len__(self):
        return len(self.data)


@dataclass
class CausalCollator:

    tokenizer: PreTrainedTokenizerBase = field(default=None, metadata={"help": "tokenizer of the LLM"})

    def __call__(self, batch):
        max_length = max([i["input_ids"].size(-1) for i in batch])
        input_ids = pad_sequence([i["input_ids"] for i in batch],
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id if self.tokenizer is not None else 3).int()
        attention_mask = torch.stack([
            F.pad(i["attention_mask"],
                  (0, max_length - i["attention_mask"].size(-1), 0, max_length - i["attention_mask"].size(-1)),
                  value=False) for i in batch
        ])
        position_ids = torch.stack(
            [F.pad(i["position_ids"], (0, max_length - i["position_ids"].size(-1)), mode="replicate") for i in batch]).int()
        labels = pad_sequence([i["labels"] for i in batch], batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, "labels": labels}


def main():

    # TODO collect with a dataclass
    model_name_or_path = "/media/E/lichunyu/models/pretrained_models/chatglm2-6b"  # change to `chatglm-6b` without local weights
    tokenizer_name_or_path = "/media/E/lichunyu/models/pretrained_models/chatglm2-6b"  # change to `chatglm-6b` without local weights
    file_path = "../data/data_example.jsonl"
    max_q_length = 100
    max_a_length = 500
    max_length = max_q_length + max_a_length + 1
    batch_size = 1
    # learning_rate = 1e-4
    # accumulate_step = 1
    # num_epochs = 100
    # num_warmup_steps = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beam search")
    parser.add_argument("--synced_gpus", type=bool, default=False, help="True if zero3 enabled")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--top_p", type=float, default=0.7, help="")
    parser.add_argument("--temperature", type=float, default=0.95, help="")
    parser = deepspeed.add_config_arguments(parser=parser)
    args = parser.parse_args()
    deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=4,
                             lora_alpha=8,
                             lora_dropout=0.1,
                             target_modules=["query_key_value"],
                             fan_in_fan_out=False,
                             bias="all")
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, revision="main").half()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, dist_init_required=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,
                                              trust_remote_code=True,
                                              revision="main",
                                              padding_side="left")
    train_dataset = CausalDataset(file_path=file_path,
                                  tokenizer=tokenizer,
                                  max_q_length=max_q_length,
                                  max_a_length=max_a_length)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                    model=model,
                                                                    label_pad_token_id=-100,
                                                                    pad_to_multiple_of=None,
                                                                    padding=True))

    eval_dataset = GenerationDataset(file_path=file_path,
                                     tokenizer=tokenizer,
                                     max_q_length=max_q_length,
                                     max_a_length=max_a_length)
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=batch_size,
                                 sampler=eval_sampler,
                                 collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                   model=model,
                                                                   label_pad_token_id=-100,
                                                                   pad_to_multiple_of=None,
                                                                   padding=True))

    for epoch in range(4):

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model.local_rank) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            model.backward(loss)
            model.step()

        model.eval()
        eval_metrics = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu@4': []}
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(model.local_rank) for k, v in batch.items()}
            gen_kwargs = get_gen_kwargs(inputs=batch, args=args)
            gen_kwargs["max_length"] = max_length
            generated_tokens = model.generate(**gen_kwargs)
            generated_tokens = generated_tokens[:, gen_kwargs["input_ids"].size()[-1]:]
            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels_text = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            metrics = get_eval_metrics(generated_text, labels_text)
            for k, v in metrics.items():
                eval_metrics[k].append(v)
        for k, v in eval_metrics.items():
            eval_metrics[k] = np.mean(v)
        print(eval_metrics)
        ...

    print("success")


if __name__ == "__main__":
    main()
