# -*- encoding: utf-8 -*-
'''
@create_time: 2023/07/25 10:04:06
@author: lichunyu
'''
import argparse
from dataclasses import dataclass, field

import deepspeed
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
)

from utils.data import get_data


class CausalDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer_name_or_path: str,
        max_q_length=100,
        max_a_length=50,
    ) -> None:
        super().__init__()
        self.data = get_data(file_path=file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, revision="main")
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
    batch_size = 1
    # learning_rate = 1e-4
    # accumulate_step = 1
    # num_epochs = 100
    # num_warmup_steps = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, help="deepspeed config")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=4,
                             lora_alpha=32,
                             lora_dropout=0.1,
                             target_modules=["query_key_value"],
                             fan_in_fan_out=False,
                             bias="all")

    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, revision="main").half()
    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model)
    train_dataset = CausalDataset(file_path=file_path,
                                  tokenizer_name_or_path=tokenizer_name_or_path,
                                  max_q_length=max_q_length,
                                  max_a_length=max_a_length)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, revision="main")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                    model=model,
                                                                    label_pad_token_id=-100,
                                                                    pad_to_multiple_of=None,
                                                                    padding=True))

    for step, batch in enumerate(train_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        output = model_engine(**batch)
        loss = output.loss
        model_engine.backward(loss)
        model_engine.step()
        print("step success")

    print("success")


if __name__ == "__main__":
    main()
