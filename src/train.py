# -*- encoding: utf-8 -*-
'''
@create_time: 2023/07/25 10:04:06
@author: lichunyu
'''
import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import deepspeed
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beam search")
    parser.add_argument("--synced_gpus", type=bool, default=False, help="True if zero3 enabled")
    parser.add_argument("--do_sample", type=bool, default=True, help="")
    parser.add_argument("--top_p", type=float, default=0.7, help="")
    parser.add_argument("--temperature", type=float, default=0.95, help="")
    parser.add_argument("--train_file_path", type=str, default="../data/data_example.jsonl", help="")
    parser.add_argument("--eval_file_path", type=str, default="../data/data_example.jsonl", help="")
    parser.add_argument("--num_epochs", type=int, default=8, help="")
    parser.add_argument("--train_batch_size_per_gpu", type=int, default=2, help="")
    parser.add_argument("--eval_batch_size_per_gpu", type=int, default=2, help="")
    parser.add_argument("--max_q_length", type=int, default=100, help="")
    parser.add_argument("--max_a_length", type=int, default=500, help="")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default="/media/E/lichunyu/models/pretrained_models/chatglm2-6b",
                        help="")
    parser.add_argument("--tokenizer_name_or_path",
                        type=str,
                        default="/media/E/lichunyu/models/pretrained_models/chatglm2-6b",
                        help="")
    parser.add_argument("--lora_r", type=int, default=4, help="")
    parser.add_argument("--lora_alpha", type=int, default=8, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")
    parser.add_argument("--lora_target_modules", type=str, default="query_key_value", help="")
    parser.add_argument("--tensorboard_dir", type=str, default="../data/output_dir/logs/", help="")
    parser.add_argument("--tensorboard_project_name", type=str, default="finetuning", help="")
    parser = deepspeed.add_config_arguments(parser=parser)
    args = parser.parse_args()

    project_dir = Path(os.path.join(args.tensorboard_dir, args.tensorboard_project_name))
    project_dir.mkdir(parents=True, exist_ok=True)

    deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()
    args.max_length = args.max_q_length + args.max_a_length + 1

    writer = SummaryWriter(log_dir=project_dir)

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             target_modules=[args.lora_target_modules],
                             fan_in_fan_out=False,
                             bias="all")
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, revision="main", use_cache=True).half()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, dist_init_required=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                              trust_remote_code=True,
                                              revision="main",
                                              padding_side="left")
    train_dataset = CausalDataset(file_path=args.train_file_path,
                                  tokenizer=tokenizer,
                                  max_q_length=args.max_q_length,
                                  max_a_length=args.max_a_length)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size_per_gpu,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                    model=model,
                                                                    label_pad_token_id=-100,
                                                                    pad_to_multiple_of=None,
                                                                    padding=True))

    eval_train_dataset = GenerationDataset(file_path=args.train_file_path,
                                           tokenizer=tokenizer,
                                           max_q_length=args.max_q_length,
                                           max_a_length=args.max_a_length)
    eval_train_sampler = DistributedSampler(eval_train_dataset, shuffle=False)
    eval_train_dataloader = DataLoader(eval_train_dataset,
                                       batch_size=args.eval_batch_size_per_gpu,
                                       sampler=eval_train_sampler,
                                       drop_last=True,
                                       collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                         model=model,
                                                                         label_pad_token_id=-100,
                                                                         pad_to_multiple_of=None,
                                                                         padding=True))

    eval_dataset = GenerationDataset(file_path=args.eval_file_path,
                                     tokenizer=tokenizer,
                                     max_q_length=args.max_q_length,
                                     max_a_length=args.max_a_length)
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.eval_batch_size_per_gpu,
                                 sampler=eval_sampler,
                                 drop_last=True,
                                 collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                   model=model,
                                                                   label_pad_token_id=-100,
                                                                   pad_to_multiple_of=None,
                                                                   padding=True))

    for epoch in range(args.num_epochs):

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model.local_rank) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            if args.global_rank <= 0:
                # TODO reduce loss
                writer.add_scalar("Train/Loss", loss.item(), len(train_dataloader) * epoch + step)
            model.backward(loss)
            model.step()

        model.eval()
        eval_train_rouge_1, eval_train_rouge_2, eval_train_rouge_l, eval_train_bleu_4 = torch.tensor(0.0).to(
            model.local_rank), torch.tensor(0.0).to(model.local_rank), torch.tensor(0.0).to(
                model.local_rank), torch.tensor(0.0).to(model.local_rank)
        for step, batch in enumerate(eval_train_dataloader):
            batch = {k: v.to(model.local_rank) for k, v in batch.items()}
            gen_kwargs = get_gen_kwargs(inputs=batch, args=args)
            generated_tokens = model.generate(**gen_kwargs)
            generated_tokens = generated_tokens[:, gen_kwargs["input_ids"].size()[-1]:]
            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels_text = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            metrics = get_eval_metrics(generated_text, labels_text)
            eval_train_rouge_1 += torch.tensor(metrics["rouge-1"]).to(model.local_rank)
            eval_train_rouge_2 += torch.tensor(metrics["rouge-2"]).to(model.local_rank)
            eval_train_rouge_l += torch.tensor(metrics["rouge-l"]).to(model.local_rank)
            eval_train_bleu_4 += torch.tensor(metrics["bleu_4"]).to(model.local_rank)
        eval_train_rouge_1 /= step + 1
        eval_train_rouge_2 /= step + 1
        eval_train_rouge_l /= step + 1
        eval_train_bleu_4 /= step + 1
        torch.distributed.reduce(eval_train_rouge_1, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(eval_train_rouge_2, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(eval_train_rouge_l, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(eval_train_bleu_4, dst=0, op=torch.distributed.ReduceOp.SUM)
        if args.global_rank <= 0:
            eval_train_rouge_1 /= torch.distributed.get_world_size()
            eval_train_rouge_2 /= torch.distributed.get_world_size()
            eval_train_rouge_l /= torch.distributed.get_world_size()
            eval_train_bleu_4 /= torch.distributed.get_world_size()
            writer.add_scalar("Train/ROUGE-1", eval_train_rouge_1.item(), epoch)
            writer.add_scalar("Train/ROUGE-2", eval_train_rouge_2.item(), epoch)
            writer.add_scalar("Train/ROUGE-l", eval_train_rouge_l.item(), epoch)
            writer.add_scalar("Train/BLEU-4", eval_train_bleu_4.item(), epoch)
            # print(f"eval_train_rouge_1: {eval_train_rouge_1}")
            # print(f"eval_train_rouge_2: {eval_train_rouge_2}")
            # print(f"eval_train_rouge_l: {eval_train_rouge_l}")
            # print(f"eval_train_bleu_4: {eval_train_bleu_4}")

        model.eval()
        eval_rouge_1, eval_rouge_2, eval_rouge_l, eval_bleu_4 = torch.tensor(0.0).to(model.local_rank), torch.tensor(0.0).to(
            model.local_rank), torch.tensor(0.0).to(model.local_rank), torch.tensor(0.0).to(model.local_rank)
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(model.local_rank) for k, v in batch.items()}
            gen_kwargs = get_gen_kwargs(inputs=batch, args=args)
            generated_tokens = model.generate(**gen_kwargs)
            generated_tokens = generated_tokens[:, gen_kwargs["input_ids"].size()[-1]:]
            generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels_text = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            metrics = get_eval_metrics(generated_text, labels_text)
            eval_rouge_1 += torch.tensor(metrics["rouge-1"]).to(model.local_rank)
            eval_rouge_2 += torch.tensor(metrics["rouge-2"]).to(model.local_rank)
            eval_rouge_l += torch.tensor(metrics["rouge-l"]).to(model.local_rank)
            eval_bleu_4 += torch.tensor(metrics["bleu_4"]).to(model.local_rank)
        eval_rouge_1 /= step + 1
        eval_rouge_2 /= step + 1
        eval_rouge_l /= step + 1
        eval_bleu_4 /= step + 1
        torch.distributed.reduce(eval_rouge_1, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(eval_rouge_2, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(eval_rouge_l, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(eval_bleu_4, dst=0, op=torch.distributed.ReduceOp.SUM)
        if args.global_rank <= 0:
            eval_rouge_1 /= torch.distributed.get_world_size()
            eval_rouge_2 /= torch.distributed.get_world_size()
            eval_rouge_l /= torch.distributed.get_world_size()
            eval_bleu_4 /= torch.distributed.get_world_size()
            writer.add_scalar("Eval/ROUGE-1", eval_rouge_1.item(), epoch)
            writer.add_scalar("Eval/ROUGE-2", eval_rouge_2.item(), epoch)
            writer.add_scalar("Eval/ROUGE-l", eval_rouge_l.item(), epoch)
            writer.add_scalar("Eval/BLEU-4", eval_bleu_4.item(), epoch)
            # print(f"eval_rouge_1: {eval_rouge_1}")
            # print(f"eval_rouge_2: {eval_rouge_2}")
            # print(f"eval_rouge_l: {eval_rouge_l}")
            # print(f"eval_bleu_4: {eval_bleu_4}")


if __name__ == "__main__":
    main()
