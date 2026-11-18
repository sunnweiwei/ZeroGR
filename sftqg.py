import argparse
import importlib.util
import json
import logging
import os
import pickle
import shutil
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from file_io import *

try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
except ImportError:
    AutoLigerKernelForCausalLM = None

logging.getLogger("transformers").setLevel(logging.ERROR)


class LogMessage:
    def __init__(self, log_file, disable=False):
        self.log_file = log_file
        self.disable = disable
        if not self.disable and os.path.dirname(self.log_file):
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, message):
        if not self.disable:
            current_time = "[" + time.strftime("%H:%M:%S") + "]"
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(current_time + " " + message + "\n")
            print(current_time + " " + message)


def _to_id_list(enc):
    if isinstance(enc, list):
        return enc
    if hasattr(enc, "input_ids"):
        return list(enc.input_ids)
    if isinstance(enc, dict) and "input_ids" in enc:
        return list(enc["input_ids"])
    return list(enc)


def llama_encode(question, answer, tokenizer):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    ids = _to_id_list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    )
    q_ids = _to_id_list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    labels = [-100] * len(q_ids) + ids[len(q_ids) :]
    return ids, labels


def t5_encode(question, answer, tokenizer):
    ids = tokenizer.encode(question)
    labels = tokenizer.encode(answer)
    return ids, labels


class TrainGenData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        query = item["query"] if "query" in item else item["input"]
        doc = item["doc"] if "doc" in item else item["output"]

        query = self.tokenizer.decode(
            self.tokenizer.encode(query)[:128], skip_special_tokens=True
        )
        inst = item["instruction"] if "instruction" in item else ""
        inst = self.tokenizer.decode(
            self.tokenizer.encode(inst)[:128], skip_special_tokens=True
        )
        doc = self.tokenizer.decode(
            self.tokenizer.encode(doc)[:128], skip_special_tokens=True
        )

        prompt = f"You are a document to query generator. For the retrieval task: {inst}\n\nGenerate relevant search query for the following document:\n\n{doc}"
        answer = query

        if "t5" in self.tokenizer.name_or_path.lower():
            ids, labels = t5_encode(prompt, answer, self.tokenizer)
        else:
            ids, labels = llama_encode(prompt, answer, self.tokenizer)
        return torch.tensor(ids[:512]), torch.tensor(labels[:512])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "labels": labels}


def get_attn_impl():
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def main():
    parser = argparse.ArgumentParser()
    default_model = (
        "models/Llama-3.2-1B-Instruct"
        if os.path.exists("models/Llama-3.2-1B-Instruct")
        else "meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument("--model_name", type=str, default=default_model)
    parser.add_argument("--dataset_name", type=str, default="luxinyayaya/OpenInstIR")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--data_jsonl", type=str, default="")
    parser.add_argument("--save_path", type=str, default="models/Llama-3.2-1B-Instruct-qg")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator = Accelerator(gradient_accumulation_steps=1)

    logger = LogMessage("log/sftqg.log", disable=not accelerator.is_main_process)
    logger.log("#" * 20)
    logger.log("Query Generator SFT start")

    model_name = args.model_name
    save_path = args.save_path

    if args.data_jsonl:
        data = read_jsonl(args.data_jsonl)
        if args.max_samples > 0:
            data = data[: args.max_samples]
    else:
        # Memory-mapped Arrow loading via Hugging Face datasets (avoids loading millions of rows into RAM)
        data = load_dataset(args.dataset_name, args.dataset_config, split="train")
        if args.max_samples > 0:
            data = data.select(range(min(args.max_samples, len(data))))

    logger.log(save_path)
    logger.log(f"data {len(data)}")

    os.makedirs(save_path, exist_ok=True)

    if "Qwen" in model_name:
        model_type = "Qwen"
    elif "t5" in model_name.lower():
        model_type = "t5"
    else:
        model_type = "Llama"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    default_pads = {"Qwen": 151643, "Llama": 128004, "t5": 0}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            default_pads[model_type]
            if default_pads[model_type] < len(tokenizer)
            else tokenizer.eos_token_id
        )

    attn_impl = get_attn_impl()
    model_cls = AutoLigerKernelForCausalLM if AutoLigerKernelForCausalLM is not None else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False

    dataset = TrainGenData(data, tokenizer)
    accelerator.print(tokenizer.decode(dataset[0][0]))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps
    )
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    global_step = 0
    for epoch in range(args.epochs):
        logger.log(f"Training {epoch}")
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(
            data_loader, total=len(data_loader), disable=not accelerator.is_main_process
        )
        loss_report = []
        for batch in tk0:
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            loss_report.append(accelerator.gather(loss).mean().item())
            tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))
            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{save_path}/Epoch-{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(f"{save_path}/Epoch-{epoch}")
        accelerator.wait_for_everyone()
        if args.max_steps > 0 and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
