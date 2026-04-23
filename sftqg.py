import json
from torch.utils.data import Dataset
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch
from torch.optim import AdamW
import time
from torch.utils.data import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import os
import numpy as np
from file_io import *
import pickle
import shutil
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


def write_pkl(obj, filename):
    dirname = "/".join(filename.split("/")[:-1])
    os.makedirs(dirname, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class LogMessage:
    def __init__(self, log_file, disable=False):
        self.log_file = log_file
        self.disable = disable

    def log(self, message):
        if not self.disable:
            current_time = "[" + time.strftime("%H:%M:%S") + "]"
            with open(self.log_file, "a") as file:
                file.write(current_time + " " + message + "\n")
            print(current_time + " " + message)


def format_number(num):
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num / 1_000:.3g}K"
    elif num < 1_000_000_000:
        return f"{num / 1_000_000:.3g}M"
    elif num < 1_000_000_000_000:
        return f"{num / 1_000_000_000:.3g}B"
    else:
        return f"{num / 1_000_000_000_000:.3g}T"


def llama_encode(question, answer, tokenizer):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    q_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
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
        query = self.data[index]["query"]
        doc = self.data[index]["doc"]

        query = self.tokenizer.decode(
            self.tokenizer.encode(query)[:128], skip_special_tokens=True
        )
        inst = (
            self.data[index]["instruction"] if "instruction" in self.data[index] else ""
        )
        inst = self.tokenizer.decode(
            self.tokenizer.encode(inst)[:128], skip_special_tokens=True
        )
        doc = self.tokenizer.decode(
            self.tokenizer.encode(doc)[:128], skip_special_tokens=True
        )

        prompt = f"You are a document to query generator. For the retrieval task: {inst}\n\nGenerate relevant search query for the following document:\n\n{doc}"
        answer = query

        if "t5" in self.tokenizer.name_or_path:
            ids, labels = t5_encode(prompt, answer, self.tokenizer)
        else:
            ids, labels = llama_encode(prompt, answer, self.tokenizer)
        # ids = ids + [0] * 512
        # labels = labels + [-100] * 512
        return torch.tensor(ids[:512]), torch.tensor(labels[:512])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        features = {"input_ids": input_ids, "labels": labels}
        return features


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    accelerator = Accelerator(gradient_accumulation_steps=1)

    logger = LogMessage("log/dsi-pt.log", disable=not accelerator.is_main_process)
    logger.log("")
    logger.log("#" * 20)
    logger.log("Pre train start")
    logger.log("Token token")

    # model_name = 'models/Qwen2.5-0.5B-Instruct-PT/Epoch-2'
    model_name = "models/Llama-3.2-1B-Instruct"
    save_path = "models/Llama-3.2-1B-Instruct-qg"
    data = []

    logger.log(save_path)
    logger.log(f"data {len(data)}")

    batch_size = 32
    lr = 5e-5
    num_warmup_steps = 1000

    os.makedirs(save_path, exist_ok=True)

    if "Qwen" in model_name:
        model_type = "Qwen"
    elif "t5" in model_name:
        model_type = "t5"
    else:
        model_type = "Llama"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = {"Qwen": 151643, "Llama": 128004, "t5": 0}[model_type]
    tokenizer.add_tokens(["<|g|>", "<|/g|>"] + [f"<|g{i}|>" for i in range(20000)])
    first_token = {"Qwen": 151643, "Llama": 128000, "t5": 0}[model_type]
    os.system("sh kill_gpu.sh")
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    # model.gradient_checkpointing_enable()

    config = model.config

    dataset = TrainGenData(data, tokenizer)
    # dataset = TrainGenData(data, docs, tokenizer, cand_num=2)
    accelerator.print(tokenizer.decode(dataset[0][0]))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
    )

    optimizer = AdamW(model.parameters(), lr=lr)

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps
    )
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    os.system("sh kill_gpu.sh")
    loss_fct = CrossEntropyLoss()
    for epoch in range(10):
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
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{save_path}/Epoch-{epoch}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        unwrapped_tok = AutoTokenizer.from_pretrained(model_name)
        unwrapped_tok.save_pretrained(f"{save_path}/Epoch-{epoch}")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
