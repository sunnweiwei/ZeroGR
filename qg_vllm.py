import os
import json
import time
import torch
import random
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
from vllm import LLM, SamplingParams
from file_io import *
import subprocess


def gen(
    docs_path,
    data_name,
    pid=1,
    total_num=16,
    model_sufix="D2Q",
    model_name=None,
    num_q=16,
):
    dist.init_process_group(init_method="env://", rank=0, world_size=1)
    model_path = model_name
    docs = read_jsonl(docs_path)
    docs = [docs[i] for i in range(pid, len(docs), total_num)]

    query_path = docs_path.replace("MAIR-Docs", "MAIR-Queries").replace(
        "docs.jsonl", "queries.jsonl"
    )
    inst = read_jsonl(query_path)[0]["instruction"]
    inst_last = read_jsonl(query_path)[-1]["instruction"]

    if inst != inst_last:
        inst = "Retrieve document relevant to the given query."

    llm = LLM(
        model=model_path,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.3,
        max_model_len=1024,
        max_seq_len_to_capture=1024,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = []
    for doc in docs:
        doc = " ".join(doc["doc"].split()[:128])
        doc = tokenizer.decode(tokenizer.encode(doc)[:300])
        prompt = f"You are a document to query generator. For the retrieval task: {inst}\n\nGenerate relevant search query for the following document:\n\n{doc}"
        item = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(item)

    os.system("sh kill_gpu.sh")
    sampling_params = SamplingParams(n=num_q, temperature=1, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)
    outputs = [[x.text for x in output.outputs] for output in outputs]

    qg_data = []
    for item, out in zip(docs, outputs):
        doc_id = item["id"]
        for query in out:
            qg_data.append(
                {
                    "instruction": inst,
                    "query": query,
                    "labels": [{"id": doc_id, "score": 1}],
                }
            )
    write_jsonl(
        qg_data,
        f"dataset/MAIR-Data/{model_sufix}-{num_q}/{data_name}/queries-{pid}-{total_num}.jsonl",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_name", type=str, default="")
    parser.add_argument("-docs_path", type=str, default="")
    parser.add_argument("-pid", type=int, default=0)
    parser.add_argument("-total_num", type=int, default=0)
    parser.add_argument("-model_sufix", type=str, default="D2Q")
    parser.add_argument("-model_name", type=str, default="D2Q")
    parser.add_argument("-num_q", type=int, default=16)
    args = parser.parse_args()
    docs_path = args.docs_path
    data_name = args.data_name
    pid = args.pid
    total_num = args.total_num
    model_sufix = args.model_sufix
    model_name = args.model_name
    num_q = args.num_q

    if total_num > 0:
        gen(
            docs_path,
            data_name,
            pid=pid,
            total_num=total_num,
            model_sufix=model_sufix,
            model_name=model_name,
            num_q=num_q,
        )
        return

    total_num = 8
    num_q = 16
    tasks = []
    for name in tasks:
        print(name)
        model_sufix = "QG"
        model_name = "models/Llama-3.2-1B-Instruct-qg"
        docs_path = f"dataset/MAIR-Docs/{name}/docs.jsonl"
        if not os.path.exists(docs_path):
            continue
        if os.path.exists(
            f"dataset/MAIR-Data/{model_sufix}-{num_q}/{name}/queries.jsonl"
        ):
            print("exists")
            continue
        processes = []
        for i in range(total_num):
            gpu_id = i % 4
            port = i + 21400
            cmd = (
                f"export CUDA_VISIBLE_DEVICES={gpu_id}; export MASTER_PORT={port}; export MASTER_ADDR='localhost'; export RANK=0; "
                f"python qg_vllm.py "
                f"-pid {i} -total_num {total_num} -docs_path {docs_path} -data_name {name} -model_sufix {model_sufix} -model_name {model_name} -num_q {num_q}"
            )
            print(cmd)
            with open(f"log/log{i}", "w") as log_file:
                process = subprocess.Popen(
                    cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT
                )
                processes.append(process)
        for process in processes:
            process.wait()
        data = []
        for pid in range(total_num):
            sub_path = f"dataset/MAIR-Data/{model_sufix}-{num_q}/{name}/queries-{pid}-{total_num}.jsonl"
            if not os.path.exists(sub_path):
                continue
            sub = read_jsonl(sub_path)
            print(pid, len(sub))
            data.extend(sub)
        if len(data) > 0:
            write_jsonl(
                data, f"dataset/MAIR-Data/{model_sufix}-{num_q}/{name}/queries.jsonl"
            )
            print(f"dataset/MAIR-Data/{model_sufix}-{num_q}/{name}/queries.jsonl")


if __name__ == "__main__":
    main()
