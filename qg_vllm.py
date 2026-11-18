import argparse
import json
import os
import random
import subprocess
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from file_io import *
from mair_config import SHARE_CORPUS


def get_doc_corpus_name(task):
    share_map = {v: k for k, values in SHARE_CORPUS.items() for v in values}
    return share_map.get(task, task)


def gen(
    docs_path,
    data_name,
    pid=0,
    total_num=1,
    model_sufix="QG",
    model_name=None,
    num_q=16,
    max_docs=0,
):
    if not docs_path:
        doc_corpus = get_doc_corpus_name(data_name)
        docs_path = f"dataset/MAIR-Docs/{doc_corpus}/docs.jsonl"
    if not os.path.exists(docs_path):
        ensure_mair_task(data_name)

    docs = read_jsonl(docs_path)
    if max_docs > 0:
        docs = docs[:max_docs]
    docs = [docs[i] for i in range(pid, len(docs), total_num)]

    query_path = f"dataset/MAIR-Queries/{data_name}/queries.jsonl"
    if not os.path.exists(query_path):
        ensure_mair_task(data_name)
    queries = read_jsonl(query_path)
    inst = queries[0].get("instruction", "Retrieve document relevant to the given query.")
    inst_last = queries[-1].get("instruction", inst)

    if inst != inst_last:
        inst = "Retrieve document relevant to the given query."

    llm = LLM(
        model=model_name,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.5,
        max_model_len=1024,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []
    for doc in docs:
        doc_text = " ".join(doc["doc"].split()[:128])
        doc_text = tokenizer.decode(
            tokenizer.encode(doc_text, add_special_tokens=False)[:300],
            skip_special_tokens=True,
        )
        prompt = f"You are a document to query generator. For the retrieval task: {inst}\n\nGenerate relevant search query for the following document:\n\n{doc_text}"
        item = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(item)

    sampling_params = SamplingParams(n=num_q, temperature=1.0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)
    outputs = [[x.text.strip() for x in output.outputs] for output in outputs]

    qg_data = []
    for item, out in zip(docs, outputs):
        doc_id = str(item["id"])
        for query in out:
            qg_data.append(
                {
                    "instruction": inst,
                    "query": query,
                    "labels": [{"id": doc_id, "score": 1}],
                }
            )

    doc_corpus = get_doc_corpus_name(data_name)
    shard_path = f"dataset/MAIR-Data/{model_sufix}-{num_q}/{doc_corpus}/queries-{pid}-{total_num}.jsonl"
    write_jsonl(qg_data, shard_path)
    if total_num == 1 and pid == 0:
        merged_path = f"dataset/MAIR-Data/{model_sufix}-{num_q}/{doc_corpus}/queries.jsonl"
        write_jsonl(qg_data, merged_path)
        print(f"Saved {len(qg_data)} generated queries to {merged_path}")


def main():
    default_model = (
        "models/Llama-3.2-1B-Instruct-qg"
        if os.path.exists("models/Llama-3.2-1B-Instruct-qg")
        else "luxinyayaya/ZeroGR-Query-Generator"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_name", type=str, default="ArguAna")
    parser.add_argument("-docs_path", type=str, default="")
    parser.add_argument("-pid", type=int, default=0)
    parser.add_argument("-total_num", type=int, default=0)
    parser.add_argument("-model_sufix", type=str, default="QG")
    parser.add_argument("-model_name", type=str, default=default_model)
    parser.add_argument("-num_q", type=int, default=16)
    parser.add_argument("-max_docs", type=int, default=0)
    args = parser.parse_args()

    if args.total_num > 0:
        gen(
            args.docs_path,
            args.data_name,
            pid=args.pid,
            total_num=args.total_num,
            model_sufix=args.model_sufix,
            model_name=args.model_name,
            num_q=args.num_q,
            max_docs=args.max_docs,
        )
        return

    num_gpus = max(torch.cuda.device_count(), 1)
    total_num = num_gpus
    num_q = args.num_q
    tasks = [t.strip() for t in args.data_name.split(",") if t.strip()]
    os.makedirs("log", exist_ok=True)

    for name in tasks:
        print(f"Running Query Generator on task: {name} (workers={total_num})")
        ensure_mair_task(name)
        doc_corpus = get_doc_corpus_name(name)
        model_sufix = args.model_sufix
        model_name = args.model_name
        docs_path = f"dataset/MAIR-Docs/{doc_corpus}/docs.jsonl"
        merged_out = f"dataset/MAIR-Data/{model_sufix}-{num_q}/{doc_corpus}/queries.jsonl"
        if os.path.exists(merged_out):
            print(f"Already exists: {merged_out}")
            continue
        if total_num == 1:
            gen(
                docs_path,
                name,
                pid=0,
                total_num=1,
                model_sufix=model_sufix,
                model_name=model_name,
                num_q=num_q,
                max_docs=args.max_docs,
            )
            continue

        processes = []
        for i in range(total_num):
            gpu_id = i % num_gpus
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu_id} {sys.executable} qg_vllm.py "
                f"-pid {i} -total_num {total_num} -docs_path {docs_path} -data_name {name} "
                f"-model_sufix {model_sufix} -model_name {model_name} -num_q {num_q} -max_docs {args.max_docs}"
            )
            print(cmd)
            with open(f"log/qg_log{i}", "w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT
                )
                processes.append(process)
        for process in processes:
            process.wait()
        data = []
        for pid in range(total_num):
            sub_path = f"dataset/MAIR-Data/{model_sufix}-{num_q}/{doc_corpus}/queries-{pid}-{total_num}.jsonl"
            if not os.path.exists(sub_path):
                continue
            sub = read_jsonl(sub_path)
            print(pid, len(sub))
            data.extend(sub)
        if len(data) > 0:
            write_jsonl(data, merged_out)
            print(merged_out)


if __name__ == "__main__":
    main()
