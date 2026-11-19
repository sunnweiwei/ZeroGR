import os
import json
from tqdm import tqdm
import numpy as np
import re
import time
import datetime
import pickle


def mp(func, data, processes=20, **kwargs):
    from torch.multiprocessing import multiprocessing
    import copy

    pool = multiprocessing.Pool(processes=processes)
    length = len(data) // processes + 1
    results = []
    for ids in range(processes):
        collect = data[ids * length : (ids + 1) * length]
        kwargs["seed"] = ids
        results.append(
            pool.apply_async(func, args=(collect,), kwds=copy.deepcopy(kwargs))
        )
    pool.close()
    pool.join()
    result_collect = []
    for j, res in enumerate(results):
        result = res.get()
        result_collect.extend(result)
    return result_collect


def all_file(dirname):
    fl = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            path = os.path.join(root, item)
            fl.append(path)
    return fl


def read_file(file):
    out = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            out.append(line)
    return out


def read_json(file):
    if not os.path.exists(file):
        return {}
    with open(file, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(file):
    out = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def create_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def write_json(obj, file):
    create_dir(file)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def write_jsonl(obj, file):
    create_dir(file)
    with open(file, "w", encoding="utf-8") as f:
        for line in obj:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def add_jsonl(obj, file):
    create_dir(file)
    with open(file, "a", encoding="utf-8") as f:
        for line in obj:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_pkl(obj, file):
    create_dir(file)
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def read_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def ensure_mair_task(task):
    """Ensure dataset/MAIR-Docs/<task>/docs.jsonl and dataset/MAIR-Queries/<task>/queries.jsonl exist.
    Automatically downloads from Hugging Face MAIR-Bench if missing."""
    from mair_config import SHARE_CORPUS

    share_map = {v: k for k, values in SHARE_CORPUS.items() for v in values}
    doc_name = share_map.get(task, task)

    docs_file = f"dataset/MAIR-Docs/{doc_name}/docs.jsonl"
    task_docs_file = f"dataset/MAIR-Docs/{task}/docs.jsonl"
    queries_file = f"dataset/MAIR-Queries/{task}/queries.jsonl"

    if not os.path.exists(docs_file):
        from datasets import load_dataset

        ds_docs = load_dataset("MAIR-Bench/MAIR-Docs", doc_name)
        split_name = "docs" if "docs" in ds_docs else list(ds_docs.keys())[0]
        write_jsonl(ds_docs[split_name], docs_file)

    if doc_name != task and not os.path.exists(task_docs_file):
        create_dir(task_docs_file)
        os.symlink(os.path.abspath(docs_file), task_docs_file)

    if not os.path.exists(queries_file):
        from datasets import load_dataset

        ds_q = load_dataset("MAIR-Bench/MAIR-Queries", task)
        split_name = "queries" if "queries" in ds_q else list(ds_q.keys())[0]
        write_jsonl(ds_q[split_name], queries_file)

    return docs_file, queries_file
