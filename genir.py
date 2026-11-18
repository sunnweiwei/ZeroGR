import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import copy
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
from collections import defaultdict
from itertools import product
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adafactor, AdamW
from torch.utils.data import Dataset, DistributedSampler
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

from file_io import (
    ensure_mair_task,
    read_json,
    read_jsonl,
    read_pkl,
    write_json,
    write_jsonl,
    write_pkl,
)
from mair_config import SHARE_CORPUS

sys.setrecursionlimit(5000)


def get_doc_corpus_name(task):
    share_map = {v: k for k, values in SHARE_CORPUS.items() for v in values}
    return share_map.get(task, task)


def get_attn_impl():
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


class newTree:
    def __init__(self, start_id, end_id, tokenizer):
        self.node_count = 1
        self.end_id = end_id
        self.start_id = start_id
        self.node_list = dict()
        self.node2doc = dict()
        self.root = self.node_count
        self.node_list[self.root] = dict()
        self.id_map = dict()
        self.tokenizer = tokenizer

    def id2node(self, id):
        return self.node_list[id]

    def add_node(self):
        self.node_count += 1
        self.node_list[self.node_count] = dict()
        return self.node_count

    def del_node(self, id):
        def dfs(id):
            pointer = self.id2node(id)
            for k, v in pointer.items():
                dfs(v)
            del self.node_list[id]
            if id in self.node2doc:
                del self.node2doc[id]

        dfs(id)

    def show(self):
        results = []

        def dfs(id, path):
            pointer = self.id2node(id)
            if len(pointer) == 0:
                results.append(path)
                return
            for k, v in pointer.items():
                dfs(v, path + [k])

        dfs(self.root, [])
        return results

    def set(self, path, id):
        now_id = self.root
        for i in path:
            pointer = self.id2node(now_id)
            if i not in pointer:
                pointer[i] = self.add_node()
            now_id = pointer[i]
        if now_id not in self.node2doc:
            self.node2doc[now_id] = []
        self.node2doc[now_id].append(id)

    def set_short(self, path, ids):
        now_id = self.root
        for i in path:
            pointer = self.id2node(now_id)
            if i not in pointer:
                pointer[i] = self.add_node()
                now_id = pointer[i]
                if i == self.end_id:
                    break
                pointer = self.id2node(now_id)
                pointer[self.end_id] = self.add_node()
                now_id = pointer[self.end_id]
                break
            now_id = pointer[i]
        if now_id not in self.node2doc:
            self.node2doc[now_id] = []
        self.node2doc[now_id].extend(ids)

    def remove(self, path):
        def dfs(id, path):
            if len(path) == 0:
                return True
            pointer = self.id2node(id)
            if path[0] not in pointer:
                return True
            temp = dfs(pointer[path[0]], path[1:])
            if (id == self.root or len(pointer) > 1) and temp:
                self.del_node(pointer[path[0]])
                del pointer[path[0]]
                temp = False
            return temp

        dfs(self.root, path)

    def get(self, path):
        now_id = self.root
        for i in path:
            pointer = self.id2node(now_id)
            if i not in pointer:
                return None
            now_id = pointer[i]
            if i == self.end_id:
                break
        try:
            return self.node2doc[now_id]
        except Exception:
            return None

    def build(self, docs):
        error_cnt = 0
        encodes = []
        for doc in docs:
            encode = self.tokenizer.encode(doc["gist"], add_special_tokens=False)
            encode = encode[:1000] + [self.end_id]
            self.set(encode, str(doc["id"]))
            encodes.append(encode)
        for encode in tqdm(encodes):
            try:
                ids = self.get(encode)
                self.remove(encode)
                self.set_short(encode, ids)
            except Exception:
                error_cnt += 1
        temp = self.show()
        cnt = sum([len(self.get(i)) for i in temp])
        return {
            "docs": len(docs),
            "save_docs": cnt,
            "nodes": len(temp),
            "avg_token": sum([len(i) for i in temp]) / max(len(temp), 1),
        }

    def last_id(self, lst, target):
        if target is None:
            return 0
        if isinstance(target, int):
            target = [target]
        target_len = len(target)
        for i in range(len(lst) - target_len, -1, -1):
            if lst[i : i + target_len] == target:
                return i + target_len
        return -1

    def __call__(self, batch_id, path):
        if isinstance(path, torch.Tensor):
            path = path.cpu().tolist()
        start_pos = self.last_id(path, self.start_id)
        path = [] if start_pos < 0 else path[start_pos:]
        now_id = self.root
        for i in path:
            pointer = self.id2node(now_id)
            if i not in pointer:
                return [self.end_id]
            now_id = pointer[i]
        results = list(self.id2node(now_id).keys())
        if len(results) == 0:
            results = [self.end_id]
        return results


class WandbTableLogger:
    def __init__(self, project="genir", entity=None, **kwargs):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, entity=entity)
        self._tables = defaultdict(lambda: defaultdict(dict))
        self.config = self.run.config
        self.config_set = False

    def update(self, model: str, metric: str, value, table_name: str = "results_table"):
        self._tables[table_name][model][metric] = value
        data = self._tables[table_name]
        cols = ["Model"] + sorted({m for scores in data.values() for m in scores})
        wb_table = self.wandb.Table(columns=cols)
        for mdl, scores in data.items():
            wb_table.add_data(mdl, *[scores.get(c, None) for c in cols[1:]])
        self.wandb.log({table_name: wb_table})


class LogMessage:
    def __init__(self, log_file, disable=False):
        self.log_file = log_file
        self.disable = disable
        if not self.disable and os.path.dirname(self.log_file):
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, *message):
        message = " ".join([str(i) for i in message])
        if not self.disable:
            current_time = "[" + time.strftime("%H:%M:%S") + "]"
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(current_time + " " + message + "\n")
            print(current_time + " " + message)


def py_trec_eval(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int] = (10, 50, 100, 200, 1000),
) -> Dict[str, float]:
    query_ids = set(qrels.keys()) & set(results.keys())
    if not query_ids:
        return {}
    max_k = max(k_values) if k_values else 0
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    _map = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    num_queries = len(query_ids)

    for qid in query_ids:
        qrel = qrels[qid]
        sorted_docs = sorted(results[qid].items(), key=lambda x: (-x[1], x[0]))
        rel_list = [qrel.get(doc_id, 0) for doc_id, _ in sorted_docs]
        total_relevant = sum(r > 0 for r in qrel.values())

        contributions = []
        relevant_count = 0
        for i in range(1, max_k + 1):
            rel = rel_list[i - 1] if i - 1 < len(rel_list) else 0
            if rel > 0:
                relevant_count += 1
                contributions.append(relevant_count / i)
            else:
                contributions.append(0.0)

        for k in k_values:
            dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(rel_list[:k], 1))
            ideal_relevances = sorted(
                (r for r in qrel.values() if r > 0), reverse=True
            )[:k]
            idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal_relevances, 1))
            ndcg[f"NDCG@{k}"] += dcg / idcg if idcg else 0.0

            retrieved_relevant = sum(1 for r in rel_list[:k] if r > 0)
            recall[f"Recall@{k}"] += (
                retrieved_relevant / total_relevant if total_relevant else 0.0
            )

            sum_precision = sum(contributions[:k])
            _map[f"MAP@{k}"] += (
                sum_precision / total_relevant if total_relevant else 0.0
            )

    for metric in [ndcg, _map, recall]:
        for key in metric:
            metric[key] = round(metric[key] / num_queries, 5)

    return {**ndcg, **_map, **recall}


def trec_eval(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int] = (10, 50, 100, 200, 1000),
) -> Dict[str, float]:
    import pytrec_eval

    ndcg, _map, recall = {}, {}, {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    all_metrics = {}
    for mt in [_normalize(ndcg), _normalize(_map), _normalize(recall)]:
        all_metrics.update(mt)
    return all_metrics


def eval_bm25(task, instruct=False):
    import bm25s

    ensure_mair_task(task)
    doc_name = get_doc_corpus_name(task)
    data = read_jsonl(f"dataset/MAIR-Queries/{task}/queries.jsonl")
    docs = read_jsonl(f"dataset/MAIR-Docs/{doc_name}/docs.jsonl")
    doc_content = [item["doc"] for item in docs]
    doc_ids = [str(item["id"]) for item in docs]
    corpus_tokens = bm25s.tokenize(doc_content, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    results = {}
    for item in data:
        query = item["query"]
        if instruct:
            query = item["instruction"] + " " + query
        query_tokens = bm25s.tokenize(query)
        if len(query_tokens.vocab) == 0:
            query_tokens = bm25s.tokenize("NONE", stopwords=[])
        hits, scores = retriever.retrieve(
            query_tokens, corpus=doc_ids, k=min(100, len(doc_ids))
        )
        qid = str(item["qid"])
        results[qid] = {}
        for i in range(len(hits[0])):
            results[qid][str(hits[0, i])] = float(scores[0, i])
    qrels = {}
    for item in data:
        qrels[str(item["qid"])] = {
            str(x["id"]): int(x["score"]) for x in item["labels"]
        }
    return py_trec_eval(qrels, results, k_values=(1, 5, 10, 100))


def infer_bm25(task, data, allowed_doc_ids=None, instruct=False, k=100):
    import bm25s

    ensure_mair_task(task)
    doc_name = get_doc_corpus_name(task)
    docs = read_jsonl(f"dataset/MAIR-Docs/{doc_name}/docs.jsonl")
    if allowed_doc_ids is not None:
        docs = [d for d in docs if str(d["id"]) in allowed_doc_ids]
    doc_content = [item["doc"] for item in docs]
    doc_ids = [str(item["id"]) for item in docs]
    corpus_tokens = bm25s.tokenize(doc_content, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    new_data = []
    start_time = time.time()
    for item in data:
        query = item["query"]
        if instruct:
            query = item["instruction"] + " " + query
        query_tokens = bm25s.tokenize(query)
        if len(query_tokens.vocab) == 0:
            query_tokens = bm25s.tokenize("NONE", stopwords=[])
        hits, scores = retriever.retrieve(
            query_tokens, corpus=doc_ids, k=min(k, len(doc_ids))
        )
        old_labels = {str(a["id"]) for a in item["labels"]}
        negative = []
        for i in range(len(hits[0])):
            cand_id = str(hits[0, i])
            if cand_id not in old_labels:
                negative.append({"id": cand_id, "score": float(scores[0, i])})
        item["negative"] = negative
        new_data.append(item)
    print(
        "Done BM25 Inference of",
        len(new_data),
        "data | Time cost",
        round(time.time() - start_time, 2),
    )
    return new_data


def ensure_doc_as_q(doc_name, task_name, docs_gist_map, num_q=16):
    """Ensure auxiliary document-span indexing queries (DocAsQ) exist for corpus indexing."""
    out_path = f"dataset/MAIR-Data/DocAsQ-{num_q}/{doc_name}/queries.jsonl"
    if os.path.exists(out_path):
        return out_path

    ensure_mair_task(task_name)
    queries = read_jsonl(f"dataset/MAIR-Queries/{task_name}/queries.jsonl")
    inst = queries[0].get("instruction", "Retrieve document relevant to the given query.")
    if inst != queries[-1].get("instruction", inst):
        inst = "Retrieve document relevant to the given query."

    raw_docs = read_jsonl(f"dataset/MAIR-Docs/{doc_name}/docs.jsonl")
    doc_as_q = []
    for item in raw_docs:
        did = str(item["id"])
        if did not in docs_gist_map:
            continue
        text = item["doc"].strip()
        # Split document into sentence/line spans for auxiliary indexing
        spans = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        if not spans:
            spans = [text[:256]]
        gist = docs_gist_map[did]
        doc_as_q.append(
            {
                "instruction": inst,
                "query": f"{gist}\n{spans[0]}",
                "labels": [{"id": did, "score": 1}],
            }
        )
        for sp in spans[1:num_q]:
            doc_as_q.append(
                {
                    "instruction": inst,
                    "query": sp,
                    "labels": [{"id": did, "score": 1}],
                }
            )
    write_jsonl(doc_as_q, out_path)
    return out_path


def normalized_sigmoid(x, k=10, m=0.5):
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    a = sigmoid(k * (0 - m))
    b = sigmoid(k * (1 - m))
    return (sigmoid(k * (x - m)) - a) / (b - a)


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


class RerankData(Dataset):
    def __init__(self, data, docs, tokenizer, group_size=8):
        self.data = data
        self.docs = docs
        self.tokenizer = tokenizer
        self.group_size = group_size

    def __getitem__(self, index):
        question = self.data[index]["query"]
        labels = self.data[index]["labels"]
        negative = self.data[index]["negative"]

        question = self.tokenizer.decode(
            self.tokenizer.encode(question)[:64], skip_special_tokens=True
        )
        inst = (
            self.data[index]["instruction"] if "instruction" in self.data[index] else ""
        )
        inst = self.tokenizer.decode(
            self.tokenizer.encode(inst)[:64], skip_special_tokens=True
        )
        question = inst + "\n" + question

        neg_id = [i for i in range(len(negative)) if str(negative[i]["id"]) in self.docs]
        np.random.shuffle(neg_id)

        group = []
        for i in [0]:
            cand_answer = self.docs[str(labels[i]["id"])]
            ids, _ = llama_encode(question, cand_answer, self.tokenizer)
            group.append(torch.tensor(ids[:200]))

        for i in neg_id[: self.group_size - 1]:
            cand_answer = self.docs[str(negative[i]["id"])]
            ids, _ = llama_encode(question, cand_answer, self.tokenizer)
            group.append(torch.tensor(ids[:200]))

        while len(group) < self.group_size:
            cand_answer = "NONE"
            ids, _ = llama_encode(question, cand_answer, self.tokenizer)
            group.append(torch.tensor(ids[:200]))
        assert len(group) == self.group_size
        return group

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids = sum(batch, [])
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class TrainGenData(Dataset):
    def __init__(self, data, docs, tokenizer):
        self.data = data
        self.docs = docs
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        question = self.data[index]["query"]
        labels = self.data[index]["labels"]
        answer = self.docs[str(labels[0]["id"])]
        question = self.tokenizer.decode(
            self.tokenizer.encode(question)[:64], skip_special_tokens=True
        )
        inst = (
            self.data[index]["instruction"] if "instruction" in self.data[index] else ""
        )
        inst = self.tokenizer.decode(
            self.tokenizer.encode(inst)[:64], skip_special_tokens=True
        )
        question = inst + "\n" + question

        if "t5" in self.tokenizer.name_or_path.lower():
            ids, labels = t5_encode(question, answer, self.tokenizer)
        else:
            ids, labels = llama_encode(question, answer, self.tokenizer)
        return torch.tensor(ids[:200]), torch.tensor(labels[:200])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "labels": labels}


def rank_net_loss(
    y_pred, y_true=None, weight_by_diff=False, weight_by_diff_powed=False
):
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))
    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]
    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(
            pairs_true[:, :, 1], 2
        )
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return torch.nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def train(
    name="ArguAna",
    gist_type="title",
    prompt="",
    qg_path="QG-16",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    save_path="models/Llama-Tmp-1B",
    batch_size=16,
    lr=5e-5,
    epochs=5,
    max_steps=0,
    max_eval_queries=0,
    decode_steps=100,
    wandb_logger=None,
    save=False,
    do_eval=True,
):
    accelerator = Accelerator(gradient_accumulation_steps=1)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log_path = "log/scale.log"
    logger = LogMessage(log_path, disable=not accelerator.is_main_process)
    logger.log("")
    logger.log("#" * 20)
    logger.log(name)
    logger.log("#" * 20)

    group_size = 8
    rerank_train_step = 8

    if "Qwen" in model_name:
        model_type = "Qwen"
    elif "t5" in model_name.lower():
        model_type = "t5"
    elif "Phi" in model_name:
        model_type = "Phi"
    else:
        model_type = "Llama"

    if accelerator.is_main_process and wandb_logger is not None:
        if not wandb_logger.config_set:
            wandb_logger.config.model_type = model_type
            wandb_logger.config.model_name = model_name
            wandb_logger.config.save_path = save_path
            wandb_logger.config.batch_size = batch_size
            wandb_logger.config.lr = lr
            wandb_logger.config.gist_type = gist_type
            wandb_logger.config.qg_path = qg_path
            wandb_logger.config.log_path = log_path
            wandb_logger.config.group_size = group_size
            wandb_logger.config.rerank_train_step = rerank_train_step
            wandb_logger.config_set = True

    doc_name = get_doc_corpus_name(name)
    t = time.time()

    docs = {
        str(x["id"]): x["gist"]
        for x in read_jsonl(f"dataset/MAIR-Data/gist_ids/{doc_name}/{gist_type}.jsonl")
    }
    if accelerator.is_main_process:
        ensure_doc_as_q(doc_name, name, docs, num_q=16)
    accelerator.wait_for_everyone()

    raw_data = read_jsonl(f"dataset/MAIR-Data/{qg_path}/{doc_name}/queries.jsonl")
    raw_data += read_jsonl(f"dataset/MAIR-Data/DocAsQ-16/{doc_name}/queries.jsonl")
    data = [x for x in raw_data if str(x["labels"][0]["id"]) in docs]

    accelerator.print(data[0])
    accelerator.print("Data", len(data), "Docs", len(docs))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    default_pads = {"Qwen": 151643, "Llama": 128004, "t5": 0, "Phi": 199999}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            default_pads[model_type]
            if default_pads[model_type] < len(tokenizer)
            else tokenizer.eos_token_id
        )

    if "t5" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=get_attn_impl(),
        )
        model.config.use_cache = False

    dataset = TrainGenData(data, docs, tokenizer)
    accelerator.print(tokenizer.decode(dataset[0][0]))
    accelerator.wait_for_everyone()
    accelerator.print("Time load data", round(time.time() - t, 2), "s")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    num_warmup_steps = max(len(data_loader) // 2, 1)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps
    )

    if accelerator.is_main_process:
        filter_data = []
        seen_doc = set()
        for item in read_jsonl(f"dataset/MAIR-Data/{qg_path}/{doc_name}/queries.jsonl"):
            did = str(item["labels"][0]["id"])
            if did in docs and did not in seen_doc:
                filter_data.append(item)
                seen_doc.add(did)
        filter_data = infer_bm25(
            name, filter_data, allowed_doc_ids=set(docs.keys()), instruct=False, k=100
        )
        write_jsonl(filter_data, "dataset/cache.jsonl")
    accelerator.wait_for_everyone()

    filter_data = read_jsonl("dataset/cache.jsonl")
    rerank_dataset = RerankData(filter_data, docs, tokenizer, group_size=group_size)
    rerank_loader = torch.utils.data.DataLoader(
        rerank_dataset,
        collate_fn=rerank_dataset.collate_fn,
        shuffle=True,
        batch_size=max(batch_size // group_size, 1),
        num_workers=0,
    )
    rerank_loader = accelerator.prepare(rerank_loader)
    rerank_iter = iter(rerank_loader)
    rerank_token = {"Llama": 128002, "Qwen": 151643, "t5": 0, "Phi": 199999}[model_type]

    global_step = 0
    for epoch in range(epochs):
        logger.log(f"Training {epoch}")
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(
            data_loader, total=len(data_loader), disable=not accelerator.is_main_process
        )
        loss_report = []
        rank_loss_report = []
        step = 0
        for batch in tk0:
            step += 1
            global_step += 1
            with accelerator.accumulate(model):
                if step % rerank_train_step == 0 and len(rerank_dataset) > 0:
                    try:
                        rerank_batch = next(rerank_iter)
                    except StopIteration:
                        rerank_iter = iter(rerank_loader)
                        rerank_batch = next(rerank_iter)
                    logits = model(**rerank_batch).logits
                    logits = logits[:, :, rerank_token]

                    attention_mask = rerank_batch["attention_mask"]
                    token_indices = torch.arange(
                        attention_mask.shape[-1],
                        device=logits.device,
                        dtype=torch.int32,
                    )
                    last_non_pad_token = (token_indices * attention_mask).argmax(-1)
                    cur_bs = logits.shape[0]
                    pooled_logits = logits[
                        torch.arange(cur_bs, device=logits.device),
                        last_non_pad_token,
                    ]
                    pooled_logits = pooled_logits.view(-1, group_size)
                    y_true = torch.zeros_like(pooled_logits)
                    y_true[:, 0] = 1
                    rerank_loss = rank_net_loss(pooled_logits, y_true)
                    accelerator.backward(rerank_loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    rank_loss_report.append(
                        accelerator.gather(rerank_loss).mean().item()
                    )

                out = model(**batch)
                loss = out.loss

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            loss_report.append(accelerator.gather(loss).mean().item())
            tk0.set_postfix(
                loss=sum(loss_report[-100:]) / len(loss_report[-100:]),
                rank_loss=sum(rank_loss_report[-100:])
                / max(len(rank_loss_report[-100:]), 1),
            )
            if max_steps > 0 and global_step >= max_steps:
                break
        accelerator.wait_for_everyone()
        logger.log()

        if do_eval:
            eval_results, rerank_eval_results = eval(
                name,
                gist_type=gist_type,
                prompt=prompt,
                model=accelerator.unwrap_model(model),
                tokenizer=tokenizer,
                model_type=model_type,
                max_eval_queries=max_eval_queries,
                decode_steps=decode_steps,
            )

            logger.log(eval_results)
            logger.log(rerank_eval_results)
            if accelerator.is_main_process and wandb_logger is not None and eval_results:
                wandb_logger.update(
                    name, f"Epoch-{epoch}", eval_results["NDCG@1"] * 100, "main/ndcg1"
                )
                wandb_logger.update(
                    name, f"Epoch-{epoch}", eval_results["NDCG@10"] * 100, "main/ndcg10"
                )
                wandb_logger.update(
                    name,
                    f"Epoch-{epoch}",
                    eval_results["Recall@100"] * 100,
                    "main/recall100",
                )
                wandb_logger.update(
                    name,
                    f"Epoch-{epoch}",
                    rerank_eval_results["NDCG@1"] * 100,
                    "rank/ndcg1",
                )
                wandb_logger.update(
                    name,
                    f"Epoch-{epoch}",
                    rerank_eval_results["NDCG@10"] * 100,
                    "rank/ndcg10",
                )

        accelerator.wait_for_everyone()
        if save:
            os.makedirs(save_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{save_path}/Epoch-{epoch}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(f"{save_path}/Epoch-{epoch}")
            accelerator.print("Save ckpt at", f"{save_path}/Epoch-{epoch}")
            accelerator.wait_for_everyone()
        if max_steps > 0 and global_step >= max_steps:
            break
    return accelerator.unwrap_model(model), tokenizer


def split_data(length, num_parts, make_equal=True):
    if make_equal:
        part_size = length // num_parts
        return [
            list(range(i * part_size, (i + 1) * part_size)) for i in range(num_parts)
        ]
    else:
        indices = list(range(length))
        avg = length / num_parts
        return [indices[int(i * avg) : int((i + 1) * avg)] for i in range(num_parts)]


@torch.no_grad()
def generate(steps, model, input_ids, tree, model_type, tokenizer=None, **kwargs):
    outs = []
    temp_tree = copy.deepcopy(tree)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer is not None and tokenizer.pad_token_id is not None
        else {"Qwen": 151643, "Llama": 128004, "t5": 0, "Phi": 199999}[model_type]
    )
    eos_id = {"Qwen": 151645, "Llama": 128009, "t5": 1, "Phi": 200020}[model_type]
    for t in range(steps):
        use_sample = t != 0
        gen_kwargs = dict(kwargs)
        gen_kwargs["do_sample"] = use_sample
        if use_sample:
            gen_kwargs["temperature"] = max(
                float(normalized_sigmoid(t / steps, k=10, m=0.1) * 1.5), 1e-4
            )
        else:
            gen_kwargs.pop("temperature", None)
        temp_out = model.generate(
            input_ids=torch.tensor([input_ids]).to(model.device),
            max_new_tokens=64,
            **gen_kwargs,
            prefix_allowed_tokens_fn=temp_tree,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        temp_out = temp_out.cpu().tolist()
        temp_out = [temp[len(input_ids) :] for temp in temp_out]
        for temp in temp_out:
            temp_tree.remove(temp)
            outs.append(temp)
    outs = [tree.get(out) for out in outs]
    outs = [out for out in outs if out is not None]
    out = sum(outs, [])
    return out[:100]


@torch.no_grad()
def eval(
    name="ArguAna",
    gist_type="title",
    prompt="",
    model=None,
    tokenizer=None,
    model_type="Llama",
    max_eval_queries=0,
    decode_steps=100,
):
    accelerator = Accelerator()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ensure_mair_task(name)
    doc_name = get_doc_corpus_name(name)
    doc_ids = read_jsonl(f"dataset/MAIR-Data/gist_ids/{doc_name}/{gist_type}.jsonl")
    docs = {str(x["id"]): x["gist"] for x in doc_ids}

    data = read_jsonl(f"dataset/MAIR-Queries/{name}/queries.jsonl")
    if max_eval_queries > 0:
        data = data[:max_eval_queries]

    model.eval()
    model = model.cuda()

    start_id = {
        "Qwen": [77091, 198],
        "Llama": [128007, 271],
        "t5": [0],
        "Phi": [200019],
    }[model_type]
    end_id = {"Qwen": 151645, "Llama": 128009, "t5": 1, "Phi": 200020}[model_type]
    tree = newTree(start_id=start_id, end_id=end_id, tokenizer=tokenizer)
    accelerator.print(tree.build(doc_ids))

    indexs = split_data(len(data), accelerator.num_processes, make_equal=False)[
        accelerator.process_index
    ]

    qrels = {}
    results = {}
    rerank_results = {}
    rerank_token = {"Llama": 128002, "Qwen": 151643, "t5": 0, "Phi": 199999}[model_type]

    for idx in tqdm(indexs, disable=not accelerator.is_main_process):
        item = data[idx]
        qid = str(item["qid"])
        query = item["query"]
        query = tokenizer.decode(
            tokenizer.encode(query, add_special_tokens=False)[:128],
            skip_special_tokens=True,
        )
        inst = item["instruction"] + "\n" + query
        if "t5" in model_type.lower():
            input_ids = tokenizer.encode(inst)
        else:
            input_ids = _to_id_list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": inst}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )

        out = generate(
            min(decode_steps, len(docs)),
            model,
            input_ids,
            tree,
            model_type,
            tokenizer=tokenizer,
        )
        out = [str(d) for d in out if str(d) in docs]

        results[qid] = {}
        for i in range(len(out)):
            results[qid][out[i]] = 1 / (i + 1)

        rerank_results[qid] = {}
        if len(out) > 0:
            gist = [docs[docid] for docid in out]
            group = []
            for cand_answer in gist:
                ids, _ = llama_encode(inst, cand_answer, tokenizer)
                group.append(torch.tensor(ids[:200]))
            scores = []
            chunk_size = 16
            for start in range(0, len(group), chunk_size):
                sub_group = group[start : start + chunk_size]
                batch_ids = pad_sequence(
                    sub_group, batch_first=True, padding_value=tokenizer.pad_token_id
                )
                attention_mask = batch_ids.ne(tokenizer.pad_token_id)
                token_indices = torch.arange(
                    attention_mask.shape[-1], dtype=torch.long
                )
                last_non_pad_token = (token_indices * attention_mask.long()).argmax(-1)
                sub_logits = model(
                    input_ids=batch_ids.cuda(), attention_mask=attention_mask.cuda()
                ).logits
                sub_scores = sub_logits[
                    torch.arange(len(sub_group), device=sub_logits.device),
                    last_non_pad_token.to(sub_logits.device),
                    rerank_token,
                ]
                scores.extend(sub_scores.float().cpu().tolist())
            for i in range(len(out)):
                rerank_results[qid][out[i]] = scores[i]

        qrels[qid] = {str(x["id"]): int(x["score"]) for x in item["labels"]}

    os.makedirs("out/tmp", exist_ok=True)
    with open(f"out/tmp/qrels-{accelerator.process_index}.json", "w", encoding="utf-8") as f:
        json.dump(qrels, f)
    with open(f"out/tmp/results-{accelerator.process_index}.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    with open(
        f"out/tmp/rerank_results-{accelerator.process_index}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(rerank_results, f)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged_qrels, merged_results, merged_rerank = {}, {}, {}
        for i in range(accelerator.num_processes):
            merged_qrels.update(json.load(open(f"out/tmp/qrels-{i}.json", encoding="utf-8")))
            merged_results.update(json.load(open(f"out/tmp/results-{i}.json", encoding="utf-8")))
            merged_rerank.update(
                json.load(open(f"out/tmp/rerank_results-{i}.json", encoding="utf-8"))
            )
        eval_results = py_trec_eval(merged_qrels, merged_results, k_values=(1, 5, 10, 100))
        rerank_eval_results = py_trec_eval(
            merged_qrels, merged_rerank, k_values=(1, 5, 10, 100)
        )
    else:
        eval_results = None
        rerank_eval_results = None

    accelerator.print(eval_results)
    accelerator.print(rerank_eval_results)
    return eval_results, rerank_eval_results


def main():
    default_model = (
        "models/Llama-3.2-1B-Instruct"
        if os.path.exists("models/Llama-3.2-1B-Instruct")
        else "meta-llama/Llama-3.2-1B-Instruct"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="ArguAna")
    parser.add_argument("--model_name", type=str, default=default_model)
    parser.add_argument("--save_path", type=str, default="models/Llama-Tmp-1B")
    parser.add_argument("--qg_path", type=str, default="QG-16")
    parser.add_argument("--gist_type", type=str, default="title")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_eval_queries", type=int, default=0)
    parser.add_argument("--decode_steps", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()
    if accelerator.is_main_process and args.use_wandb:
        wandb_logger = WandbTableLogger()
    else:
        wandb_logger = None

    my_task = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for name in my_task:
        if accelerator.is_main_process:
            bm25_results = eval_bm25(name)
            print("BM25 baseline:", bm25_results)
            if wandb_logger is not None and bm25_results:
                wandb_logger.update(
                    name, "BM25", bm25_results["NDCG@1"] * 100, "main/ndcg1"
                )
                wandb_logger.update(
                    name, "BM25", bm25_results["NDCG@10"] * 100, "main/ndcg10"
                )
                wandb_logger.update(
                    name, "BM25", bm25_results["Recall@100"] * 100, "main/recall100"
                )
                wandb_logger.update(
                    name, "BM25", bm25_results["NDCG@1"] * 100, "rank/ndcg1"
                )
                wandb_logger.update(
                    name, "BM25", bm25_results["NDCG@10"] * 100, "rank/ndcg10"
                )
        accelerator.wait_for_everyone()
        train(
            name,
            gist_type=args.gist_type,
            prompt="Query: ",
            qg_path=args.qg_path,
            model_name=args.model_name,
            save_path=args.save_path,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            max_steps=args.max_steps,
            max_eval_queries=args.max_eval_queries,
            decode_steps=args.decode_steps,
            wandb_logger=wandb_logger,
            save=args.save,
        )


if __name__ == "__main__":
    main()
