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

# from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch
from torch.optim import AdamW, Adafactor
import time
from torch.utils.data import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# import faiss
from tqdm import tqdm
import copy
import json
from itertools import product
from torch.nn import CrossEntropyLoss
from typing import Dict, Tuple
import numpy as np
from collections import defaultdict
import pickle
import shutil
import os
from file_io import read_jsonl, read_json, write_pkl, read_pkl, write_json, write_jsonl
import wandb


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
            # print(path)
            if len(path) == 0:
                return True
            pointer = self.id2node(id)
            if path[0] not in pointer:
                return True  # bug?
                # raise ValueError("Path not found")
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
        except:
            return None

    def build(self, docs):
        error_cnt = 0
        encodes = []
        for doc in docs:
            encode = self.tokenizer.encode(doc["gist"], add_special_tokens=False)
            encode = encode[:1000] + [self.end_id]
            self.set(encode, doc["id"])
            encodes.append(encode)
        for encode in tqdm(encodes):
            try:
                ids = self.get(encode)
                self.remove(encode)
                self.set_short(encode, ids)
            except:
                error_cnt += 1
        temp = self.show()
        cnt = sum([len(self.get(i)) for i in temp])
        return {
            "docs": len(docs),
            "save_docs": cnt,
            "nodes": len(temp),
            "avg_token": sum([len(i) for i in temp]) / len(temp),
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
        self.run = wandb.init(project=project, entity=entity)
        self._tables = defaultdict(lambda: defaultdict(dict))
        self.config = self.run.config
        self.config_set = False

    def update(self, model: str, metric: str, value, table_name: str = "results_table"):
        self._tables[table_name][model][metric] = value
        data = self._tables[table_name]
        cols = ["Model"] + sorted({m for scores in data.values() for m in scores})
        wb_table = wandb.Table(columns=cols)
        for mdl, scores in data.items():
            wb_table.add_data(mdl, *[scores.get(c, None) for c in cols[1:]])
        wandb.log({table_name: wb_table})


class LogMessage:
    def __init__(self, log_file, disable=False):
        self.log_file = log_file
        self.disable = disable

    def log(self, *message):
        message = " ".join([str(i) for i in message])
        if not self.disable:
            current_time = "[" + time.strftime("%H:%M:%S") + "]"
            with open(self.log_file, "a") as file:
                file.write(current_time + " " + message + "\n")
            print(current_time + " " + message)


from typing import Dict, Tuple
import math


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

        # Precompute AP contributions up to max_k
        contributions = []
        relevant_count = 0
        for i in range(1, max_k + 1):
            if i - 1 < len(rel_list):
                rel = rel_list[i - 1]
            else:
                rel = 0
            if rel > 0:
                relevant_count += 1
                contributions.append(relevant_count / i)
            else:
                contributions.append(0.0)

        for k in k_values:
            # Compute NDCG@k
            current_k = min(k, len(rel_list))
            dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(rel_list[:k], 1))
            ideal_relevances = sorted(
                (r for r in qrel.values() if r > 0), reverse=True
            )[:k]
            idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal_relevances, 1))
            ndcg_val = dcg / idcg if idcg else 0.0
            ndcg[f"NDCG@{k}"] += ndcg_val

            # Compute Recall@k
            retrieved_relevant = sum(1 for r in rel_list[:k] if r > 0)
            recall_val = retrieved_relevant / total_relevant if total_relevant else 0.0
            recall[f"Recall@{k}"] += recall_val

            # Compute MAP@k
            sum_precision = sum(contributions[:k])
            map_val = sum_precision / total_relevant if total_relevant else 0.0
            _map[f"MAP@{k}"] += map_val

    # Normalize metrics by the number of queries
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

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics


def eval_bm25(task, instruct=False):
    import bm25s

    data = read_jsonl(f"dataset/MAIR-Queries/{task}/queries.jsonl")
    docs = read_jsonl(f"dataset/MAIR-Docs/{task}/docs.jsonl")
    doc_content = [item["doc"] for item in docs]
    doc_ids = [item["id"] for item in docs]
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
        results[item["qid"]] = {}
        for i in range(len(hits[0])):
            results[item["qid"]][hits[0, i]] = float(scores[0, i])
    qrels = {}
    for item in data:
        qrels[item["qid"]] = {str(x["id"]): int(x["score"]) for x in item["labels"]}
    eval_results = py_trec_eval(qrels, results, k_values=(1, 5, 10, 100))
    return eval_results


def infer_bm25(task, data, instruct=False, k=100):
    import bm25s

    docs = read_jsonl(f"dataset/MAIR-Docs/{task}/docs.jsonl")
    doc_content = [item["doc"] for item in docs]
    doc_ids = [item["id"] for item in docs]
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
        old_labels = [a["id"] for a in item["labels"]]
        negative = []
        for i in range(len(hits[0])):
            if hits[0, i] not in old_labels:
                negative.append({"id": hits[0, i], "score": float(scores[0, i])})
        item["negative"] = negative
        new_data.append(item)
    print(
        "Done BM25 Inference of",
        len(new_data),
        "data | Time cost",
        time.time() - start_time,
    )
    return new_data


def normalized_sigmoid(x, k=10, m=0.5):
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    a = sigmoid(k * (0 - m))
    b = sigmoid(k * (1 - m))
    return (sigmoid(k * (x - m)) - a) / (b - a)


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

        neg_id = [i for i in range(len(negative))]
        np.random.shuffle(neg_id)

        group = []

        for i in [0]:
            cand_answer = self.docs[labels[i]["id"]]
            ids, _ = llama_encode(question, cand_answer, self.tokenizer)
            group.append(torch.tensor(ids[:200]))

        for i in neg_id[: self.group_size - 1]:
            cand_answer = self.docs[negative[i]["id"]]
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
        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return features


class TrainGenData(Dataset):
    def __init__(self, data, docs, tokenizer):
        self.data = data
        self.docs = docs
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        question = self.data[index]["query"]
        labels = self.data[index]["labels"]
        answer = self.docs[labels[0]["id"]]
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

        if "t5" in self.tokenizer.name_or_path:
            ids, labels = t5_encode(question, answer, self.tokenizer)
        else:
            ids, labels = llama_encode(question, answer, self.tokenizer)
        # ids = ids + [0] * 512
        # labels = labels + [-100] * 512
        return torch.tensor(ids[:200]), torch.tensor(labels[:200])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        features = {"input_ids": input_ids, "labels": labels}
        return features


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
    gist_type="query",
    prompt="",
    qg_path="D2Q",
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

    model_name = "models/Llama-3.2-1B-Instruct"
    save_path = "models/Llama-Tmp-1B"

    batch_size = 16
    lr = 5e-5
    group_size = 8
    rerank_train_step = 8

    if "Qwen" in model_name:
        model_type = "Qwen"
    elif "t5" in model_name:
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

    from mair_config import SHARE_CORPUS

    doc_name = {v: k for k, values in SHARE_CORPUS.items() for v in values}
    doc_name = doc_name.get(name, name)

    t = time.time()
    data = read_jsonl(f"dataset/MAIR-Data/{qg_path}/{doc_name}/queries.jsonl")
    data += read_jsonl(f"dataset/MAIR-Data/DocAsQ-16/{doc_name}/queries.jsonl")

    docs = {
        str(x["id"]): x["gist"]
        for x in read_jsonl(f"dataset/MAIR-Data/gist_ids/{doc_name}/{gist_type}.jsonl")
    }
    print(len(docs))

    accelerator.print(data[0])
    accelerator.print("Data", len(data), "Docs", len(docs))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = {"Qwen": 151643, "Llama": 128004, "t5": 0, "Phi": 199999}[
        model_type
    ]
    tokenizer.add_tokens(["<|g|>", "<|/g|>"] + [f"<|g{i}|>" for i in range(20000)])

    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False

    dataset = TrainGenData(data, docs, tokenizer)
    accelerator.print(tokenizer.decode(dataset[0][0]))
    accelerator.wait_for_everyone()
    accelerator.print("Time load data", time.time() - t, "s")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    num_warmup_steps = len(data_loader) // 2
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps
    )

    if accelerator.is_main_process:
        filter_data = []
        seen_doc = set()
        for item in read_jsonl(f"dataset/MAIR-Data/{qg_path}/{doc_name}/queries.jsonl"):
            if item["labels"][0]["id"] not in seen_doc:
                filter_data.append(item)
                seen_doc.add(item["labels"][0]["id"])
        filter_data = infer_bm25(name, filter_data, instruct=False, k=100)
        write_jsonl(filter_data, "dataset/cache.jsonl")
    accelerator.wait_for_everyone()
    filter_data = read_jsonl("dataset/cache.jsonl")
    rerank_dataset = RerankData(filter_data, docs, tokenizer, group_size=group_size)
    rerank_loader = torch.utils.data.DataLoader(
        rerank_dataset,
        collate_fn=rerank_dataset.collate_fn,
        shuffle=True,
        batch_size=batch_size // group_size,
        num_workers=0,
    )
    rerank_loader = accelerator.prepare(rerank_loader)
    rerank_iter = iter(rerank_loader)
    rerank_token = {"Llama": 128002}[model_type]

    os.system("sh kill_gpu.sh")
    for epoch in range(5):
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
            with accelerator.accumulate(model):
                if step % rerank_train_step == 0:
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
                    pooled_logits = logits[
                        torch.arange(batch_size, device=logits.device),
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
            )

            logger.log(eval_results)
            logger.log(rerank_eval_results)
            if accelerator.is_main_process and wandb_logger is not None:
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
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{save_path}/Epoch-{epoch}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(f"{save_path}/Epoch-{epoch}")
            accelerator.print("Save ckpt at", f"{save_path}/Epoch-{epoch}")
            accelerator.wait_for_everyone()
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


def cut(path, base=32000):
    if base in path:
        return path[path.index(base) + 1 :]
    return []


@torch.no_grad()
def generate(steps, model, input_ids, tree, model_type, **kwargs):
    outs = []
    temp_tree = copy.deepcopy(tree)
    for t in range(steps):
        temp_out = model.generate(
            input_ids=torch.tensor([input_ids]).to(model.device),
            max_new_tokens=64,
            do_sample=t != 0,
            temperature=normalized_sigmoid(t / steps, k=10, m=0.1) * 1.5,
            **kwargs,
            prefix_allowed_tokens_fn=temp_tree,
            pad_token_id={"Qwen": 151643, "Llama": 128004, "t5": 0, "Phi": 199999}[
                model_type
            ],
            eos_token_id={"Qwen": 151645, "Llama": 128009, "t5": 1, "Phi": 200020}[
                model_type
            ],
        )
        temp_out = temp_out.cpu().tolist()
        temp_out = [temp[len(input_ids) :] for temp in temp_out]
        for temp in temp_out:
            temp_tree.remove(temp)
            outs.append(temp)
    outs = [tree.get(out) for out in outs]
    outs = [out for out in outs if out is not None]
    out = sum(outs, [])
    out = out[:100]
    return out


@torch.no_grad()
def eval(
    name="ArguAna",
    gist_type="query",
    prompt="",
    model=None,
    tokenizer=None,
    model_type="Llama",
):
    accelerator = Accelerator()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from mair_config import SHARE_CORPUS

    doc_name = {v: k for k, values in SHARE_CORPUS.items() for v in values}
    doc_name = doc_name.get(name, name)

    doc_ids = read_jsonl(f"dataset/MAIR-Data/gist_ids/{doc_name}/{gist_type}.jsonl")

    docs = {str(x["id"]): x["gist"] for x in doc_ids}

    data = read_jsonl(f"dataset/MAIR-Queries/{name}/queries.jsonl")

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

    if accelerator is not None:
        indexs = split_data(len(data), accelerator.num_processes, make_equal=False)[
            accelerator.process_index
        ]
        process_index = accelerator.process_index
    else:
        indexs = [i for i in range(len(data))]
        process_index = 0

    if accelerator.is_main_process:
        os.system("sh kill_gpu.sh")

    qrels = {}
    results = {}
    rerank_results = {}
    for idx in tqdm(indexs):
        item = data[idx]
        query = item["query"]
        query = tokenizer.decode(
            tokenizer.encode(query, add_special_tokens=False)[:128],
            add_special_tokens=False,
        )
        inst = item["instruction"]
        inst = inst + "\n" + query
        if "t5" in model_type:
            input_ids = tokenizer.encode(inst)
        else:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": inst}],
                tokenize=True,
                add_generation_prompt=True,
            )

        out = generate(
            100,
            model,
            input_ids,
            tree,
            model_type,
            do_sample=True,
        )

        results[item["qid"]] = {}
        for i in range(len(out)):
            results[item["qid"]][out[i]] = 1 / (i + 1)

        with torch.no_grad():
            rerank_token = {"Llama": 128002}[model_type]
            gist = [docs[docid] for docid in out]
            group = []
            for cand_answer in gist:
                ids, _ = llama_encode(inst, cand_answer, tokenizer)
                group.append(torch.tensor(ids[:200]))
            input_ids = pad_sequence(
                group, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits = logits[:, :, rerank_token]
            logits = logits.cpu()
            attention_mask = attention_mask.cpu()
            token_indices = torch.arange(attention_mask.shape[-1], dtype=torch.int32)
            last_non_pad_token = (token_indices * attention_mask).argmax(-1)
            pooled_logits = logits[torch.arange(len(group)), last_non_pad_token]
            scores = pooled_logits.tolist()

        rerank_results[item["qid"]] = {}
        for i in range(len(out)):
            rerank_results[item["qid"]][out[i]] = scores[i]

        qrels[item["qid"]] = {x["id"]: x["score"] for x in item["labels"]}

    if accelerator is not None:
        json.dump(qrels, open(f"out/tmp/qrels-{accelerator.process_index}.json", "w"))
        json.dump(
            results, open(f"out/tmp/results-{accelerator.process_index}.json", "w")
        )
        json.dump(
            rerank_results,
            open(f"out/tmp/rerank_results-{accelerator.process_index}.json", "w"),
        )
        accelerator.wait_for_everyone()

    if accelerator is None:
        eval_results = py_trec_eval(qrels, results, k_values=(1, 10, 100))
        rerank_eval_results = py_trec_eval(qrels, rerank_results, k_values=(1, 10, 100))
    elif accelerator.is_main_process:
        qrels = {}
        results = {}
        rerank_results = {}
        for i in range(accelerator.num_processes):
            qrels.update(json.load(open(f"out/tmp/qrels-{i}.json")))
            results.update(json.load(open(f"out/tmp/results-{i}.json")))
            rerank_results.update(json.load(open(f"out/tmp/rerank_results-{i}.json")))
        eval_results = py_trec_eval(qrels, results, k_values=(1, 5, 10, 100))
        rerank_eval_results = py_trec_eval(
            qrels, rerank_results, k_values=(1, 5, 10, 100)
        )

    else:
        eval_results = None
        rerank_eval_results = None
    accelerator.print(eval_results)
    accelerator.print(rerank_eval_results)
    return eval_results, rerank_eval_results


def main():

    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb_logger = WandbTableLogger()
    else:
        wandb_logger = None

    qg_path = "QG-16"
    my_task = [""]

    for name in my_task:

        if accelerator.is_main_process:
            bm25_results = eval_bm25(name)
            print(bm25_results)
            wandb_logger.update(
                name, f"BM25", bm25_results["NDCG@1"] * 100, "main/ndcg1"
            )
            wandb_logger.update(
                name, f"BM25", bm25_results["NDCG@10"] * 100, "main/ndcg10"
            )
            wandb_logger.update(
                name, f"BM25", bm25_results["Recall@100"] * 100, "main/recall100"
            )
            wandb_logger.update(
                name, f"BM25", bm25_results["NDCG@1"] * 100, "rank/ndcg1"
            )
            wandb_logger.update(
                name, f"BM25", bm25_results["NDCG@10"] * 100, "rank/ndcg10"
            )
        accelerator.wait_for_everyone()
        model, tokenizer = train(
            name,
            gist_type="title",
            prompt="Query: ",
            qg_path=qg_path,
            wandb_logger=wandb_logger,
            save=False,
        )


if __name__ == "__main__":
    main()
