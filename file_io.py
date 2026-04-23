import os
import json
from tqdm import tqdm
import numpy as np
import re
import time
import datetime
import pickle
import os


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
    with open(file) as f:
        for line in f:
            out.append(line)
    return out


def read_json(file):
    if not os.path.exists(file):
        return {}
    return json.load(open(file))


def read_jsonl(file):
    out = []
    for line in read_file(file):
        out.append(json.loads(line))
    return out


def create_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_json(obj, file):
    create_dir(file)
    with open(file, "w") as f:
        json.dump(obj, f)


def write_jsonl(obj, file):
    create_dir(file)
    with open(file, "w") as f:
        for line in obj:
            f.write(json.dumps(line) + "\n")


def add_jsonl(obj, file):
    create_dir(file)
    with open(file, "a") as f:
        for line in obj:
            f.write(json.dumps(line) + "\n")


def write_pkl(obj, file):
    create_dir(file)
    pickle.dump(obj, open(file, "wb"))


def read_pkl(file):
    return pickle.load(open(file, "rb"))
