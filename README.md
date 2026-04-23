# ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieval

Official implementation of the paper *"ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieval"* ([arXiv:2510.10419](https://arxiv.org/abs/2510.10419)).

ZeroGR is a zero-shot generative retrieval framework that leverages natural-language task instructions to extend GR across a wide range of IR tasks. It is composed of three key components:

1. **LM-based DocID Generator** — unifies heterogeneous documents (text, tables, code) into semantically meaningful DocIDs.
2. **Instruction-tuned Query Generator** — generates diverse pseudo-queries conditioned on natural-language task descriptions to enhance corpus indexing.
3. **Reverse-Annealed Decoding** — a decoding strategy that balances precision and recall during DocID generation.

Empirical results on the **BEIR** and **MAIR** benchmarks show that ZeroGR outperforms strong dense retrieval and generative baselines in zero-shot settings, establishing a new state-of-the-art for instruction-driven GR.

## Authors

Weiwei Sun¹\*, Keyi Kong²\*, Xinyu Ma³, Shuaiqiang Wang³, Dawei Yin³, Maarten de Rijke⁴, Zhaochun Ren⁵†, Yiming Yang¹

¹ Carnegie Mellon University  ² Shandong University  ³ Baidu Inc.  ⁴ University of Amsterdam  ⁵ Leiden University

\*Equal contribution  †Corresponding author

## Framework Overview

```
                Document Indexing                              Document Retrieval
  ┌────────────────────────────────────────┐       ┌────────────────────────────────────┐
  │  Documents ──► Query Generator  ──►    │       │   Search Query                     │
  │              Pseudo Queries            │       │        │                           │
  │                    │                   │       │        ▼                           │
  │              Instruction Tuning        │       │     ZeroGR ──► Constrained         │
  │                    │                   │       │        │        Decoding           │
  │  Documents ──► DocID Generator ──►     │       │        ▼                           │
  │              DocID                     │       │   DocID List                       │
  └────────────────────────────────────────┘       └────────────────────────────────────┘
```

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── file_io.py          # I/O utilities (JSON / JSONL / pickle, multiprocessing, dir mgmt)
├── mair_config.py      # MAIR task/domain configuration and corpus sharing
├── sftqg.py            # Supervised fine-tuning of the Query Generator
├── qg_vllm.py          # vLLM-based batched inference for the Query Generator
├── sftid.py            # Supervised fine-tuning of the DocID (Title) Generator
├── title_vllm.py       # vLLM-based batched inference for the DocID Generator
└── genir.py            # Core generative retriever: training, indexing, reverse-annealed decoding
```

## Hardware Requirements

- Training and inference are validated on **8×A800 (80GB)** GPUs.
- Lower-memory setups may work with reduced batch size and gradient accumulation.

## Installation

```bash
git clone https://github.com/sunnweiwei/ZeroGR.git
cd ZeroGR
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `accelerate`, `vllm`, `liger-kernel`, `datasets`, `wandb`, `tqdm`, `numpy`.

## Datasets

ZeroGR is trained and evaluated on the [MAIR](https://huggingface.co/datasets/MAIR-Bench/MAIR-Docs) benchmark, which spans 6 domains (Medical, Financial, Academic, Coding, Legal, Web-based) and 69 IR tasks. Download the following to `dataset/`:

| Resource      | Source                                                               |
|---------------|----------------------------------------------------------------------|
| MAIR-Docs     | <https://huggingface.co/datasets/MAIR-Bench/MAIR-Docs>               |
| MAIR-Queries  | <https://huggingface.co/datasets/MAIR-Bench/MAIR-Queries>            |
| MAIR-Data     | Generated pseudo-queries and DocIDs (produced by this pipeline)      |

Expected layout:

```
dataset/
├── MAIR-Docs/<task>/docs.jsonl
├── MAIR-Queries/<task>/queries.jsonl
└── MAIR-Data/<model_sufix>-<num_q>/<task>/queries.jsonl
```

ZeroGR-Train statistics (Table 1 of the paper):

| Domain     | #Tasks | #Samples   |
|------------|-------:|-----------:|
| Medical    | 5      | 421,430    |
| Financial  | 8      | 31,315     |
| Academic   | 18     | 744,160    |
| Coding     | 13     | 1,969,586  |
| Legal      | 7      | 23,086,948 |
| Web-based  | 18     | 15,319,445 |

Evaluation: [BEIR](https://github.com/beir-cellar/beir) (12 tasks) and MAIR (seen / unseen splits).

## Usage

The pipeline follows the *Document Indexing → Document Retrieval* workflow.

### 1. Train and run the Query Generator

```bash
# Fine-tune a Llama-3.2-1B-Instruct model for pseudo-query generation (<1 day on 8×A800)
python sftqg.py

# Generate pseudo-queries with vLLM (<1 day on 8×A800)
python qg_vllm.py
```

Inference can also be launched per-task / per-GPU via CLI:

```bash
python qg_vllm.py \
  -docs_path dataset/MAIR-Docs/<task>/docs.jsonl \
  -data_name <task> \
  -pid 0 -total_num 8 \
  -model_sufix QG \
  -model_name models/Llama-3.2-1B-Instruct-qg \
  -num_q 16
```

### 2. Train and run the DocID Generator

```bash
# Fine-tune a Llama-3.2-1B-Instruct model for unified DocID generation (<1 day on 8×A800)
python sftid.py

# Generate DocIDs with vLLM (<1 day on 8×A800)
python title_vllm.py
```

### 3. Train and evaluate the Generative Retriever

```bash
# End-to-end training + evaluation; ~2 weeks on 8×A800 for the full ZeroGR-3B run
python genir.py
```

`genir.py` contains the core components: the constrained prefix-tree decoder, the reverse-annealed sampler (Eq. 5-6 of the paper), indexing, and evaluation (Acc@1, nDCG@10, Recall@100).

## Reverse-Annealed Decoding

ZeroGR proposes **reverse-annealed sampling** for DocID decoding. Each DocID is generated token-by-token under a constrained prefix tree, with the sampling temperature gradually *increased* over iterations to trade off precision and recall:

```
t_i = g(i) = T_max * ( sigma(k*(i/K - m)) - sigma(-k*m) )
                    / ( sigma(k*(1   - m)) - sigma(-k*m) )

sigma(z) = 1 / (1 + exp(-z))
```

where `K` is the total number of DocIDs to generate, `k > 0` controls the slope, and `m ∈ (0, 1)` sets the midpoint. Starting low yields high-precision early selections; increasing `t_i` over iterations boosts exploration.

## Main Results

Combined domain-wise results on MAIR (Acc@1) and BEIR (nDCG@10):

| Model                 | MAIR Avg | BEIR Avg |
|-----------------------|:--------:|:--------:|
| BM25                  |   36.1   |   42.4   |
| Contriever            |   33.6   |   47.6   |
| GTR-T5-large          |   35.4   |   48.0   |
| E5-Large              |   38.2   |   49.2   |
| BGE-Large             |   39.4   |   51.8   |
| OpenAI-Embed-v3-Small |   40.6   |   54.2   |
| E5-mistral-7B         |   46.8   |   55.7   |
| GritLM-7B             | **47.0** |   45.0   |
| **ZeroGR-3B**         |   41.1   | **48.1** |

See Tab. 2–4 and Fig. 2–6 of the paper for full per-task numbers, docid-design ablations, scaling analyses, and decoding comparisons.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sun2025zerogr,
  title   = {ZeroGR: A Generalizable and Scalable Framework for Zero-Shot Generative Retrieval},
  author  = {Sun, Weiwei and Kong, Keyi and Ma, Xinyu and Wang, Shuaiqiang and Yin, Dawei and de Rijke, Maarten and Ren, Zhaochun and Yang, Yiming},
  journal = {arXiv preprint arXiv:2510.10419},
  year    = {2025}
}
```

## Acknowledgements

This work was funded by the Dutch Research Council (NWO), under project numbers 024.004.022, NWA.1389.20.183, and KICH3.LTP.20.006, and the European Union under grant agreements No. 101070212 (FINDHR) and No. 101201510 (UNITE). Views and opinions expressed are those of the authors only.

## License

Released under the Apache License 2.0 — see [LICENSE](LICENSE).

## Contact

- Weiwei Sun — `sunnweiwei@gmail.com`
- Keyi Kong  — `luxinyayaya01@gmail.com`
