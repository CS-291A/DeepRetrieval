<div align="center">

# DeepRetrieval - Hacking Real Search Engines & Retrievers with LLM via RL  
<img src="images/logo.png" alt="DeepRetrieval Logo" width="400">


<p align="center">
  <a href="https://arxiv.org/pdf/2503.00223">
    <img src="https://img.shields.io/badge/arXiv-2503.00223-b31b1b.svg" alt="arXiv">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <!-- Adds horizontal space -->
  <a href="https://huggingface.co/DeepRetrieval">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" height="28">
  </a>
</br>
  ⭐️ Star ⭐️ our repo to stay tuned with new features!!
</p>

</div>

## News
- [2025-05-19] If you're particularly interested in RAG tasks, we recommend checking out our latest work, [s3](https://github.com/pat-jj/s3), a training-efficient framework specifically designed for RAG.
- [2025-05-06] You can now easily deploy our model and call the api for query rewriting. See [here](https://github.com/pat-jj/DeepRetrieval?tab=readme-ov-file#%EF%B8%8F-easy-to-use-api-for-query-rewriting) for more details.
- [2025-02-28] 🎉 We are excited to release the code for DeepRetrieval!


## What is DeepRetrieval?

DeepRetrieval is a novel reinforcement learning approach that trains Large Language Models (LLMs) for query generation to enhance information retrieval performance. Unlike traditional methods that rely on supervised learning with labeled query-augmentation pairs, DeepRetrieval lets models learn through direct trial and error, using retrieval metrics as rewards to generate queries that maximize retrieval performance.

The system works by having an LLM generate reasoning steps in a `<think>` section followed by the final augmented query in an `<answer>` section. This structured approach enables explicit chain-of-thought reasoning before committing to a query formulation.
![alt text](/images/framework.png "reward curve during training (on pubmed)")

## Key Features and Results

- **No Supervision Required**: Eliminates the need for expensive human-annotated or distilled reference queries
- **Powerful Performance**: Significantly outperforms previous state-of-the-art methods
  - 65.07% recall (vs. previous SOTA 24.68%) for publication search
  - 63.18% recall (vs. previous SOTA 32.11%) for clinical trials search
- **Versatile Applications**: Excels across diverse retrieval tasks: (1) Literature search using real-world search engines (2) Evidence-seeking retrieval (3) Classic information retrieval (4) SQL database search
- **Parameter Efficient**: Achieves superior results with only 3B parameters, outperforming larger models like **GPT-4o** and **Claude-3.5-Sonnet**


![alt text](/images/performance_overview.png "performance overview")


[Example Wandb Training Log on PubMed Search Engine](https://wandb.ai/patjj/literature_search?nw=nwuserpj20)


⭐️ Star our repository to stay up-to-date with exciting new features and improvements! 🌟

## Table of Contents

- [📦 Installation](#-installation)
- [⚡️ Easy-to-use API for Query Rewriting](#️-easy-to-use-api-for-query-rewriting)
- [🫧 Get Started](#-get-started)
- [🏃 Run Training](#-run-training)
- [☁️ Running on Runpod](#️-running-on-runpod)
- [🧐 Run Evaluation](#-run-evaluation)
- [📚 Cite DeepRetrieval](#-cite-deepretrieval)

## 📦 Installation

The Python project lives in `code/` and is managed with [uv](https://docs.astral.sh/uv/) (Python 3.10; `code/uv.lock` pins every dependency, including `torch==2.4.0`, `vllm==0.6.3`, `numpy<2` and `transformers<4.48`).

```bash
cd code
uv sync                                  # base: verl, torch, transformers, ray, wandb, ...
uv sync --extra retrieval                # + pyserini 0.22.1, faiss, func_timeout, install-jdk (BM25 / dense / SQL)
uv sync --extra gpu --extra retrieval    # Linux + CUDA only: + vllm 0.6.3, flash-attn 2.6.3 (needed for training/rollout)
uv sync --extra baselines                # + openai, anthropic, boto3, langchain-*, trl (GPT/Claude baselines, SQL cold-start SFT)
uv run pytest                            # run the test suite (GPU/data tests are skipped when unavailable)
```

Extras combine (`uv sync --extra gpu --extra retrieval --extra baselines`); each `uv sync` installs exactly the extras you name. Run commands with `uv run <cmd>` or after `source .venv/bin/activate`.

The GPT/Claude baselines (`src/utils/`) read credentials on first use from `$OPENAI_API_KEY`, `$AZURE_OPENAI_API_KEY`, or `$AWS_ACCESS_KEY_ID` + `$AWS_SECRET_ACCESS_KEY`, falling back to the key files in the working directory (`openai_api.key`, `openai_api_azure.key`, `aws_access.key` / `claude_api_aws.key`).

**(If you only want to run the Search Engine Retrieval, you can skip the following step.)**

Pyserini (sparse/dense retrieval) needs a Java runtime. If you don't have one, install it with:
```bash
uv run python -c "import jdk; jdk.install('11')"
```

## ⚡️ Easy-to-use API for Query Rewriting

```bash
cd code
sh ../vllm_host.sh # you can specify the local port and the model to use
python src/query_rewrite.py --query "Who built DeepRetrieval in 2025?"  # add --model <name> if you serve a different model
# Original query: Who built DeepRetrieval in 2025?
# Rewritten query: (The DeepRetrieval system, which was built in 2025)
```



## 🫧 Get Started

### **1. Dataset Download/Preprocess**

We provide two options for data preparation:

<details>
<summary style="font-weight: bold;">🚀 Option 1: Download pre-processed datasets from Huggingface (Recommended)</summary>

All preprocessed datasets are available on our Huggingface repository. You can download them using the provided script:

```bash
cd code
# List available datasets
python download_datasets.py --list_only --repo_id DeepRetrieval/datasets

# Download all datasets
python download_datasets.py --repo_id DeepRetrieval/datasets --output_dir ./data

# Or download specific categories/datasets
python download_datasets.py --categories search_engine --datasets pubmed_32 --output_dir ./data
```
</details>


<details>
<summary style="font-weight: bold;">⛏️ Option 2: Process the data yourself</summary>

For example, for PubMed:
```bash
cd code
python download_datasets.py --categories raw_data --datasets pubmed --output_dir ./data
python data_preprocess/pubmed_32.py
```
(This will generate the required data structures in the appropriate format, but requires raw data access and more processing time.)
</details>


### **2. Get Your Search Engine API Key (required if use search engine)**

For example, for PubMed, you may get it following the instruction [here](https://support.nlm.nih.gov/kbArticle/?pn=KA-05317). (PubMed API is 100% ✨FREE✨ to use!)

Then, put it in under `code/verl/utils/reward_score/apis/` as `pubmed_api.key`.


### **3. Reward function Related (optional)**

Reward Design (e.g., in `code/verl/utils/reward_score/pubmed.py`):


| Recall      | ≥ 0.7 | ≥ 0.5 | ≥ 0.4 | ≥ 0.3 | ≥ 0.1 | ≥ 0.05 | < 0.05 |
|-------------|-------|-------|-------|-------|-------|--------|--------|
| **Reward**  | +5.0  | +4.0  | +3.0  | +1.0  | +0.5  | +0.1   | -3.5   |



### **4. Customize Monitor Info (optional)**

modify `compute_reward_metrics()` in `code/verl/trainer/ppo/ray_trainer.py`


## 🏃 Run Training
```bash
cd code
source .venv/bin/activate   # environment created by `uv sync` (see Installation)
```

For example, for PubMed:
```bash
sh scripts/train/pubmed_32.sh 
```
(if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script)

Every launcher under `scripts/train/` and `scripts/eval/` first sources `scripts/common.sh`, which loads `code/.env`, reads datasets from `$DEEPRETRIEVAL_DATA_DIR` (default `code/data`), creates `exp_log/` and changes to `code/`. The launchers no longer set `CUDA_VISIBLE_DEVICES`: all visible GPUs are used unless you export it first (e.g. `CUDA_VISIBLE_DEVICES=0,1 sh scripts/train/pubmed_32.sh`).

## ☁️ Running on Runpod

`code/scripts/setup_pod.sh` prepares a rented pod and keeps everything slow to rebuild on a network volume (repo, venv, uv cache, uv-managed Python, Hugging Face cache, datasets), so a new pod on the same volume is ready in minutes. It is idempotent: rerun it on every new pod.

**Pod settings**
- GPU: 2× A100 80GB, or another Ampere/Hopper GPU (A40, L40S, H100). Not Blackwell: torch 2.4 has no kernels for it.
- Template: the official "Runpod Pytorch 2.4.0" image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (verified in the GPU smoke test).
- Network volume mounted at `/workspace`; container disk ≥ 80 GB.
- Optional pod environment variables / Runpod secrets: `PUBMED_API_KEY` (NCBI 10 requests/s instead of 3), `WANDB_API_KEY`, `HF_TOKEN` (needed for gated models such as `meta-llama/Llama-3.2-3B-Instruct`, used by `pubmed_32.sh`).

**On the pod** (first time; on later pods only the `bash` line is needed):
```bash
cd /workspace
git clone https://github.com/objones25/DeepRetrieval.git
bash DeepRetrieval/code/scripts/setup_pod.sh
cd DeepRetrieval/code
sh scripts/train/pubmed_32.sh
```

`setup_pod.sh` checks `/workspace` is a mount and `nvidia-smi` sees the GPUs, installs uv onto the volume if missing, fast-forwards the checkout (a checkout with uncommitted changes is left alone), writes `code/.env` (mode 600), runs `uv sync --extra gpu --extra retrieval`, downloads `pubmed_32` into `/workspace/data` (skipped when already present), and checks the Triton toolchain and the train/test parquet files. Secrets come from the environment first, then the existing `code/.env`, then a hidden prompt when run in a terminal. Values are never printed. Useful options:
- `--non-interactive`: never prompt; missing optional secrets are skipped. Add `--require-pubmed-key` to fail instead when `PUBMED_API_KEY` is missing.
- `--datasets pubmed_32,ctgov_32 --categories search_engine`: download more datasets; `--force` downloads again.
- `--branch NAME`, `--repo URL`: which code to check out. `--gpus N` sets how many GPUs to expect (default 2). `--dry-run` prints the steps without running them.

Training can outlive your SSH session, so start it inside `tmux` (or `nohup ... &`). Logs go to `code/exp_log/` and checkpoints to `code/checkpoints/`, both on the volume. Without `WANDB_API_KEY`, export `WANDB_MODE=offline` before launching.

### Think/Query Length During Training (PubMed)

![alt text](/images/length_study_horizontal.png "think length and query length during training")


## 🧐 Run Evaluation

```
sh scripts/eval/pubmed_32.sh
```
(You can run this script without training, as it will download our trained model from Huggingface by default)

**Results on Search Engines**

| Model | Method | Publication Recall | Clinical Trials Recall |
|-------|--------|-------------------|----------------------|
| Original Query | - | 10.36 | 18.01 |
| GPT-3.5 | - | 11.67 | 9.42 |
| | w/o reasoning | 18.68 | 13.94 |
| GPT-4o | - | 17.59 | 16.25 |
| | w/o reasoning | 19.72 | 14.26 |
| Claude-3-Haiku | - | 11.26 | 10.10 |
| | w/o reasoning | 20.92 | 24.68 |
| Claude-3.5-Sonnet | - | 20.94 | 18.33 |
| | w/o reasoning | 19.08 | 18.41 |
| Mistral-7B-Inst | - | 7.18 | 8.08 |
| LEADS-7B (SFT) | - | **24.68** | **32.11** |
| Qwen2.5-3B-Inst | - | 6.59 | 6.09 |
| | w/o reasoning | 9.46 | 7.97 |
| **DeepRetrieval-3B** | - | **✨ 65.07 ✨** | **✨ 63.18 ✨** |
| | w/o reasoning | 51.90 | 53.31 |

*Table: Comparison of different models and methods on publication search and clinical trials search tasks.

**Note:** LEADS-7B is a state-of-the-art literature mining LLM trained on 20K reviews and 400K publications [https://arxiv.org/pdf/2501.16255]

## ⚠️ Commonly Encountered Issues

### ⚙️ Cluster Setup: CUDA and Conda Environment

<details>
<summary><b>💻 slurm-related</b></summary>

Before running any training or evaluation scripts on the cluster, make sure to **load the correct CUDA module** and **activate your Conda environment**.

---

### 🧩 GPUs Allocation

If your CUDA inside a SLURM job doesn’t match the CUDA seen by your Python script, you may get a warning like:

UserWarning: CUDA initialization: CUDA driver initialization failed, you might not have a CUDA GPU.
(Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
return torch._C._cuda_getDeviceCount() > 0


To fix this, **do not manually set your CUDA device** inside the script.  
Instead, load the CUDA module directly inside your SLURM job:

Add the following lines **before running any Python commands**:

```bash
# --- Load CUDA module ---
module load cuda/12.6

# --- Activate Conda environment ---
source ~/miniconda/etc/profile.d/conda.sh  # adjust path if your conda installation differs
conda activate zero
```
The launchers in `code/scripts/` no longer set `CUDA_VISIBLE_DEVICES`. If your own job file sets a device (e.g. `export CUDA_VISIBLE_DEVICES=4,7`), remove that line so SLURM manages CUDA correctly.

### ⚙️ Dependency Conflicts
The pip/conda dependency conflicts previously documented here (missing `pyparsing`/`decorator`/`bitsandbytes`, and `numpy>=2` breaking vllm 0.6.3 / outlines 0.0.46) do not arise with the uv setup: `code/uv.lock` resolves one consistent set with `numpy<2`, `transformers<4.48` and `vllm==0.6.3`. If you still install with pip, pin those three versions.
</details>

## 🤝 Acknowledgement

This implementation is mainly based on [verl](https://github.com/volcengine/verl) and [PySerini](https://github.com/castorini/pySerini). The base model during the experiment is [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). We sincerely appreciate their contributions to the open-source community.

## 📚 Cite DeepRetrieval

```
@article{jiang2025deepretrieval,
  title={Deepretrieval: Hacking real search engines and retrievers with large language models via reinforcement learning},
  author={Jiang, Pengcheng and Lin, Jiacheng and Cao, Lang and Tian, Runchu and Kang, SeongKu and Wang, Zifeng and Sun, Jimeng and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.00223},
  year={2025}
}
```

Thanks for your interests! 😊
