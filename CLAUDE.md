# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Code for *DeepRetrieval* (arXiv 2503.00223): PPO-trains a small LLM (Qwen2.5-3B / Llama-3.2-3B) to rewrite queries, using real retrieval recall as the reward. The model emits `<think>…</think><answer>…</answer>`; the answer (often JSON `{"query": ...}`) is executed against a search engine / BM25 / dense index / SQL DB and scored. It is a fork of **verl 0.1** — the installable package is still named `verl`.

## Environment (uv, not conda)

All Python work happens in `code/` (`code/pyproject.toml`, `code/uv.lock`, `code/.python-version` = 3.10). The README's conda instructions are outdated.

```bash
cd code
uv sync                          # base deps (torch 2.4.0, transformers<4.48, numpy<2, ray, ...)
uv sync --extra retrieval        # + pyserini 0.22.1, faiss, func_timeout, install-jdk (BM25/dense/SQL tasks)
uv sync --extra gpu --extra retrieval   # Linux+CUDA only: vllm 0.6.3, flash-attn 2.6.3, torch cu121
uv sync --extra baselines        # + openai/anthropic/boto3/langchain-*/trl for src/ GPT/Claude baselines
```

- `gpu` extra deps carry `sys_platform == 'linux'` markers; on macOS they are skipped, so training/rollout cannot run there — only linting, type-checking and CPU tests.
- Pins that must stay together: `torch==2.4.0` ↔ `vllm==0.6.3`; `numpy<2` (vllm/outlines); `transformers<4.48`; `tensordict<0.6`. Change them only via `pyproject.toml` + `uv lock`.
- uv is pinned to its own CPython builds (`[tool.uv] python-preference = "only-managed"`): Triton's runtime JIT (flash-attn kernels on the training path) needs a C compiler and `Python.h`, which system Pythons on Debian-based GPU images lack. `verl.utils.toolchain.require_triton_toolchain()` checks both (Triton's exact lookup) in `main_ppo.main` and in `require_cuda` (every CUDA worker).
- Runpod pods: `bash code/scripts/setup_pod.sh` (run on the pod; idempotent) installs uv, clones/fast-forwards the repo, `uv sync --extra gpu --extra retrieval`, writes `code/.env` (mode 600, gitignored; template `code/.env.example`), downloads pubmed_32 and verifies GPUs/Triton/parquet. Repo, venv, uv cache, uv-managed Python (`UV_PYTHON_INSTALL_DIR`), HF cache and data all live on the `/workspace` volume. Secrets (`PUBMED_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`): environment, else existing `.env`, else hidden prompt; never printed. Tests: `npx --yes bats@1.13.0 tests/shell` (stubs git/uv/nvidia-smi; `TEST_BASH=/bin/bash` for bash 3.2).
- Pyserini needs a JDK: `utils/java_init.py` (imported first thing by `verl/trainer/main_ppo.py`) sets `JAVA_HOME` from `~/.jdk/jdk-*` (installed via `install-jdk`).

## Commands (run from `code/`)

```bash
uv run ruff check .
uv run pyright                   # uses repo-root pyrightconfig.json (works from code/ or the root)
uv run pytest tests/sanity tests/utility tests/verl          # CPU-runnable subset
uv run pytest tests/utility/test_tensor_dict_utilities.py::test_union_tensor_dict   # single test
uv run pytest -p no:randomly                 # fixed order (pytest-randomly shuffles by default; reproduce with --randomly-seed=N)
```

Type checking: `pyrightconfig.json` at the repo root is shared by Pylance and the CLI. It excludes the vendored `verl/third_party/vllm` copies and the megatron strategy files (their deps aren't installable from the lock; DeepRetrieval trains with FSDP only). Pylance ignores `venvPath`, so select `code/.venv/bin/python` as the editor interpreter. `pandas-stubs` is in the dev group so the CLI sees the same pandas types as Pylance's bundled stubs; the project type-checks clean (0 errors). Linux-only imports (vllm, flash_attn) carry line-scoped `# pyright: ignore[reportMissingImports]` with a reason; use that form, never bare `# type: ignore`.

`tests/conftest.py` makes plain `uv run pytest` work on CPU: test files importing flash_attn/vllm aren't collected when those are missing, `tests/{ray,gpu_utility,model,rollout,e2e}` get the `gpu` marker and skip without CUDA, `tests/verl/utils/dataset/*` get `data` and skip without parquet files in `~/verl-data/`, and RNGs are seeded per test. Markers are strict (`gpu`, `data`, `network`). DeepRetrieval-specific unit tests live in `tests/deepretrieval/{rewards,core,evals}/`; they stub pyserini/flash_attn/vllm/API SDKs via `sys.modules` so they need no JVM, GPU or network.

Training / eval / data. Every launcher under `scripts/{train,eval}/` first sources `scripts/common.sh` (POSIX sh, launchers run under `sh`/dash): it loads `code/.env` without overriding variables already set, defaults `DEEPRETRIEVAL_DATA_DIR=<code>/data` and `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`, uses `code/.venv` if no venv is active, creates `exp_log/` and cds to `code/`. Launchers read data as `"$DEEPRETRIEVAL_DATA_DIR"/...` and never set `CUDA_VISIBLE_DEVICES` (export it to pick GPUs). `tests/shell/launchers.bats` enforces this for new launchers:

```bash
python download_datasets.py --categories search_engine --datasets pubmed_32 --output_dir ./data
sh scripts/train/pubmed_32.sh        # -> python -m verl.trainer.main_ppo <hydra overrides>
sh scripts/eval/pubmed_32.sh         # -> src/eval/SearchEngine/eval_pubmed.py
```

Search-engine rewards read the PubMed key from `code/verl/utils/reward_score/apis/pubmed_api.key` (resolved relative to the package; `*.key` is gitignored; missing key → warning + keyless, rate-limited client). `src/utils/*` baseline clients read `OPENAI_API_KEY` / `AZURE_OPENAI_API_KEY` / AWS env vars first, then the legacy key files.

## Architecture

**Training entry:** `verl/trainer/main_ppo.py` — Hydra (`verl/trainer/config/ppo_trainer.yaml`, all settings overridden from the `scripts/train/*.sh` command lines) → `ray.init` → `main_task` builds worker groups and `RayPPOTrainer` (`verl/trainer/ppo/ray_trainer.py`), which runs the PPO loop: vLLM rollout → reward → ref/critic log-probs → GAE (`ppo/core_algos.py`) → actor/critic updates.

**Single-controller / workers:** `verl/single_controller/` is verl's Ray abstraction — a driver calls methods on a `RayWorkerGroup`; `@register(dispatch_mode=...)` decorators (`single_controller/base/decorator.py`) define how a `DataProto` is split across and gathered from workers. Workers live in `verl/workers/` (`fsdp_workers.py` is the path actually used; megatron variants are upstream leftovers). `DataProto` (`verl/protocol.py`) is the batch container passed everywhere: a `TensorDict` (`.batch`) plus numpy object arrays (`.non_tensor_batch`, holding `data_source`, `reward_model.ground_truth`, `pub_date`, `extra_info.db_path` …).

**Logging.** Library code only creates `logging.getLogger(__name__)` loggers (no `print` in reward/API code; `import verl` configures nothing). Entry points (`main_ppo.main` *and* its Ray `main_task`, `main_generation`, `main_eval`, `fsdp_sft_trainer`) call `verl.utils.logging_utils.configure_logging()`: root at WARNING, the `verl`/`__main__` loggers at `VERL_LOGGING_LEVEL` (default INFO). Reward policy: per-rollout detail at DEBUG (set `VERL_LOGGING_LEVEL=DEBUG` to see every query), the `do_print`-sampled dumps at INFO, infrastructure problems at WARNING.

**GPU policy: fail fast.** `main_ppo.main` checks Ray sees enough GPUs right after `ray.init`; FSDP workers and `initialize_global_process_group` call `verl.utils.distributed.require_cuda`; flash-attn imports re-raise a readable `ImportError`. There is no CPU fallback for training.

**`DataProto.batch` is `Optional[TensorDict]`.** It is `None` for metrics-only results (e.g. `update_actor`/`update_critic`). Where a batch must exist, narrow once per function with `assert data.batch is not None, '<why>'` and keep using `data.batch[...]`; re-assert after rebinding `data`.

**vLLM integration:** `verl/third_party/vllm/` holds vendored, patched vLLM copies per version (`vllm_v_0_3_1` … `vllm_v_0_6_3`), selected at import time by installed vllm version in `third_party/vllm/__init__.py`. Only `vllm_v_0_6_3` matters with the current lock. `verl/workers/sharding_manager/fsdp_vllm.py` syncs FSDP weights into the rollout engine.

**Rewards (the DeepRetrieval-specific part):** `RewardManager` in `main_ppo.py` decodes each response and routes by **substring match on `data_source`** through the ordered `ROUTES` table in `verl/trainer/reward_registry.py` (row order is precedence: e.g. `msmarco_beir` before `msmarco`; `dense`, `no_reason`, `wor` substrings pick a row's variant module). Each row also names the extra kwargs builder and the validation/training metric function, so `main_ppo` and `ray_trainer` contain no data_source chains. Rewards go to a `compute_score` in:
- `verl/utils/reward_score/<task>.py` — sparse/search-engine/SQL rewards (pubmed, ctgov, scifact, nq_serini, bird, spider, …). Each file repeats the same `extract_solution` / `validate_response_structure` / recall→reward step-table pattern; a fix in one usually applies to all.
- `verl/utils/reward_score_dense/` — dense-retrieval variants.
- `verl/utils/apis/` — PubMed / ClinicalTrials.gov HTTP clients (rate-limited) used as the "search engine".
The scalar reward is placed at the last valid response token. Failure policy in rewards: misconfiguration (search client/index is None, empty label) fails fast; malformed model JSON and any error while executing/scoring the model's query return the task's penalty constant, the latter logged via `logger.exception` (a model-written query must never crash training); SQL execution timeouts return `None` (never a match). Add a new task with a `compute_score` module plus one row in `reward_registry.ROUTES` (characterisation tests in `tests/deepretrieval/{rewards/test_reward_routing.py,core/test_reward_metric_routing.py}` pin every existing route). The metric functions in the registry invert the reward step tables (e.g. reward `6` ⇒ hit@5), so changing a reward table silently breaks the metrics. W&B: `RayPPOTrainer.fit` finishes the run with exit code 1 on any exception (Ray swallows it before `sys.excepthook`), 0 otherwise.

**Outside the package:**
- `data_preprocess/` — builds the `train/test.parquet` files (prompt template + `data_source` + ground truth) per task; `no_reason/` variants drop the `<think>` section.
- `src/` — standalone scripts, not imported by training: `eval/` (BM25 / Dense / SearchEngine / SQL evaluation and GPT/Claude baselines), `Lucene/` & `Dense/` index building, `sql/` executors, `cold_start/` SFT, `query_rewrite.py` (client for the vLLM server started by root `vllm_host.sh`).
- `examples/`, `docs/`, `docker/` are mostly unmodified upstream verl.
