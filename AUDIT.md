# Audit fixes (branch `audit-fixes`)

Static analysis (ruff, pyright), the `pytest-expert` / `negative-space-programming` / `design-patterns-expert` audit scripts and manual review found the issues below. Every code fix was made test-first: a failing test, then the fix, then a passing test. Exceptions are recorded in the item itself: the SQL half of A6 and one B17 test were written right after the code change and then confirmed against HEAD; A10 is checked with ruff because megatron isn't installed; and the G2 test errored rather than failed on HEAD. An independent audit ran the new tests against an untouched HEAD worktree: 326 failed and 2 errored there, and all of them pass on this branch. Every test that also passed on HEAD was reviewed and is a deliberate guard.

Item IDs: `P*` packaging/infra, `A*` crash bugs, `B*` silent failures/hangs, `C*` deprecations, `D*` performance, `E*` CUDA fail-fast, `F*` packaging/docs/tests, `G*` approved behaviour changes.

**Result:** `uv run pytest -q` (from `code/`): before `30 tests collected, 4 errors`; after `485 passed, 16 skipped` on macOS arm64 (CPU). The 16 skipped need CUDA or `~/verl-data`.

## Behaviour changes (approved)

These items deliberately change rewards, training, or reported numbers. Runs made after these fixes are not directly comparable with the paper's numbers for the affected tasks.

| ID | Change | Affects |
|---|---|---|
| G1 | KL loss is now **added** to the policy loss (was subtracted); `actor/kl_loss` logged per micro-batch | runs with `actor.use_kl_loss=True` (no shipped script sets it) |
| G2 | Claude-3-Haiku baseline model name `hiku` → `haiku` (every call used to fail and score 0) | `src/eval/baselines/evidence_gpt_claude*.py` numbers |
| G3 | One shared reward→metric mapping for training and validation; hit@50 threshold corrected; validation keys now `val/recall@k/{data_source}`; msmarco training NDCG no longer subtracts a non-existent 0.2 bonus; SQL `wrong_answer_ratio` fixed | wandb metric names and values (dashboards keyed on `val/recall@5` need updating) |
| G4 | NDCG ideal ranking sorted by relevance; duplicate hits counted once; list targets without scores are treated as binary relevance (no reward passes one). scifact uses a single docid target; it changes only through duplicate-hit dedup (e.g. hits d1, d1 used to score NDCG 1.63, now 1) | fever/hotpotqa/msmarco_beir/nfcorpus/scifact/fiqa rewards (sparse & dense) and the scifact evals |
| G5 | SQL execution timeout returns `None`; a timed-out prediction never matches; a timed-out gold query is logged and scores 0 | bird/spider (+wikisql) rewards and SQL evals |
| G7 | Every `src/eval` script counts a failed/unparseable query as 0 in the denominator (BM25/Dense msmarco and `answer_filtered_test.py` used to drop them) | those eval numbers |

Also user-visible but not in the G list:
- **B17/B14/E33 (misconfiguration fails fast):** a missing search index or client, an empty ground-truth label, or too few GPUs now stops the run with a clear error. Before, the bad config was silently scored as a −2/−3/−4 penalty, or the run hung waiting for GPUs. Reward values for every model output are unchanged, and search/API failures caused by a model's query are logged with a traceback and keep HEAD's penalty.
- **B14:** with no PubMed key file, rewards use keyless (3 req/s) requests. Before, every PubMed reward was −3.
- **A11:** an msmarco `data_source` without `sparse`/`dense` raises up front. It previously crashed on the first well-formatted answer.
- **A4:** CT.gov raises at the API layer when every retry fails. Before, it silently returned partial results. The reward layer logs this and scores HEAD's −4.
- **A6:** JSON answers that are not objects (`5`, `null`, `["sql"]`) used to crash the reward. They now get the format penalty, except `["query"]` in the query rewards, which keeps HEAD's 0.
- **A13:** `src/query_rewrite.py` now defaults to the NQ model that `vllm_host.sh` serves. **D31:** API-key environment variables take precedence over the `./*.key` files.

## Not done (needs a decision)

- **URL-encoding PubMed/CT.gov queries** (`verl/utils/apis/pubmed.py`, `ctgov.py`). A model-written `&`/`#` can change request parameters. Encoding changes which documents are retrieved.
- **Opening SQL databases read-only (`?mode=ro`)**. A model-written `DROP TABLE` persists in the bird/spider fixture DBs today; this also blocks safely caching gold-SQL results (D30).
- **Narrowing the SQL rewards' broad `except Exception`**. Model-written SQL raises many exception types.
- **`java_init` JDK choice** sorts versions as text (`jdk-9` > `jdk-21`).
- **`use_dynamic_bsz` loss scaling** in `dp_actor.py`.
- **FSDP `set_state_dict_type` deprecation** in `fsdp_vllm.py`.
- **Batched dense msmarco_subset search**.
- **Eval rate limiter only warns.**
- **Baseline generators re-save the previous query's text** when all 10 API attempts fail.
- **`ray_trainer.py:689`** still lists the removed `trial_no_reason` data source in a metric condition. It's harmless because no module exists for it.


# Part 1 — Packaging and test infrastructure



### P1 — Replace requirements.txt/setup.py/conda with uv
- Files: `code/pyproject.toml` (new), `code/uv.lock` (new), `code/.python-version` (new, `3.10`); deleted `code/requirements.txt`, `code/setup.py`.
- Problem: README's conda + `torch==2.4.0 --index-url .../cu121` flow cannot install on macOS; `flash-attn` and `vllm` in requirements.txt break `pip install -e .` on non-CUDA hosts; unpinned resolution pulled `transformers 4.57.6` despite `transformers<4.48`; numpy<2 had to be fixed by hand (README "Dependency Conflicts").
- Fix: PEP 621 `pyproject.toml` (setuptools backend, version read from `verl/version/version`, packages `verl*`, `utils*`, `tests*` as before). Base deps = old requirements minus GPU-only packages, plus `huggingface_hub`, `requests` (directly imported). Extras: `gpu` (`vllm==0.6.3`, `flash-attn==2.6.3`, Linux-only markers; flash-attn built against the runtime torch via `extra-build-dependencies`, metadata declared so locking works on macOS), `retrieval` (`pyserini==0.22.1`, `faiss-gpu==1.7.2` on Linux / `faiss-cpu` elsewhere, `func_timeout`, `install-jdk`). `dev` group: pytest, pyright, ruff. torch comes from the PyTorch cu121 index on Linux, PyPI elsewhere. Pins encoded: `torch==2.4.0` (vllm 0.6.3), `numpy<2`, `transformers<4.48`, `tensordict<0.6`.
- Python bumped 3.9 → 3.10 (3.9 is EOL; vllm 0.6.3 supports ≤3.12).
- Verification: `uv lock` resolves 216 packages; Linux resolution = torch 2.4.0+cu121, vllm 0.6.3, transformers 4.47.1, numpy 1.26.4; `uv sync --extra retrieval` + imports of `verl`, `verl.utils.reward_score.pubmed`, `pyserini` succeed on macOS arm64.
- Behaviour impact: torch, vllm and transformers resolve to the README's pins. Dependencies that used to be unpinned are now locked to whatever was current at lock time (e.g. ray 2.59.0, python-dateutil 2.9.0.post0 because botocore needs <3). C21/C22 rely on that ray version's public `ray.util.state` / `ray.util.get_node_ip_address`. None of this has been run on a GPU host yet.

### P2 — Delete stray `code/verl/utils/reward_score/setup.py`
- Problem: copy of verl's packaging script inside the reward package; reads `reward_score/verl/version/version`, `reward_score/README.md`, `./requirements.txt` — none exist, and nothing downloads them (`download_datasets.py` writes only to `--output_dir`, default `./data`). Importing `verl.utils.reward_score.setup` would execute `setup()`. Not referenced anywhere; added in the initial commit.
- Behaviour impact: none.

### P3 — Shared pytest configuration
- Files: `code/tests/conftest.py` (new), `code/pyproject.toml` `[tool.pytest.ini_options]`.
- Problem: on a CPU-only machine `pytest` stopped with 4 collection errors (flash_attn, vllm, `typing.Unpack`) before running anything; `tests/ray/*` would block forever on GPU placement groups; dataset tests failed with FileNotFoundError; no markers registered; unseeded RNG in tests.
- Fix: `collect_ignore_glob` for test files whose hard dependency (flash_attn / vllm) is not installed; auto-apply `gpu` marker + skip without CUDA for `tests/{ray,gpu_utility,model,rollout,e2e}`; auto-apply `data` marker + skip when `~/verl-data` contains no parquet files; autouse fixture seeding `random`, `numpy`, `torch`; `--strict-markers` with `gpu`, `data`, `network` registered.
- Test: `uv run pytest -q` — before: `30 tests collected, 4 errors`; after: `14 passed, 16 skipped`.
- Behaviour impact: none (GPU hosts with data run the same tests as before).

### P4 — CLAUDE.md
- New repo-root `CLAUDE.md` describing the uv workflow, test commands and architecture.

# Part 2 — Rewards, search APIs, trainer entry point

All tests are in `code/tests/deepretrieval/rewards/` (the test ids below leave off that prefix). Fixtures in `conftest.py` stub pyserini and the `src/{Lucene,Dense}/*/search.py` modules in `sys.modules` and import every reward module from an empty cwd, so no test needs a JVM, an index or the network.

### A1: `extract_solution` UnboundLocalError when the response has no header
- Files: every `verl/utils/reward_score/*.py` and `verl/utils/reward_score_dense/*.py` that has an `extract_solution`, in the "Failed to locate model response header" branch (around line 24 to 41 in each file). That is 27 files.
- Problem: the branch ran `return None, processed_str` before `processed_str` was assigned.
- Fix: it now returns `None, ""`. The format check then fails and the existing format penalty applies.
- Test: `test_extract_solution.py::test_extract_solution_without_header_returns_no_answer_and_empty_text` (parametrized over 27 modules).
- Behaviour impact: none. Training used to crash on this input; now it gets the normal format penalty.

### A3: fiqa routing and `trial_no_reason`
- Files: `verl/trainer/main_ppo.py:54` (new fiqa branch), `:134`, and the RewardManager conditions.
- Problem: `_select_rm_score_fn` had no 'fiqa' branch. `RewardManager` routed 'trial_no_reason', but no module exists for it.
- Fix: added the fiqa branch. There is no dense fiqa (`reward_score_dense/` has no fiqa.py; verified). Removed 'trial_no_reason' from both RewardManager conditions. The `NotImplementedError` now names the data_source.
- Test: `test_main_ppo.py::test_select_rm_score_fn_routes_fiqa_to_fiqa_reward` and `::test_select_rm_score_fn_names_unknown_data_source`.
- Behaviour impact: none. **Not edited:** `verl/trainer/ppo/ray_trainer.py:689` still lists 'trial_no_reason' in its metrics chain.

### A4: PubMed / CT.gov retry bugs
- Files: `verl/utils/apis/pubmed.py:50,104,189`; `verl/utils/apis/ctgov.py:45` and the final raise.
- Problem: in pubmed, `err_msg` was never cleared, so a retry that succeeded still raised. `pmid_list` and `retstart` carried over between attempts. In ctgov, `nctid_list` was not reset per attempt, so a retry duplicated ids. When every attempt failed after a partial page, ctgov silently returned the partial list.
- Fix: both clients reset their state at the start of each attempt and clear `err_msg` on success. ctgov now raises whenever every attempt failed, and the message includes the last error.
- Test: `test_apis.py::test_pubmed_search_returns_results_when_a_retry_succeeds`, `::test_pubmed_get_papers_returns_papers_when_a_retry_succeeds`, `::test_ctgov_search_does_not_duplicate_ids_when_a_retry_succeeds`, `::test_ctgov_search_raises_instead_of_returning_partial_results_when_every_attempt_fails`, `::test_pubmed_search_raises_when_every_attempt_fails`.
- Behaviour impact: retries now work. When every attempt fails, ctgov raises instead of returning partial results. The reward layer (B17) logs that error with a traceback and scores HEAD's −4 (`test_answer_score_errors.py::test_ctgov_reward_scores_api_failing_every_retry_as_minus_four`).

### A5: imports of modules that are not shipped
- Files: `verl/utils/reward_score/wikisql.py:12-30` and the top of `calculate_answer_score`; `verl/trainer/main_eval.py:26`.
- Fix: in wikisql, a `ModuleNotFoundError` is caught at import. `get_searcher()` raises a clear `ImportError` (chained to the original), and it is called before the SQL `except Exception` so it cannot be scored as bad SQL. In main_eval, `math` is imported lazily inside `select_reward_fn` with a clear `ImportError`. `gsm8k` was unused and was dropped.
- Test: `test_missing_modules.py` (4 tests).
- Behaviour impact: none.

### A6: `check_json_format` accepted non-object JSON
- Files: `check_json_format` in bird, bird_no_reason, spider, spider_no_reason and wikisql. Also extended to the 12 query-JSON rewards (fever, fiqa, hotpotqa, msmarco, msmarco_beir, nfcorpus, scifact and the 5 dense variants), which had the same bug: `5` raised TypeError and crashed, and `["query"]` passed the check.
- Fix: the 5 SQL files got `isinstance(data, dict)`. The 12 query-JSON rewards don't: their `check_json_format` now also catches `TypeError`, so `5`/`null` get the format penalty instead of crashing, and `["query"]` keeps HEAD's value of 0.
- Test: `test_sql.py::test_sql_reward_rejects_non_object_json_as_format_error`, `test_answer_score_errors.py::test_ndcg_reward_scores_non_object_json_like_head_without_crashing`.
- Behaviour impact: inputs that used to crash training now get the format penalty. All other values are unchanged.

### A11: `mode` unbound in msmarco
- Files: `verl/utils/reward_score/msmarco.py:248`.
- Fix: raises `ValueError` naming the data_source.
- Test: `test_msmarco.py::test_compute_score_rejects_data_source_without_retrieval_mode`.
- Behaviour impact: a data_source with neither 'sparse' nor 'dense' (for example the `msmarco_sub.py` default `msmarco`) now raises on every call. Before, it gave -2 for badly formatted output and crashed on well-formatted output.

### B14: PubMed key path depended on the cwd
- Files: `verl/trainer/main_ppo.py:38` and `RewardManager.__init__`.
- Fix: `PUBMED_API_KEY_PATH` is built from `project_root`, at the same location the README gives. The key is read with `with open`. If the file is missing, a `logger.warning` is logged and `PubmedAPI(api_key=None)` is created; PubmedAPI only appends the key when it is set (verified).
- Test: the 4 `test_main_ppo.py::test_pubmed_key_*` / `test_reward_manager_*` tests.
- Behaviour impact: without a key file, PubMed rewards now use keyless requests (NCBI allows 3 req/s) instead of silently becoming -3. CLAUDE.md has been updated to match.

### B15: requests had no timeout
- Files: `pubmed.py:12` and all 4 `requests.get` calls; `ctgov.py:9` and both calls.
- Fix: `timeout=REQUEST_TIMEOUT` (5, 60). Every call already sits inside a retry `try`; efetch is reached through `get_papers_by_pmids`' try.
- Test: the `test_apis.py::*_passes_a_timeout_to_every_request` tests and `::*_retries_after_a_timeout`.
- Behaviour impact: none.

### B16: unbounded CT.gov paging loops
- Files: `ctgov.py:12` (`MAX_PAGES = 1000`), the `search_with_query` loop and `ctgov.py:149`.
- Fix: both loops raise `RuntimeError` once they go past MAX_PAGES. The error is inside the retry, so it surfaces as "A CTGOV API error occurred: ... more than N pages".
- Test: `test_apis.py::test_ctgov_search_raises_when_paging_exceeds_max_pages`, `::test_ctgov_get_trials_raises_when_paging_exceeds_max_pages`, `::test_ctgov_search_topk_3000_returns_exactly_3000_ids_in_three_pages`.
- Behaviour impact: none (topk=3000 still takes 3 pages).

### B17: swallowed infrastructure errors (revised after independent audit)
- Files: `calculate_answer_score` in pubmed, pmd_no_reason, ctgov (-4); fever, fiqa, hotpotqa, msmarco_beir, nfcorpus, scifact and the 5 dense variants (-2); msmarco (-2). `calculate_answer_score_scale` in nq_serini, nq_no_reason, triviaqa, tqa_no_reason (-4) and squad, squad_no_reason (-3). `except:` changed to `except ImportError:` around `utils.java_init` in those 6 files.
- Problem: one bare `except:` covered everything. Misconfiguration (index not loaded, `search_api is None`, an empty label) turned into a silent constant penalty for the whole run, and KeyboardInterrupt was swallowed.
- Fix:
  - **Misconfiguration fails fast.** `search_api is not None` is asserted, `search_system is None` raises `RuntimeError`, an empty label raises `ValueError`, and pubmed asserts `literature_type == 'publication'`. None of these can be triggered by model output.
  - **Malformed model JSON** (`JSONDecodeError`, `KeyError`, `TypeError` when parsing) returns the HEAD constant.
  - **Any failure while running or scoring the model's query** (an API error payload, exhausted retries, ctgov `MAX_PAGES`, a pyserini error) goes through `logging.getLogger(__name__).exception(...)` with a query snippet and returns the HEAD constant. It is handled and logged, not swallowed, and a model-written query can never crash training.
- Test: `test_answer_score_errors.py`. Search failure → HEAD constant: `::test_api_reward_scores_search_api_failure_as_minus_four`, `::test_pubmed_reward_scores_pubmed_error_payload_as_minus_four`, `::test_ndcg_reward_scores_search_failure_as_minus_two`, `::test_msmarco_reward_scores_search_failure_as_minus_two`, `::test_span_reward_scores_search_failure_as_error_constant`. Logged with traceback: `::test_*_logs_search_failure_with_traceback`. Also misconfiguration raises, a non-string query is still executed like HEAD (`::test_sparse_ndcg_reward_still_executes_a_non_string_query_like_head`), and java_init ImportError is tolerated while RuntimeError propagates.
- Behaviour impact: reward values are identical to HEAD for every model output. Only misconfiguration now raises, and failures are logged with a traceback instead of a bare print. fiqa used to have no try block at all, so malformed output crashed training; it now gets -2 like its siblings.
- History: the first version of this fix let search/API errors propagate. The independent audit showed a model-written query (e.g. one containing `&pageSize=1`, or one PubMed rejects) could then crash a PPO run, so it was revised to the policy above.
- Skipped: the SQL rewards' `except Exception` around SQL execution was kept. Model-written SQL raises many exception types (sqlite3.Error, sqlite3.Warning, TypeError, OverflowError, …), so narrowing it would crash training on bad model output. screening.py was also left as is, since its excepts only cover model output.

### B18: msmarco dense fallback passed a None encoder
- Files: `msmarco.py:46-53`.
- Fix: passes `'castorini/tct_colbert-msmarco'`. pyserini's installed metadata has no encoder field. The name follows the `2cr/msmarco-v1-passage.yaml` convention (index `msmarco-v1-passage.tct_colbert-v2-hnp` ↔ encoder `castorini/tct_colbert-v2-hnp-msmarco`) and `encoded_query_info.py` (`tct_colbert-msmarco-passage-dev-subset`). If pyserini returns None, a clear `RuntimeError` is raised.
- Test: `test_msmarco.py::test_dense_fallback_passes_the_tct_colbert_query_encoder` and `::test_dense_fallback_raises_when_prebuilt_index_is_unavailable`.
- Behaviour impact: none. The primary path `/home/azureuser/cloudfiles/code/DeepRetrieval/indexes/contriever-msmarco-passage-dense-index` (contriever) is still hard-coded and was not changed. The fallback uses a different model (TCT-ColBERT) from the primary path.

### D29: hits were all parsed before scanning
- Files: `run_index_search_bm25` in nq_serini.py:138 and in nq_no_reason, squad, squad_no_reason, triviaqa, tqa_no_reason.
- Fix: it now returns a generator, and the caller iterates with `enumerate`.
- Test: `test_answer_score_errors.py::test_span_reward_parses_hits_lazily_and_stops_at_first_answer` and `::test_span_reward_ranks_answer_by_first_matching_hit`.
- Behaviour impact: none (identical ranks).

### D30: SQL searcher caching and connection closing
- Files: `get_searcher()` in bird, bird_no_reason, spider, spider_no_reason and wikisql; `src/sql/bird.py:23` and `spider.py`.
- Fix: the searcher is cached at module level. `_search` uses `contextlib.closing`.
- Test: `test_sql.py::test_sql_reward_reuses_one_searcher` and `::test_executor_closes_connection_after_query` / `_after_failing_query`.
- Behaviour impact: none.
- **Skipped: caching gold-SQL results.** The databases are not read-only: `sqlite3.connect(db_path)` runs DDL in autocommit, so a model-written `DROP TABLE` persists (verified). Caching gold results would hide that corruption. The fix I'd recommend instead is to open with `file:{db_path}?mode=ro` (needs approval).

### F40: dead code
- Deleted:
  - `verl/utils/reward_score/apis/{__init__,pubmed,ctgov}.py`; `apis/.gitkeep` was added so the key directory stays
  - `reward_score/msmarco_.py`
  - `reward_score/utils.py`
  - the `run_search_ctgov` stub and the 'trial' branch in pubmed.py
  - unused `pdb` / `ast` / `operator` imports in every reward file
  - `FaissSearcher` in the sparse nfcorpus.py

  I grepped `.py/.sh/.ipynb/.yaml/.md/.toml` for references and found none; the README only refers to the apis directory.
- Test: `test_dead_code.py` (4 parametrized + 2).
- Behaviour impact: none.

### E33: GPU preflight
- Files: `main_ppo.py:242`.
- Fix: after `ray.init`, `main` raises `RuntimeError` if `cluster_resources()['GPU'] < n_gpus_per_node * nnodes`.
- Test: `test_main_ppo.py::test_main_raises_when_cluster_has_fewer_gpus_than_requested`, `::test_main_does_not_start_training_when_gpus_are_missing`, `::test_main_starts_training_when_cluster_has_enough_gpus`.
- Behaviour impact: a run that used to hang forever now fails fast.

### G4: NDCG ideal DCG and duplicates (approved change)
- Files: `src/Lucene/utils.py:9-46`.
- Fix:
  - The ideal DCG uses relevance scores sorted in descending order.
  - A doc id retrieved more than once only gains at its first occurrence; repeats keep their rank slot with gain 0.
  - Postcondition `0 ≤ ndcg ≤ 1` and a non-negative relevance check.
  - A list target with no `rel_scores` now means binary relevance. Before, `np.array(retrieved) == target_list` raised a broadcast `ValueError` unless the lengths happened to match. No reward passes such a target: scifact uses `str(ground_truth['target'])`, a single docid.
- Test: `test_ndcg.py` (15 tests, including `ndcg_at_k(['b','a'],['a','b'],10,[1,2]) == 1` and `['a','a','a']` vs `'a'` == 1).
- Behaviour impact (approved change): NDCG rewards whose qrels are not in descending order, or whose results contain duplicates, change and are now ≤ 1. scifact changes only when a docid is retrieved more than once (hits d1, d1, x with target d1: NDCG 1.63 → 1, total reward 0.1 + 1.63 → 1.1; `test_answer_score_errors.py::test_scifact_reward_counts_a_duplicated_hit_once[sparse, dense]`). The `src/eval/*/scifact.py` scripts change the same way.

### G5: timed-out SQL scored as a match (approved change)
- Files: `src/sql/bird.py:17` and `spider.py` (timeout now returns `None`); `calculate_answer_score` in bird, bird_no_reason, spider, spider_no_reason and wikisql.
- Fix:
  - A timed-out prediction never matches and gets the mismatch score (0.3 for bird, 0 for spider).
  - A timed-out gold query is always printed as `[Error] Gold SQL timed out on <db>` and scores 0.
- Test: `test_sql.py::test_executor_returns_none_on_timeout`, `::test_sql_reward_timed_out_prediction_does_not_match_empty_gold`, `::test_sql_reward_timed_out_gold_scores_zero`, `::test_sql_reward_timed_out_gold_is_logged`, plus `::test_sql_reward_scores_matching_results_as_one` and `::..._different_results_as_mismatch` (non-timeout scoring unchanged).
- Behaviour impact (approved change): the eval scripts `src/eval/SQL/{bird,spider,bird_gpt,spider_gpt,*_case_study}.py` call `set(None)` on a timeout. That raises TypeError, which they catch and score as 0, so in eval a timeout also stops counting as a match.

# Part 3 — verl core (workers, single controller, trainer metrics)

Tests: `code/tests/deepretrieval/core/` (53 tests, CPU-only; flash-attn / vllm stubbed through `monkeypatch.setitem(sys.modules, ...)` fixtures in `core/conftest.py`). `pytest-expert` audit_tests.py: clean. NSP audit (NSP005/6/7/9) on worker_group.py, ray/base.py, protocol.py: clean.
Test ids below are relative to `code/tests/deepretrieval/core/`.

### A2 — `typing.Unpack` breaks import on Python 3.10
- Files: verl/models/transformers/llama.py:17
- Problem: `from typing import ... Unpack` exists only on Python 3.11+, so importing the module raises ImportError on the project's 3.10. No other `from typing import Unpack` in verl/ (qwen2.py was checked and imports fine).
- Fix: import `Unpack` from `typing_extensions`.
- Test: test_import_hygiene.py::test_llama_ulysses_patch_imports_on_python_310
- Behaviour impact: none (the module can now be imported).

### A7 — `raise NotImplemented(...)`
- Files: verl/utils/model.py:124
- Problem: `NotImplemented` is a constant, not an exception class. Calling it raised `TypeError: 'NotImplementedType' object is not callable`.
- Fix: `raise NotImplementedError(...)`.
- Test: test_error_paths.py::test_get_model_size_rejects_unknown_scale
- Behaviour impact: none (an unknown scale now raises the intended error type).

### A8 — NotImplementedError built but never raised
- Files: verl/workers/megatron_workers.py:256
- Problem: the exception was created and then thrown away, so a non-vLLM rollout fell through and returned unbound locals.
- Fix: added `raise`.
- Test: test_import_hygiene.py::test_no_exception_is_constructed_without_raise (an AST scan of verl/, skipping third_party/; megatron is not installed, so the module cannot be imported).
- Behaviour impact: none (that config path now fails loudly).

### A9 — `__all__` names a class that does not exist
- Files: verl/workers/rollout/naive/naive_rollout.py:33
- Fix: `'NativeRollout'` → `'NaiveRollout'`.
- Test: test_import_hygiene.py::test_naive_rollout_all_names_the_exported_class
- Behaviour impact: none.

### A10 — undefined `name` in the llama_loader f-string
- Files: verl/models/llama/megatron/checkpoint_utils/llama_loader.py:335 (`_broadcast_tp_shard_tensor_qkv`)
- Problem: NameError on the "not in state_dict" path.
- Fix: the message now uses `{q_name}, {k_name}, {v_name}`.
- Test: test_import_hygiene.py::test_file_has_no_undefined_or_shadowed_names[...llama_loader.py-F821]. Megatron is not installed, so a runtime test is impractical. The test runs ruff F821 on the file instead: 1 finding before the fix, 0 after.
- Behaviour impact: none.

### Tokenizer annotations reference unimported names
- Files: verl/workers/rollout/tokenizer.py:17-22, 88, 100
- Problem: the string annotations `"np.ndarray"`, `"torch.Tensor"` and `"tf.Tensor"` named modules that are never imported (ruff F821 ×3).
- Fix: `if TYPE_CHECKING: import numpy as np; import torch`. Removed `tf.Tensor` from the annotation and the docstring.
- Test: test_import_hygiene.py::test_file_has_no_undefined_or_shadowed_names[verl/workers/rollout/tokenizer.py-F821]
- Behaviour impact: none.

### B16 — unbounded wait for workers to come alive
- Files: verl/single_controller/base/worker_group.py:113-120 (`_block_until_all_workers_alive`), :83 (`check_workers_alive`)
- Problem: `while True` polled forever if a worker never came up.
- Fix: new parameter `max_wait_seconds=1800`. The loop keeps polling every 1 s, as before, and raises `TimeoutError('not all N workers became alive within 1800s')` when the bound runs out. `check_workers_alive` is a deliberate lifetime monitor loop on its own thread; it is marked `# nsp: ignore NSP005 <reason>`.
- Test: test_error_paths.py::test_block_until_all_workers_alive_times_out, ::test_block_until_all_workers_alive_keeps_one_second_cadence, ::test_block_until_all_workers_alive_returns_once_alive
- Behaviour impact: none in normal runs. A worker group that never comes alive now fails after 30 min instead of hanging. (No caller in verl/ currently calls `start_worker_aliveness_check`.)

### B19 — `except Exception: continue` when binding worker methods
- Files: verl/single_controller/base/worker_group.py:144-151, verl/single_controller/ray/base.py:386-393
- Problem: the comment says properties fail. In fact `getattr(cls, prop)` returns the property object, and the skip came from `assert callable(...)` raising AssertionError. That skip also disappears under `python -O`. The blanket except also swallowed real errors raised by descriptors.
- Fix: catch only `AttributeError` (instance-only descriptors). Skip non-callables with an explicit `if not callable(method): continue`. Both handlers carry an NSP007 pragma with a reason.
- Test: test_error_paths.py::test_worker_group_binding_skips_properties_and_attributes, ::test_worker_group_binding_propagates_unexpected_errors, ::test_colocated_binding_skips_properties_and_attributes, ::test_colocated_binding_propagates_unexpected_errors
- Behaviour impact: none for valid worker classes. Errors other than AttributeError raised from class attribute access now propagate.

### protocol.py bare `except:`
- Files: verl/protocol.py:36
- Fix: `except Exception:`, plus an NSP007 pragma (tensordict versions without the lazy-legacy switch need no configuration).
- Test: test_import_hygiene.py::test_protocol_import_does_not_swallow_keyboard_interrupt. Also ::test_protocol_import_tolerates_missing_lazy_legacy_switch, a guard that passes both before and after the fix. Both tests execute a throwaway copy of protocol.py, so `sys.modules['verl.protocol']` is never touched.
- Behaviour impact: none (KeyboardInterrupt/SystemExit are no longer swallowed at import).

### C21 — deprecated Ray state API import
- Files: verl/single_controller/ray/base.py:22
- Fix: `from ray.util.state import get_actor` (ray 2.59).
- Test: test_import_hygiene.py::test_ray_worker_group_imports_without_deprecation_warning (a subprocess run with `-W error::DeprecationWarning`).
- Behaviour impact: none.

### C22 — private `ray._private.services.get_node_ip_address`
- Files: verl/single_controller/base/worker.py:44
- Fix: `ray.util.get_node_ip_address()`. Checked on ray 2.59: `ray.util.get_node_ip_address is ray._private.services.get_node_ip_address` → True.
- Test: test_error_paths.py::test_node_ip_comes_from_public_ray_api
- Behaviour impact: none.

### D26 — two GPU syncs per prompt in vLLM pre-processing
- Files: verl/workers/rollout/vllm_rollout/vllm_rollout.py:57-66 (new `_pre_process_inputs_batch`), :172-173 (call site)
- Fix: the prompt batch is copied to the CPU once. The first non-pad index of every row comes from one `argmax` over the non-pad mask (argmax returns the first maximal index). Each Python row list is then sliced. An all-pad prompt raises ValueError; before, it raised IndexError from `nonzero()[0]`.
- Test: test_vllm_preprocess.py::test_batch_preprocess_equals_per_row_preprocess (output equals the old per-row `_pre_process_inputs`, including pad ids inside a prompt), ::test_batch_preprocess_rejects_all_pad_prompt
- Behaviour impact: none (identical token lists).

### D27 — several `.item()` syncs per micro-batch
- Files: verl/utils/torch_functional.py:86-98 (new `scalars_to_floats`), verl/workers/actor/dp_actor.py:279-286, verl/workers/critic/dp_critic.py:197-204
- Fix: each 0-dim tensor is upcast to **float64**, then stacked and read with one `.tolist()`. bf16, fp16, fp32 and fp64 all convert to float64 exactly, so mixed dtypes give values bit-identical to `.item()`. In the actor, kl_loss joins the same single read.
- Test: test_ppo_update.py::test_scalars_to_floats_matches_item_bit_for_bit[bf16/fp16/fp32/fp64] (exact `==` against `.item()`), ::test_scalars_to_floats_rejects_non_scalars, ::test_critic_metrics_equal_per_item_reads (bf16 vpreds, exact equality), ::test_actor_micro_batch_metrics_have_one_entry_per_micro_batch
- Behaviour impact: none.

### D28 — per-element `.item()` in `_validate`
- Files: verl/trainer/ppo/ray_trainer.py:683-686
- Fix: one `reward_tensor.tolist()`, zipped with data_sources (`strict=True`).
- Test: covered by the test_reward_metrics.py::test_validation_* tests
- Behaviour impact: none.

### E34 — CUDA-only workers fail obscurely without a GPU
- Files: verl/utils/distributed.py:18-24 (new `require_cuda(component)`), :27; verl/workers/fsdp_workers.py:55, :512, :774 (first line of each `__init__`, before `super().__init__()` and `init_process_group('nccl')`)
- Fix: raises `RuntimeError("<component> requires a CUDA GPU (it uses NCCL and CUDA kernels); no CUDA device is available on this machine")`.
- Test: test_error_paths.py::test_require_cuda_passes_when_cuda_present, ::test_require_cuda_names_the_component, ::test_initialize_global_process_group_requires_cuda, ::test_fsdp_worker_requires_cuda[ActorRolloutRefWorker|CriticWorker|RewardModelWorker]
- Behaviour impact: none on GPU machines.

### E35 — unguarded flash-attn imports
- Files: verl/utils/import_utils.py:41-44 (new `flash_attn_import_error(component)`); verl/workers/actor/dp_actor.py:34-39 and verl/workers/critic/dp_critic.py:34-39 (module level); verl/workers/fsdp_workers.py:862-865; verl/utils/torch_functional.py:~327, ~359 (function level)
- Fix: `raise flash_attn_import_error(...) from e`, whose message reads "flash-attn is required for X; it is CUDA-only, install with `uv sync --extra gpu` on Linux".
- Test: test_error_paths.py::test_dp_module_import_explains_missing_flash_attn[actor|critic], ::test_rmpad_logprob_helpers_explain_missing_flash_attn[2 fns], ::test_reward_model_forward_explains_missing_flash_attn
- Behaviour impact: none (only the error message changes).

### F40 — dead code and mutable class attributes
- Files: verl/utils/seqlen_balancing.py:60-61 (deleted the dead `spread` method, which the property shadowed); verl/trainer/fsdp_sft_trainer.py:351 (deleted the self-import); verl/utils/dataset/rl_dataset.py:92 (deleted the duplicate local import); verl/single_controller/base/worker.py:89 (`'0'` default); worker.py:71 `WorkerMeta.keys` and verl/utils/tracking.py:25 `supported_backend` → tuples. A grep found only iteration and `in` membership, with no mutation or list concatenation.
- Test: test_import_hygiene.py::test_file_has_no_undefined_or_shadowed_names[seqlen_balancing|fsdp_sft_trainer|rl_dataset-F811] (ruff F811: 1 finding each before, 0 after); test_error_paths.py::test_worker_meta_keys_is_immutable, ::test_tracking_supported_backends_is_immutable. The `'0'` default has no test because `int(0) == int('0')`.
- Behaviour impact: none.

### G1 — KL loss sign, and the kl_loss metric overwritten each micro-batch (APPROVED CHANGE)
- Files: verl/workers/actor/dp_actor.py:272-273, 279-286
- Problem: `policy_loss - kl_loss * coef` rewarded divergence from the reference policy. Also, `metrics['actor/kl_loss']` was overwritten on every micro-batch, so only the last value survived.
- Fix: `policy_loss + kl_loss * kl_loss_coef`. kl_loss now goes through `append_to_dict` together with the other per-micro-batch metrics, as a list later reduced by `reduce_metrics`. `actor/kl_coef` stays a scalar.
- Test: test_ppo_update.py::test_kl_loss_increases_actor_loss (loss per micro-batch 0 → +0.1 for coef 0 → 0.4; the buggy code gave −0.1), ::test_kl_loss_gradient_pulls_policy_toward_reference (dL/dw = +coef), ::test_kl_loss_metric_keeps_every_micro_batch. All three run the real `update_policy` on CPU with a one-parameter fake policy (`_forward_micro_batch`/`_optimizer_step` monkeypatched, `TensorDict.cuda` set to identity).
- Behaviour impact: **approved change.** With `use_kl_loss=True` (GRPO configs), the KL term now penalizes divergence from the reference, as intended. Configs with `use_kl_loss=False` (the default) are unaffected. `actor/kl_loss` is now the mean over micro-batches, not the last value.

### G3 — training and validation reward metrics vs the reward tables (APPROVED CHANGE)
- Files: verl/trainer/ppo/ray_trainer.py:260-296 (new shared constants and helpers `ANSWER_SPAN_HIT_MIN_REWARD`, `compute_answer_span_hit_ratios`, `compute_mean_ndcg`, `compute_sql_accuracy`), `compute_reward_metrics_nq_serini` (~336-347), `compute_reward_metrics_ndcg` / `_recall_ndcg` (~382-420), `compute_reward_metrics_sql` (~424-455), `_validate` (~683-712)
- Reward tables read from verl/utils/reward_score/{nq_serini,squad,triviaqa,nq_no_reason,squad_no_reason,tqa_no_reason}.py. The total reward is format 1 + {5, 4, 2, 1, 0.5, 0.1} for rank ≤ {5, 20, 50, 100, 1000, 3000}, so hit@k ⇔ reward ≥ {6, 5, 3, 2, 1.5, 1.1}. A miss is 1−3.5 (squad 1−2.5), a malformed answer 1−4 (squad 1−3), a bad format −4. Training had hit@50 at `>= 4`; no reward of 4 exists, so hit@50 equalled hit@20. Validation had the right values but hard-coded equalities.
- Branches changed:
  - nq_serini/squad/triviaqa: training and validation now share `compute_answer_span_hit_ratios`. Validation keys are now `val/recall@{k}/{data_source}` for k ∈ {5, 20, 50, 100, 1000, 3000}; before, the source was missing and sources overwrote each other. Training `reward/wrong_answer_ratio` was `<= -1.5`, which also counted format errors (−4). It is now −4 < r < 1.1: well-formatted answers with no hit.
  - msmarco (training `compute_reward_metrics_recall_ndcg`): it subtracted a 0.2 recall bonus that msmarco.py no longer adds (reward = 0.1 + ndcg), so it under-reported ndcg. It now uses the shared `compute_mean_ndcg`, as do `compute_reward_metrics_ndcg` and both validation ndcg branches (formula unchanged there; `reward/ndcg` is now a float, not a tensor).
  - SQL (bird/spider/wikisql): accuracy is shared through `compute_sql_accuracy` (reward > 1.0), which training and validation already agreed on. Training `wrong_answer_ratio` counted `== 0.1`, a value no SQL reward produces. It now counts 0 ≤ r ≤ 1 (0, or bird's 0.1 + 0.3).
  - Unchanged: search_engine (pubmed/ctgov) training thresholds match pubmed.py/ctgov.py. The validation branch for pubmed/ctgov still emits no metric (`pass`, pre-existing, not one of the two defects). The validation `else` branch (`reward == 3`) serves unknown sources with no table to check against. `compute_reward_metrics_recall_at_k` and `compute_reward_metrics_` are unused.
- Test: test_reward_metrics.py::test_training_hit_at_k_matches_reward_table, ::test_training_wrong_answer_excludes_format_errors, ::test_training_ndcg_metric_for_msmarco_matches_reward_table, ::test_training_sql_metrics_match_reward_table, ::test_validation_recall_at_k_matches_training_hit_at_k, ::test_validation_recall_keys_are_per_data_source, ::test_validation_sql_accuracy_uses_shared_threshold, ::test_validation_ndcg_uses_shared_formula (the last two are guards that already passed). `_validate` runs for real against a SimpleNamespace trainer with a fake rollout worker group.
- Behaviour impact: **approved change (metrics only, no effect on training).** The hit@50 and msmarco ndcg curves change. Wrong-answer ratios for nq/squad/triviaqa and SQL change. Validation recall keys gain a `/{data_source}` suffix plus hit@1000/3000, so dashboards keyed on `val/recall@5` need updating.

### Not changed (as instructed)
- use_dynamic_bsz loss scaling in dp_actor; FSDP set_state_dict_type migration.

# Part 4 — Eval scripts, preprocessing, test hygiene, README

Every test node below lives in `code/tests/deepretrieval/evals/` unless a path says otherwise. Shared loader: `tests/deepretrieval/evals/_loader.py`. It runs a `src/` script with its SDK imports replaced by stub modules through `monkeypatch`, or compiles one function straight from the script's AST. No test uses the network, the JVM or a GPU.

### A1 — `extract_solution` returns an unbound `processed_str` when the header is missing
- **Files:** src/eval/BM25/{msmarco.py:57, nq.py:51, squad.py:51, triviaqa.py:51}, src/eval/Dense/msmarco.py:63, src/eval/SearchEngine/{eval_ctgov.py:47, eval_pubmed.py:47, eval_pubmed_no_reason.py:47}. A grep for the "Failed to locate model response header" branch finds no other copies. The SQL copies have this branch commented out.
- **Problem:** `return None, processed_str` runs before `processed_str` is assigned, so it raises UnboundLocalError.
- **Fix:** `return None, solution_str`.
- **Test:** `test_extract_solution.py::test_missing_header_returns_no_answer_and_the_raw_text` (8 params). Also `::test_header_then_answer_returns_last_answer`.
- **Behaviour impact:** none. Every caller passes the result to `json.loads(...)` inside `try/except Exception` (or a bare except). Before the fix the UnboundLocalError reached that handler; now `json.loads(None)` raises TypeError and reaches the same handler. Metrics are unchanged.

### A12 — error print in evidence_gpt_claude{,_no_reason}.py ~250 uses `generated_text`
- **Files:** src/eval/baselines/evidence_gpt_claude.py:242-250, evidence_gpt_claude_no_reason.py:239-247
- **Problem as reported:** `generated_text` may be unbound or stale.
- **Finding:** not a bug. When `response is None` the code appends zeros and `continue`s, and appending cannot raise. The next statement, `generated_text = response`, cannot raise either. So whenever the handler runs, `generated_text` is bound to the current response.
- **Fix:** skipped. No change was made, because no failing test can be written for it.
- **Behaviour impact:** none.

### A13 — query_rewrite.py asks for a model the host script doesn't serve
- **User-visible change:** the default model is now the NQ model, matching the authors' latest `vllm_host.sh` (commit 001a563). Pass `--model DeepRetrieval/DeepRetrieval-TriviaQA-BM25-3B` for the old default.
- **Files:** src/query_rewrite.py:7-9, 57-58, 67, 78-86, 88-99; vllm_host.sh:4
- **Problem:** the script hardcodes `DeepRetrieval-TriviaQA-BM25-3B`, but `vllm_host.sh` serves `DeepRetrieval-NQ-BM25-3B`. The request has no timeout. Exceptions are re-raised without their cause. `main()` exits 0 on error.
- **Fix:**
  - New `DEFAULT_MODEL` (the NQ model) and a `--model` flag, passed to `rewrite_query(..., model=)`.
  - `requests.post(..., timeout=120)`.
  - `raise RuntimeError(...) from e` (and `ValueError ... from e` in `extract_query`).
  - `main` prints errors to stderr and calls `sys.exit(1)`.
- **Test:** `test_query_rewrite.py`:
  - `::test_default_model_is_the_one_vllm_host_serves` parses `vllm_host.sh`.
  - `::test_request_has_a_timeout`
  - `::test_model_flag_is_sent_to_the_server`
  - `::test_request_failure_keeps_the_original_cause`
  - `::test_main_exits_nonzero_on_error`
- **Behaviour impact:** the default model changes to the hosted one. Errors now exit 1. The exception type changes from `Exception` to `RuntimeError`, a subclass, so existing handlers still catch it.

### B17 — bare `except:` in src/ and utils/
- **Files:** 46 sites in 28 files. The full list is in `ruff check --select E722` at HEAD: src/eval/BM25/{*_base.py, nq/squad/triviaqa.py, baselines/model_generate/*}, src/eval/Dense/baselines/**, src/eval/SearchEngine/eval_{ctgov,pubmed,pubmed_no_reason}.py, src/eval/baselines/evidence_gpt_claude{,_no_reason}.py.
- **Fix:**
  - `try: import utils.java_init` guards (nq_base:8, squad_base:7, triviaqa_base:7) become `except ImportError:`.
  - Every other site becomes `except Exception:`.
  - Control flow is otherwise unchanged.
  - utils/java_init.py:33 already used `except Exception as e:`, so nothing changed there.
  - query_rewrite.py had no bare except.
- **Test:**
  - `uv run ruff check src utils --select E722`: 46 findings before, "All checks passed!" after.
  - Representative unit test: `test_evidence_baselines.py::test_query_extraction_does_not_swallow_keyboard_interrupt` (2 params). It failed with "DID NOT RAISE" before the fix.
- **Behaviour impact:** none for ordinary errors. KeyboardInterrupt and SystemExit now propagate.

### B20 — test_tensor_dict_make_iterator could not fail on a label mismatch
- **Files:** tests/utility/test_tensor_dict_utilities.py:81-89
- **Problem:** the label check only printed, and `assert False` sat inside an `if`.
- **Fix:** the test now checks:
  - the iteration count is exactly 20;
  - every batch is a DataProto;
  - `torch.equal` on the stacked obs;
  - `np.array_equal` on the stacked labels.
- **Test:** `tests/utility/test_tensor_dict_utilities.py::test_tensor_dict_make_iterator` passes.
  - Red check, a throwaway copy with the second iterator using `dataloader_kwargs={'shuffle': True}`: failed at `torch.equal`.
  - Separate labels-only check: `np.array_equal` fails on swapped labels.
  - Note: changing only `seed` does not change the data. `make_iterator` doesn't shuffle unless `shuffle` is passed, so the seed has no effect.
- **Behaviour impact:** none (test only).

### C23 — deprecated `local_dir_use_symlinks=False`
- **Files:** code/download_datasets.py:99
- **Problem:** huggingface_hub 0.36.2 ignores this argument and warns. Checked with `inspect.getsource(hf_hub_download)`: "`local_dir_use_symlinks` parameter is deprecated and will be ignored".
- **Fix:** removed the argument.
- **Test:** `test_download_datasets.py::test_download_omits_deprecated_symlink_argument`
- **Behaviour impact:** none. The argument was already ignored; the warning is gone.

### C24 — legacy `langchain.schema` import
- **Files:** src/utils/gpt_azure.py:7
- **Problem:** `langchain` 1.x has no `langchain.schema`. Before the fix, importing gpt_azure failed with "No module named 'langchain'" (seen as a red failure in `test_api_clients`).
- **Fix:** `from langchain_core.messages import HumanMessage`.
  - Context7 `/websites/langchain_oss_python_langchain` shows v1 `HumanMessage` exported from `langchain.messages`, which re-exports `langchain_core.messages`. Legacy APIs moved to `langchain-classic`.
  - `inspect` with the baselines extra (langchain-core 1.7.3) confirms `HumanMessage(content=...)`.
- **Test:** `test_api_clients.py::test_client_module_imports_without_key_files[src/utils/gpt_azure.py-stubs1]` and `::test_gpt_azure_exports_key_from_file_on_first_call`.
- **Behaviour impact:** none.

### C25 — `eval(item['target'])` in msmarco_subset search baselines
- **Files:** src/eval/BM25/baselines/search/msmarco_subset.py:64, src/eval/Dense/baselines/search/msmarco_subset.py:95
- **Problem:** this runs arbitrary code from a results JSON file.
- **Fix:** `ast.literal_eval`. Targets are written as `str(list_of_str)`, which `literal_eval` parses identically.
- **Test:** `test_msmarco_subset_targets.py::test_target_that_is_code_is_rejected_not_executed` (2 params). Before the fix the payload ran.
- **Behaviour impact:** none for valid targets.

### D31 — OpenAI client per call; key files read at import
- **User-visible change:** environment variables (`OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`) now take precedence over the `./*.key` files. HEAD always used the key files, and gpt_azure overwrote `AZURE_OPENAI_API_KEY` from its file. With an ambient `OPENAI_API_KEY` set, the baselines run on that account. An empty key file now raises instead of sending an empty key. `claude_aws.py` no longer writes AWS keys into `os.environ`.
- **Files:** src/utils/gpt.py:5-36, gpt_azure.py:15-27 (+ one `_configure_azure()` line per function), claude_api.py:5-33, claude_aws.py:7-55, new src/utils/api_keys.py
- **Problem:**
  - `gpt_chat`/`gpt_chat_35` each built a new `OpenAI` client that shadowed the module one.
  - All four modules `open()`ed `./*.key` at import, so importing from any other cwd crashed.
- **Fix:**
  - `load_secret(env_var, key_file)` returns the env var if set, otherwise the key file (cwd-relative, as before). Otherwise it raises a RuntimeError that names both sources.
  - gpt.py: one lazily created client (`get_client()`), using `OPENAI_API_KEY`.
  - gpt_azure: `_configure_azure()` runs on each call and sets `AZURE_OPENAI_API_KEY`; endpoint and version are assigned as before.
  - claude_api: lazy boto3 client. Credentials come from `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, else aws_access.key / claude_api_aws.key.
  - claude_aws: lazy AnthropicBedrock client. Credentials come from the AWS env vars, else the two-line claude_api_aws.key; a malformed file gets a clear error.
- **Test:** `test_api_clients.py`, 14 tests:
  - import without key files (4 modules)
  - client reused and built from env
  - key-file fallback
  - clear error when neither source exists (each module)
- **Behaviour impact:**
  - Credential errors now appear on the first call instead of at import.
  - claude_aws.py no longer writes the AWS keys into `os.environ`. The only consumer was the unused `ChatAnthropic` import.
  - Running `python src/utils/gpt.py` directly now needs `code/` on `sys.path`, because it imports `src.utils.api_keys`. All real callers import the module as `src.utils.*`.

### D32 — msmarco_sub preprocessing reloads the dataset for each retrieval mode
- **Files:** data_preprocess/msmarco_sub.py:104-112, data_preprocess/no_reason/BM25/msmarco_sub.py:101-109
- **Fix:**
  - Loops reordered to `for domain` (load once) → `for retrieval_mode`.
  - `make_map_fn(split, retrieval_mode=retrieval_mode)` (B023).
- **Test:**
  - `test_msmarco_sub_preprocess.py::test_each_domain_is_loaded_once` (2 params). Red: 8 loads instead of 4.
  - `::test_each_mode_output_is_tagged_and_prompted_for_that_mode` (4 params).
  - Identity check: all 48 parquet outputs of both scripts, before vs after on synthetic raw data, are byte-for-byte equal via `to_json` (script output "identical parquet files: 48").
- **Behaviour impact:** none. Files are now written domain-major instead of mode-major; the contents are identical.

### F36 — `baselines` extra
- **Files:** code/pyproject.toml (`[project.optional-dependencies].baselines`), code/uv.lock
- **Fix:** the extra contains exactly what src/ imports:
  - `openai`, `anthropic`, `boto3`, `langchain-core`, `langchain-openai`, `langchain-anthropic`, `langchain-community`, `trl`.
  - `langchain` is not included, because after C24 nothing imports it.
  - Nothing is pinned.
  - Ran `uv lock` and `uv sync --extra retrieval --extra baselines`.
  - The resolver picked trl 0.17.0, which is compatible with transformers<4.48, and openai 3.41.0.
  - `uv lock` downgraded python-dateutil from 3.0.0 to 2.9.0.post0 (botocore needs <3). That applies to the base env too.
- **Test:** `uv lock` succeeded and the sync installed the packages. `inspect` checks in C23 and C24.
- **Behaviour impact:** dateutil 2.9.0.post0 instead of 3.0.0 everywhere.

### F37 — README
- **Files:** README.md, Installation (~61-97), API section (~99-106), Run Training (~170), Dependency Conflicts (~256-286)
- **Fix:**
  - Installation now uses the uv flow: `cd code && uv sync`, `--extra retrieval`, `--extra gpu` (Linux+CUDA), `--extra baselines`, `uv run pytest`, Python 3.10. The JDK install moved to `uv run`, and there is a note on credential env vars.
  - API commands: `sh ../vllm_host.sh` and `python src/query_rewrite.py` (plus `--model`).
  - `conda activate zero` → `source .venv/bin/activate`.
  - The numpy/vllm conflict instructions are replaced by a short note, because the lock pins numpy<2, transformers<4.48 and vllm 0.6.3, so those conflicts can't happen.
  - The SLURM CUDA section is kept as it was.
- **Test:** manual review of the rendered diff.
- **Behaviour impact:** docs only.

### F39 — tests hygiene
- **Files:**
  - tests/sanity/test_import.py:16-29 now asserts `__version__` equals the version file.
  - tests/ray/test_driverfunc_to_worker.py and test_worker_group_torch.py: module-level `os.environ` writes became an autouse `monkeypatch.setenv` fixture, which runs before `ray.init`.
  - tests/ray/test_check_worker_alive.py:27 now uses a path relative to `Path(__file__).parent`.
  - tests/ray/test_rvdz.py:31 got the `# audit: ignore[NO_ASSERT]` pragma.
  - tests/verl/utils/dataset/test_rl_dataset.py:52-54 asserts `len == 16` and `input_ids.shape == (16, 256)`. RLHFDataset left-pads to `max_prompt_length` and collate stacks the rows.
  - tests/e2e/envs/digit_completion/tokenizer.py:140: `"DigitCompletionTokenizer"` → `"CharTokenizer"` (F821).
  - tests/model/test_transformer.py:98-99: `setattr` → attribute assignment (B010).
- **Test:**
  - `uv run pytest tests/sanity` passes.
  - ruff F821/B010 are clean.
  - ray, model and data tests are auto-skipped on this Mac.
- **Behaviour impact:** none.

### G2 — `llm="hiku"` typo (approved behaviour change)
- **TDD note:** on HEAD the test errored in fixture setup (`FileNotFoundError ./aws_access.key`, the import-time key read fixed by D31) instead of failing its assertion. With D31 in place its assertion rejects `"hiku"`, which I confirmed by reverting the one-word fix.
- **Files:** src/eval/baselines/evidence_gpt_claude.py:164, evidence_gpt_claude_no_reason.py:161
- **Problem:** `claude_api.get_claude_response` looks up `BEDROCK_MODEL_NAME_MAP`, whose keys are claude, haiku, sonnet and opus. `"hiku"` raised ValueError, retries ran out and returned None, so every Claude-3-Haiku query scored 0.
- **Fix:** `llm="haiku"`.
- **Test:** `test_evidence_baselines.py::test_every_claude_model_name_is_accepted_by_get_claude_response` (2 params) checks the name against the real map loaded from claude_api.py.
- **Behaviour impact (approved):** claude3 (Haiku) baselines now produce real scores instead of all zeros. Related: `eval_ctgov_load.py:88` reads a file named `llm_responses_claude-hiku.json`. That file was probably produced by a run with the same typo; I left it unchanged.

### G7 — consistent failure handling (approved behaviour change)
- **Fix:** every failed or unparseable query now scores 0 and stays in the denominator.
- **Test:**
  - `test_failure_scoring.py::test_msmarco_ndcg_counts_unparseable_response_as_zero` (BM25, Dense). Red: 1.0, green: 0.5.
  - `::test_answer_filter_counts_unparseable_generation_as_zero`
  - `::test_answer_filter_counts_unparseable_injection_check_as_zero_after_cleaning`
  - `::test_answer_filter_scores_each_query_once_when_injection_handling_fails`. Red showed double counting: original recall 2/3.
- answer_filtered_test.py also gained a check before averaging that each metric list has exactly `len(generations)` entries. It raises RuntimeError otherwise.

| Script | Old policy | New policy |
|---|---|---|
| BM25/msmarco.py | failure dropped (not in NDCG denominator) | 0, counted |
| Dense/msmarco.py | failure dropped | 0, counted |
| baselines/answer_filtered_test.py | unparseable generation (`"error"`) skipped in both metric sets; injection check unparseable → dropped from cleaned set; exception after the original score → original counted twice (score + 0) | 0 in both sets; cleaned 0; pad only metrics not yet scored for that query (original keeps its real score) |
| BM25/{nq,squad,triviaqa}.py | 0, counted | unchanged |
| SearchEngine/eval_{ctgov,pubmed,pubmed_no_reason}.py | 0, counted | unchanged |
| SearchEngine/eval_{ctgov,pubmed}_reason.py, eval_{ctgov,pubmed}_no_reason_gpt.py, eval_ctgov_load.py | 0, counted (`batch_recalls.append(0)`) | unchanged |
| SearchEngine/eval_ctgov_base.py | 0, counted | unchanged |
| SearchEngine/eval_pubmed_base.py | no per-query handler (an exception aborts the run) | unchanged |
| SQL/{bird,spider}{,_gpt,_case_study}.py | 0.0, counted | unchanged |
| baselines/evidence_gpt_claude{,_no_reason}.py | 0, counted (`compute_filtered_recall` is defined but unused) | unchanged |
| BM25/{*_base,fever,fiqa,hotpotqa,msmarco_beir,nfcorpus,scifact}.py, Dense/{fever,hotpotqa,msmarco_beir,msmarco_sub,nfcorpus,scifact}.py, {BM25,Dense}/baselines/search/*.py | no per-query handler (raw/stored queries; an exception aborts the run) | unchanged |
| {BM25,Dense}/baselines/model_generate*/** | generation and post-processing only; no metric | unchanged |

Out of scope: in `BM25/baselines/model_generate/claude.py` and `gpt.py` (and their Dense siblings), if all 10 API attempts fail, `decoded` is left over from the previous query (or raises NameError on the first one). The stale text then gets scored. Fixing that changes generation, not aggregation, so I left it unchanged.

### Not approved, not done
URL-encoding queries, sorting JDK versions in java_init, batching the Dense msmarco_subset searches, and the "rate limiter only warns" behaviour in eval_pubmed_reason.
