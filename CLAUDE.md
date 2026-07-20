# CLAUDE.md

Guidance for Claude Code (and future you) when working in this repo.

## What this repo is

A **fork of the Biomni biomedical agent** (`biomni/`) wrapped by a small
**batch-experiment harness** (`run_biomni.py` + `*.sbatch`) that runs one
biomedical analysis task across several LLM models and temperatures on a SLURM
cluster, then collects the outputs under `Rpt_Ds/output/` for comparison.

The scientific task itself lives in `Rpt_Ds/query_*.txt` (currently
`query_05_b.txt`): an exploratory analysis of what distinguishes disease-causing
5'UTR GCN tandem-repeat loci, using AlphaGenome expansion predictions plus
supplied `.tsv`/`.txt` data. The agent is asked to form hypotheses, run stats,
and emit `pathogenic_repeat_analysis.ipynb`, `Candidate_Identification.ipynb`,
and `Top_Candidate_Pathogenic_repeats.csv`.

`Rpt_Ds/` also holds the human-side comparison/verification work
(`CompareOutput*.ipynb`, `verify_q05_*.md`, `Evaluate.ipynb`) — that is analysis
of agent runs, not part of the agent itself.

## How it runs

Environment: conda env **`biomni_e1`** (the sbatch scripts do `ml conda; conda
activate biomni_e1`). API keys come from `.env` (already populated locally;
`.env.example` is the template).

```bash
# Direct, from repo root:
python run_biomni.py s5 o4.8                       # run named model keys
python run_biomni.py o4.8 --temperatures 0.7       # one temperature
python run_biomni.py                               # no args → uses LLM_CONFIGS block

# On SLURM:
sbatch 512GB.sbatch      # runs `python run_biomni.py s5 o4.8 --output-dir Rpt_Ds/output/March_29`
sbatch 196GB.sbatch      # runs `python run_biomni.py` with the LLM_CONFIGS defaults
```

`run_biomni.py` loops over **models × temperatures**, and for each combo:
`cd`s into a fresh `Rpt_Ds/output/<prefix>/` dir, builds an `A1` agent
(`path=./biomni_data_cache`, `timeout_seconds=12000`), registers the
`DATA_FILES` via `agent.add_data(...)` as absolute paths, runs `agent.go(prompt)`,
then writes a `.log` trace, a `.md` transcript, and extracted `media/*.png`.

### Model keys (`MODEL_NAMES` in run_biomni.py)
`s5`→`claude-sonnet-5`, `o4.8`→`claude-opus-4-8`, `s4.6`→`claude-sonnet-4-6`,
`o4.6`→`claude-opus-4-6`, `o4.5`→`claude-opus-4-5-20251101`, plus gpt5/gpt5.2/
gemini/grok keys. Source (Anthropic/OpenAI/Gemini/xAI/…) is auto-detected from
the model-name prefix in `biomni/llm.py::get_llm`.

### Config knobs (top of `run_biomni.py`)
`QUERY_FILE`, `QUERY_ID`, `LLM_CONFIGS` (fallback models when no CLI args),
`TEMPERATURES` (default `[0.5, 0.7, 0.9]`), `TIMEOUT_SECONDS` (12000),
`DATA_FILES` (name → path under `Rpt_Ds/data/`).

## ⚠️ Known issues / gotchas (read before running)

1. **`temperature` is a hard blocker on Opus 4.8 and Sonnet 5.** The most recent
   run (`slurm-37585921.out`, 2026-07-19) failed **every** experiment with:
   `anthropic.BadRequestError 400 … '`temperature` is deprecated for this model.'`
   Cause: `run_biomni.py` sets `default_config.temperature = temperature`
   (0.5/0.7/0.9), and `biomni/llm.py`'s Anthropic branch passes
   `temperature=temperature` into `ChatAnthropic(...)`. Newer Anthropic models
   (opus-4-8, opus-4-7, sonnet-5) reject `temperature` (and `top_p`/`top_k`).
   **Fix options:** in `biomni/llm.py` (Anthropic branch) omit `temperature` for
   these models, OR sweep something other than temperature (the temperature loop
   is meaningless for models that ignore it). Note: `claude-sonnet-5` itself
   *resolved* fine on the API — the only error was `temperature`, so the model
   ID is valid; do not "fix" it back to sonnet-4-6.

2. **`max_tokens` is hardcoded** in `biomni/llm.py`: the Anthropic branch is set
   to **32000**; the Custom (SGLang) branch is still 8192. Opus 4.8 supports up
   to 128K output (streaming). If long analytical deliverables still truncate
   (`stop_reason: max_tokens` / cut-off notebooks), raise the Anthropic value
   further — note the SDK requires streaming above ~16K, which this call path
   does not currently use.

3. **S3 data-lake 403s (resolved).** Older runs logged repeated
   `403 Client Error: Forbidden … biomni-release.s3.amazonaws.com/data_lake/…`.
   The real Biomni data lake is already complete locally (76/76 files at
   `./biomni_data_cache/biomni_data/data_lake`), so those 403s were **not** the
   real lake failing — `add_data` writes each custom dataset into the cached
   `biomni.env_desc.data_lake_dict`, which is shared across every `A1()` in the
   process, so later experiments in the same run tried to download the custom
   `DATA_FILES` names from S3 (→ 403) before re-registering them locally. Fixed
   by passing `expected_data_lake_files=<local files>` to `A1()` in
   `run_biomni.py`, which makes A1 skip **all** S3 downloads and use the local
   lake (fully offline). If you point `data_root` at a different local lake,
   ensure it holds all 76 files listed in `biomni/env_desc.py::data_lake_dict`.

4. **Working-directory side effect.** `run_single_experiment` does
   `os.chdir(log_dir)` so agent-generated files land in the output folder. It
   restores `cwd` in a `finally`, but if you add code, remember paths are
   relative to the per-experiment dir mid-run (that's why `DATA_FILES` are
   resolved to absolute paths against `original_cwd`).

5. **Uncommitted local edit in `biomni/agent/a1.py`:** the tool `observation` is
   appended as `HumanMessage` instead of `AIMessage` (see
   `Rpt_Ds/fix_prefill_20260327.md`). This is an intentional prefill-related fix
   for newer models — keep it.

## Output layout
`Rpt_Ds/output/<QUERYID>_<modelkey>_Tmp<temp>_<timeoutid>_<timestamp>/`
containing `<prefix>.log`, `<prefix>.md`, and `media/plot_*.png`. sbatch tee's a
top-level `.log` next to it.

## Conventions
- API keys: `.env` only — never hardcode or commit them (`.gitignore` covers `.env`).
- This is a fork; keep divergence from upstream Biomni minimal and documented
  (see `Rpts_Ds_readme.md` and the `verify_q05_*.md` / `fix_prefill_*.md` notes).
- Don't commit large agent outputs, `biomni_data_cache/`, or `biomni_env/`.
