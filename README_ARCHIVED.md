# ⚠️ ARCHIVED — superseded by *George* + *biomni-fork*

**Date archived:** 2026-07-21 · **Tag:** `archive-pre-migration`

This repository (`Biomni_Rpts_Ds`) is the **original fork** in which the
**5′UTR GCN tandem-repeat pathogenicity analysis** was developed. It has been
**migrated into two clean successor repositories** and is now frozen, read-only,
for provenance. Do not develop here.

## Live successors

| Repo | Path | Role |
|---|---|---|
| **George** | `/blue/zhou/leizhou/Agents/George` · https://github.com/leizhou69/George | The clean project repo — harness, queries, analysis, the AlphaGenome **DuckDB/Parquet backend**, results. |
| **biomni-fork** | `/blue/zhou/leizhou/Agents/biomni-fork` | The patched Biomni package, now an **editable install** (`pip install -e`). See its `PATCHES.md`. |

## What changed in the migration

1. **Biomni is no longer vendored.** The old `sys.path.append("./")` shadowing is
   gone; `import biomni` resolves to the editable `biomni-fork`. The two
   load-bearing patches (temperature/max_tokens in `llm.py`, the `a1.py` prefill
   fix) moved to the fork and are documented in `biomni-fork/PATCHES.md`.
2. **AlphaGenome data moved to a DuckDB/Parquet star schema.** The three ~30 GB
   5′UTR AG TSVs were converted to a compact (~2 GB) partitioned backend
   (`George/data/ag_db/`), validated bit-faithful, and the agent now queries it
   via `query_ag` instead of loading TSVs. The **source TSVs have been deleted**
   (Phase 7); the data is preserved in the DuckDB backend.

## What remains here (why this repo is kept)

- Full git history / provenance for the manuscript (`Rpt_Ds/ms/`).
- **~292 GB of prior run outputs** under `Rpt_Ds/output/` (Dec_25, March_26,
  March_26_B, July_20_2026, …) — the historical experiment record. Left in place.
- The 15 GB Biomni data-lake cache (`biomni_data_cache/`).

The migration plan and running log live in **George** (`upgrade.md` /
`upgrade_log.md`) — that is the authoritative record.
