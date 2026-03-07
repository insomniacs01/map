# Coop Contract Index Schema (Dataset-Agnostic)

This repo uses **frames contracts** (JSON) to make training/eval *reproducible* and *fair* across runs/models.

To make this portable across datasets, we separate:
- a **dataset-specific index builder** (produces a parquet index with precomputed pairing + baseline distance),
- a **dataset-agnostic contract generator** (samples/buckets from that index into a JSON contract).

Related:
- Bucket protocol + contract schema: `docs/coop_bucket_protocol.md`
- Contract bucketing (contract-only when `baseline_distance_m` is present): `scripts/bucket_frames_by_baseline_distance.py`

---

## 1) Generic Index Parquet Schema

The **index parquet** is a row-per-frame table with (at least) the following columns:

Required columns:
- `split` (string): split name (e.g. `train`, `validate`, `test`)
- `sequence` (string): sequence/scene id
- `frame` (string/int): frame id (will be stringified in contracts)
- `main_agent` (string/int): the main/ego agent id for this cooperative sample
- `coop_agents` (list[string]/list[int]): ordered cooperative agent ids for this sample
  - must include `main_agent`
  - recommended ordering: `coop_agents[0] == main_agent`
- `baseline_distance_m` (float): cooperative baseline difficulty key (meters)
  - recommended definition: `max_i ||(x_i, y_i) - (x_main, y_main)||_2`

Optional columns:
- `dataset` (string): dataset identifier (useful when you want to merge multiple datasets into one index/contract)
  - If present, `make_frames_contract_from_index.py` will copy it into each `frames[]` entry and compute
    a dataset-aware hash `frames_hash_md5_v3`.
  - When working with a merged multi-dataset index, both `report_index_bucket_stats.py` and
    `make_frames_contract_from_index.py` support `--dataset ...` to filter to a single dataset id.
  - To merge multiple generic indices into one parquet, use `scripts/merge_coop_index_parquets.py` and
    set stable dataset ids via `--datasets ...`.
  - Practical note: actually *loading* multiple datasets from one contract still requires dataset code that routes
    `frame.dataset` to the right data root. If you don't want to change loaders, a simpler option is to **prefix**
    `sequence` with `dataset` to avoid collisions and keep a single-root loader.
- `has_assets` (bool): whether the referenced sample has all required assets
  - if `--require_assets` is set, `make_frames_contract_from_index.py` will skip rows where `has_assets` is falsy

You may include additional columns freely; they are ignored by the generic scripts.

---

## 1.1 Multi-dataset flow (index -> merge -> stats -> contracts)

When you have multiple coop datasets (or multiple index shards), the recommended flow is:

1) Build a **per-dataset** generic index parquet (same schema).
2) Merge them into one parquet (ensure stable `dataset` ids):
   ```bash
   PYTHONPATH=$(pwd) python scripts/merge_coop_index_parquets.py \
     --index_parquets /path/to/dsA_index_generic.parquet /path/to/dsB_index_generic.parquet \
     --datasets dsA,dsB \
     --require_dataset \
     --out_parquet data/merged/coop_index_merged.parquet
   ```
3) Report stats per split (optionally per dataset):
   ```bash
   PYTHONPATH=$(pwd) python scripts/report_index_bucket_stats.py \
     --index_parquet data/merged/coop_index_merged.parquet \
     --split test \
     --dataset dsA \
     --require_assets \
     --preset fine_eval \
     --sample_size 500
   ```
4) Generate contracts (main/stress/full) per split:
   ```bash
   PYTHONPATH=$(pwd) python scripts/make_frames_contract_from_index.py \
     --index_parquet data/merged/coop_index_merged.parquet \
     --splits train,validate,test \
     --dataset dsA \
     --require_assets \
     --seed 42 \
     --tag nearest \
     --out_dir eval_runs \
     --suite main,stress,full \
     --self_check
   ```

When `--dataset dsA` is set, filenames are dataset-scoped to avoid clobbering outputs when you run multiple datasets
into the same `--out_dir` (for example: `frames_test500_dsA_nearest_seed42.json`).

If you want a *cross-dataset* contract (no `--dataset` filter), make sure your dataset loader can route
each `frames[].dataset` entry to the correct data root.

## 2) Generate Frames Contracts (single JSON or a standard suite)

### 2.0 OPV2V: export to a generic index parquet (one-time)

OPV2V 本身已经有 `index_{split}.parquet`，但 schema 不同。可以先导出成通用 schema：

```bash
cd map-anything
/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python \
  scripts/export_opv2v_index_generic.py \
  --opv2v_index_dir data/opv2v/opv2v_index \
  --splits train,validate,test \
  --out_parquet data/opv2v/opv2v_index/opv2v_index_generic.parquet
```

If you plan to merge datasets later, you can also emit a `dataset` column:
```bash
/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python \
  scripts/export_opv2v_index_generic.py \
  --opv2v_index_dir data/opv2v/opv2v_index \
  --splits train,validate,test \
  --include_dataset --dataset opv2v \
  --out_parquet data/opv2v/opv2v_index/opv2v_index_generic.parquet
```

### 2.1 Report index stats (bucket counts + quantiles + recommended bucket counts)

```bash
PYTHONPATH=$(pwd) python scripts/report_index_bucket_stats.py \
  --index_parquet /path/to/index.parquet \
  --split test \
  --require_assets \
  --preset fine_eval \
  --sample_size 500
```

This prints:
- available counts per bucket,
- baseline distance quantiles,
- a recommended `--bucket_counts` string (proportional allocation, clamped to availability).

Notes:
- `--preset coarse_train` uses the standard 4-bucket train preset `[0,30,60,100,inf]`.
- If you want custom edges, use `--bucket_edges ...` instead of `--preset`.

### 2.2 Single contract (backward compatible)

```bash
PYTHONPATH=$(pwd) python scripts/make_frames_contract_from_index.py \
  --index_parquet /path/to/index.parquet \
  --split test \
  --require_assets \
  --sample_size 500 \
  --seed 42 \
  --out_json eval_runs/frames_test500_seed42.json \
  --self_check
```

### 2.3 Bucketed / stress contract (single JSON)

```bash
PYTHONPATH=$(pwd) python scripts/make_frames_contract_from_index.py \
  --index_parquet /path/to/index.parquet \
  --split test \
  --require_assets \
  --seed 42 \
  --bucket_preset fine_eval \
  --bucket_counts 150,200,100,30,20 \
  --out_json eval_runs/frames_test500_stress_seed42.json \
  --self_check
```

The output contract includes:
- `frames_hash_md5_v1`: legacy hash over `sequence/frame`
- `frames_hash_md5_v2`: fair hash that also includes `main_agent` and ordered `coop_agents`
- `frames_hash_md5_v3`: dataset-aware hash (only meaningful if `dataset` is present in frames)

### 2.4 Standard contract suite (recommended)

Write a **suite** under an output directory: `main` (representative), `stress` (bucket-covered), `full` (all candidates).

Example (OPV2V-style naming: add `--tag nearest`):
```bash
PYTHONPATH=$(pwd) python scripts/make_frames_contract_from_index.py \
  --index_parquet data/opv2v/opv2v_index/opv2v_index_generic.parquet \
  --split test \
  --require_assets \
  --sample_size 500 \
  --seed 42 \
  --tag nearest \
  --out_dir eval_runs \
  --suite main,stress \
  --self_check
```

This writes (paths depend on `--out_dir`):
- `frames_test500_nearest_seed42.json` (main)
- `frames_test500_nearest_stress_seed42.json` (stress)

For promotion / final reporting on OPV2V, prefer the **full test contract** (`full=2170` frames on the nearest index).
If full is too slow, use a smaller **stress** gate (e.g. `--sample_size 500` / `1000`) and keep `full` as the final decision set.

### 2.4.1 Decision roles (avoid pseudo-independent gates)

For OPV2V test contracts, use this hierarchy:
- **Quick**: `test200/500` (main + stress) for fast regression triage.
- **Promotion / final (default)**: `test2170_full` (OPV2V full test is not large; use full for decisions).
- **Optional gate (only if full is too slow)**: `test500_stress` / `test1000_stress` / `test1500_stress` (tail-covered, cheaper).
- `test2000_main` and `test2000_stress` are **near-full proxies** (92% overlap vs full2170), not meaningfully different decision splits.

Concrete OPV2V gate example (already generated in this repo):
- `eval_runs/frames_test1000_nearest_stress_seed42.json` (Test1000-stress; `bucket_target_counts=[278,373,220,39,90]`)

Reason: with the current seed-42 nearest contracts, `test2000_main` is almost a strict subset of `test2170_full`.
Always attach a distinguishability report when defining or changing gates:

```bash
PYTHONPATH=$(pwd) python scripts/report_contract_distinguishability.py \
  --contract_a eval_runs/frames_test2000_nearest_seed42.json \
  --contract_b eval_runs/frames_test2170_nearest_seed42_full.json \
  --label_a test2000_main \
  --label_b test2170_full \
  --out_json eval_runs/contract_distinguish_test2000_main_vs_full2170.json \
  --out_md eval_runs/contract_distinguish_test2000_main_vs_full2170.md
```

For any gate choice, attach a distinguishability report against the final set (typically `full2170`) to avoid pseudo-independent gates.

### 2.5 OPV2V full contracts for all splits (train/validate/test)

`make_frames_contract_from_index.py` supports multi-split suite generation via `--splits ...`:

```bash
PYTHONPATH=$(pwd) python scripts/make_frames_contract_from_index.py \
  --index_parquet data/opv2v/opv2v_index/opv2v_index_generic.parquet \
  --splits train,validate,test \
  --require_assets \
  --seed 42 \
  --tag nearest \
  --out_dir eval_runs \
  --suite full \
  --self_check
```

This writes:
- `frames_train*_nearest_seed42_full.json`
- `frames_validate*_nearest_seed42_full.json`
- `frames_test*_nearest_seed42_full.json`

When `--splits` has multiple entries, the suite report files are written as:
- `eval_runs/contract_suite_report_train.json`
- `eval_runs/contract_suite_report_validate.json`
- `eval_runs/contract_suite_report_test.json`

If `--dataset <id>` is set, report filenames include the dataset id to avoid clobbering:
- `eval_runs/contract_suite_report_<id>_train.json` (etc)

### 2.6 Build a truly different quick/promote set (stress or disjoint)

Use one of these patterns:

1) **Stress quick set (recommended default)**  
   Keep same split/pairing policy, but switch the quick gate to stress sampling (`--suite main,stress`) and finalize on `test2170_full`.

2) **Disjoint quick set (when strict separation is required)**  
   Start from a full contract, then split by baseline buckets (bucket contracts are disjoint by construction):
   ```bash
   PYTHONPATH=$(pwd) python scripts/bucket_frames_by_baseline_distance.py \
     --frames_json eval_runs/frames_test2170_nearest_seed42_full.json \
     --preset fine_eval \
     --out_dir eval_runs/frames_test2170_nearest_seed42_full_buckets \
     --self_check
   ```
   Then use non-overlapping bucket contracts for quick checks (for example, near-only quick vs far/tail promotion diagnostics), and verify overlap with `report_contract_distinguishability.py`.

---

## 3) Using a Contract in OPV2VCoopDataset

`OPV2VCoopDataset` now supports `frames_json=...`:
- dataset scenes are restricted to exactly the contract’s `(sequence, frame)` entries
- per-frame `main_agent` and (when `pair_agents=True`) the main/pair ordering are fixed by the contract

Example (2-agent pairing):
```python
ds = OPV2VCoopDataset(
  ...,
  split="test",
  pair_agents=True,
  frames_json="eval_runs/frames_test50_nearest_seed42.json",
)
```

Hydra config example (训练/评测共用；只需把 `frames_json=...` 拼进 dataset_str)：
```yaml
dataset:
  train_dataset: OPV2VCoopDataset(
    split='${dataset.opv2v.train.split}',
    ROOT='${dataset.opv2v.images_root}',
    depth_root='${dataset.opv2v.depth_root}',
    index_dir='${dataset.opv2v.index_dir}',
    camera_ids=${dataset.opv2v.camera_ids},
    pair_agents=${dataset.coop_pair_agents},
    pair_agent_policy='${dataset.coop_pair_agent_policy}',
    frames_json='eval_runs/frames_train6374_nearest_seed42_full.json',
    resolution=${dataset.opv2v.train.dataset_resolution},
    principal_point_centered=${dataset.principal_point_centered},
    transform='${dataset.opv2v.train.transform}',
    data_norm_type='${model.data_norm_type}',
    aug_crop=${dataset.opv2v.train.aug_crop},
    seed=${dataset.opv2v.train.seed},
    max_num_retries=${dataset.opv2v.train.max_num_retries},
  )
```

---

## 4) Quick Self-Checks

```bash
python -m py_compile \
  scripts/coop_bucket_presets.py \
  scripts/make_frames_contract_from_index.py \
  scripts/report_contract_distinguishability.py \
  scripts/merge_coop_index_parquets.py \
  scripts/report_index_bucket_stats.py \
  scripts/bucket_frames_by_baseline_distance.py \
  scripts/export_opv2v_index_generic.py
```
