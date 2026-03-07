# `map-anything` Local Path Sync Policy — 2026-03-06

## Short Version

The **local path** is named `map-anything`, but the **best remote fit for its current contents is `coopVGGT/mapanything_ft`**.

Reason: after excluding known artifact trees, the local working tree differs from:

- `coopVGGT/mapanything_ft` by `271` local-only candidates and only `3` remote-only files
- `coopVGGT/map-anything` by `289` local-only candidates and `72` remote-only files

So this rescue treats the local path as a **legacy alias path** for the FT branch, not as proof that the correct remote must be `map-anything`.

## Canonical Mapping

### Local path alias

- **Local working tree path:** `vggt_series_4_coop/map-anything`
- **Remote target for this rescue:** `https://github.com/coopVGGT/mapanything_ft.git`
- **Parent repo linkage after rescue:** `.gitmodules` maps path `map-anything` to remote `mapanything_ft`

This preserves the existing local path while making the remote truth explicit.

### Sibling repo kept out of this bulk rescue

- **Broader sibling repo:** `https://github.com/coopVGGT/map-anything.git`

This repo shares ancestry with the local tree, but it is **not** the closest current fit. It should only receive later, deliberately reviewed backports of clearly shared / generic changes.

## What Gets Synced to `mapanything_ft`

The rescue overlay uses an allowlisted source slice with a **no-delete** rule for remote-tracked files that are absent locally.

### Synced top-level files

- `.gitignore`
- `.pre-commit-config.yaml`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `LICENSE`
- `README.md`
- `environment.mapanything_ft.yml`
- `pyproject.toml`
- `setup.py`
- `train.md`

### Synced top-level directories

- `assets/`
- `bash_scripts/`
- `benchmarking/`
- `configs/`
- `data_processing/`
- `depth/`
- `docs/`
- `mapanything/`
- `scripts/`

### Force-tracked canonical eval contracts

Even though `eval_runs/` remains ignored as an artifact tree, the following JSON contracts are treated as tracked exceptions because scripts and docs depend on them:

- top-level `eval_runs/frames_*.json`
- top-level `eval_runs/gt_scale_*.json`
- `eval_runs/frames_test500_dist_buckets_20260221/*.json`

Temp / ad-hoc contracts under `_tmp*`, `shared_frames/`, or old run folders are **not** part of the tracked exception set unless separately reviewed.

## What Stays Local-Only

- `experiments/`
- `outputs/`
- `checkpoints/`
- `feature_cache/`
- `cache/`
- runtime scratch such as `pymp-*`, `torchelastic_*`
- dataset symlinks under `data/opv2v*`
- nested duplicate scratch tree `map-anything/`
- generated HTML / PNG / log artifacts under `eval_runs/**`

## Explicit Exclusions / Cleanup

These are excluded from the rescue overlay or should be removed from staging if present:

- `mapanything.egg-info/`
- `__pycache__/`, `*.pyc`, `*.pyo`
- `summary_smoke/`
- `data/`
- `external_benchmark_data/`
- `condaenv.*.requirements.txt`
- `configs/machine/local_*.yaml`
- nested `map-anything/` scratch copy

## Execution Rules

- author identity must be `MassimoQu <qqxmassimo@gmail.com>`
- GitHub access must use the workspace proxy
- pushes must stay serialized and throttled below the host's `100KB/s` cap
- overlay is **add / modify only** unless a remote deletion is explicitly reviewed
- parent linkage must be repaired only after the FT rescue commit exists remotely
