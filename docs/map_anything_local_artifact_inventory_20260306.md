# `map-anything` Local Artifact Inventory — 2026-03-06

This inventory records the major local-only trees under `vggt_series_4_coop/map-anything` and clarifies which small exceptions are intentionally force-tracked.

| Path | Current signal | Policy |
|------|----------------|--------|
| `eval_runs/` | `427` top-level dirs, `305` `summary_test.json` files | Keep run outputs local; force-track only reviewed contract JSONs |
| `experiments/` | `40` top-level dirs, `206` dirs under `experiments/local_runs/` | Local experiment workspaces only |
| `outputs/` | `4` top-level dirs | Generated exports / viz only |
| `checkpoints/` | `3` top-level entries | Binary model artifacts stay out of Git |
| `feature_cache/` | `2` top-level entries | Derived cache only |
| `cache/`, `pymp-*`, `torchelastic_*` | runtime scratch | Ignore entirely |
| `data/opv2v*` | symlinks into `/J6P-perception/yijinxiong_workspace/1090539994_仅Bear0217/...` | Keep local only; never commit the symlink targets |
| `map-anything/` (nested) | duplicate scratch subtree with local temp contracts / runs | Keep local only; exclude from rescue overlay |

## Canonical Exceptions Inside `eval_runs/`

These file families are treated as **tracked exceptions** even though the rest of `eval_runs/` stays local-only:

- top-level `frames_*.json`
- top-level `gt_scale_*.json`
- `frames_test500_dist_buckets_20260221/*.json`

They are kept because benchmark scripts and docs refer to them as canonical contracts.

## Reviewer Heuristic

- **source / config / doc / reusable script / canonical contract JSON** → track it
- **checkpoint / cache / generated eval / mutable run dir / dataset symlink** → keep it local and describe it instead
