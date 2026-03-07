# Scale Metric Notes (MapAnything / OPV2V)

Last updated: 2026-02-25

This note defines the only recommended interpretation for scale metrics in this repo.

## 1) Two different spaces: loss-space vs eval-space

1) **Loss-space (optimization)**
- key example: `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale`
- this is a training loss term after transform/criterion/weight
- objective: **toward 0**
- not directly human-interpretable across different configs

2) **Eval-space (ratio-to-GT)**
- model output: `metric_scaling_factor` (prediction `s`)
- OPV2V GT target for scale is **not constant 1**;
  it is per-frame GT normalization factor `g` from
  `normalize_multiple_pointclouds(..., norm_mode="avg_dis", ret_factor=True)`
- objective: `s/g -> 1`

---

## 1.1) What exactly is OPV2V GT scale factor `g` (`norm_mode=avg_dis`)

In this repo, OPV2V “GT scale” is **not** a constant 1. It is the per-frame normalization factor `g`
computed from GT pointmaps:

- code: `mapanything/utils/geometry.py::normalize_multiple_pointclouds(..., norm_mode="avg_dis", ret_factor=True)`
- it gathers all valid GT points (across all views) in **view0 camera frame** and computes:

  `g = mean(||p_view0||)` over valid pixels (invalid pixels are excluded by the mask).

Important consequences:
- `g` is **per-frame** and depends on the **distance distribution** of visible points.
- In coop, `g` is computed jointly over more views (2 agents * 4 cams), so its range is typically wider.

Canonical Test500 contract distribution snapshot (seed42, `avg_dis`):
- source: `map-anything/eval_runs/gt_scale_test500_seed42_avg_dis.json` (generated from `frames_test500_seed42.json`)
- `gt_scale_single` (4-cam rig): mean≈15.65, std≈1.83, min≈11.31, max≈19.32
- `gt_scale_coop` (2-agent joint): mean≈27.45, std≈14.17, min≈12.86, max≈88.35
- ratio `gt_scale_coop/gt_scale_single`: median≈1.50, p90≈2.60, max≈6.42
- tail note: the largest `gt_scale_coop` values are dominated by sequence `2021_08_18_19_48_05`

Interpretation:
- “Scale looks weird (not near 1)” is often just because the correct target is `g` (tens of meters),
  not 1.0. Quality should be judged by `s/g` metrics (`scale_to_gt_*`).
- Coop scale is harder partly because the target range is wider and more long-tailed under the same `avg_dis` contract.

---

## 2) Why old `|s-1|` can be misleading on OPV2V

Legacy fields (`scale_err_mean`, `scale_log_err_mean`, `scale_ratio_mean`) assume GT is 1.
On OPV2V (`norm_mode=avg_dis`) this assumption is wrong, because GT is per-frame `g`.

So values like `scale_err_mean≈7` can be mostly a **reference mismatch artifact** instead of “8x true scale error”.

Concrete example (why `|s-1|` is misleading when GT is `g != 1`):

- assume OPV2V per-frame GT factor `g = 15`
- model predicts `s = 8`
- legacy (wrong reference): `|s-1| = |8-1| = 7`  (looks huge)
- OPV2V-correct ratio: `s/g = 8/15 = 0.533`
  - `scale_to_gt_err = |s/g - 1| = 0.467` (i.e., ~47% under-scale)
  - `scale_to_gt_log_err = |log(s/g)| = 0.629`
  - `scale_to_gt_mult_err = exp(scale_to_gt_log_err) = 1.876` (1 is best; ~1.88x typical multiplicative error)

---

## 3) Primary metrics (must use)

For reporting/ranking/pass-fail, use ratio-to-GT metrics:

- `scale_to_gt_err_mean = mean(|s/g - 1|)`
- `scale_to_gt_log_err_mean = mean_frames( mean_views(|log(s/g)|) )`
- `scale_to_gt_eq_rel_err_mean = mean_frames( expm1(mean_views(|log(s/g)|)) )`
- (gate, tail-sensitive) `scale_to_gt_mult_err_mean = scale_to_gt_eq_rel_err_mean + 1 = mean_frames( exp(mean_views(|log(s/g)|)) )`
  - (optional, tail-robust) geometric-mean view: `scale_to_gt_mult_err_geom = exp(scale_to_gt_log_err_mean)`
- `scale_to_gt_ratio_mean` / `scale_to_gt_ratio_median_mean` / `scale_to_gt_ratio_p90_mean`

Interpretation:
- 0 is best for `*_err_mean`
- 1 is best for `scale_to_gt_ratio_*`
- 1 is best for `scale_to_gt_mult_err_mean` (mean multiplicative error; e.g. 1.3 ~= 30% mean mult error; tail-sensitive)
- `ratio < 1` means under-scale; `ratio > 1` means over-scale

---

## 4) Legacy fields policy

Keep legacy fields only for backward compatibility:

- `scale_err_mean`
- `scale_log_err_mean`
- `scale_eq_rel_err_mean`
- `scale_ratio_mean`

Do **not** use them for cross-run ranking or gate decisions on OPV2V.

---

## 5) Config knobs and comparability

These knobs still change loss magnitude and training dynamics:

- `scale_loss_weight`
- `direct_scale_loss`
- `loss_in_log`
- `norm_mode`

Therefore:
- Compare optimization behavior with loss-space metrics **within a run/config family**.
- Compare model quality across runs with **`scale_to_gt_*`**.

---

## 6) Recommended report block

Every scale report should include both:

1) loss-space (for optimization diagnostics)
- `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale`

2) eval-space (for quality/ranking)
- `scale_to_gt_err_mean`
- `scale_to_gt_log_err_mean`
- `scale_to_gt_eq_rel_err_mean`
- `scale_to_gt_mult_err_mean`
- `scale_to_gt_ratio_mean`

This avoids mixing “toward 0” (loss) with “toward 1” (ratio).

## 7) Gate policy (recommended)

- Use `scale_to_gt_err_mean` (and/or `scale_to_gt_log_err_mean`) for gating on OPV2V.
- Keep legacy `scale_err_mean` only as a compatibility field (do not gate/rank on it for OPV2V).

---

## 8) `detach_metric_scaling_factor_from_geom` (what “detach geom-scale” means)

This flag is a **stop-gradient switch** on *the denominator* used when converting metric geometry into an
up-to-scale representation for geometry losses:

- `scale_geom = scale.detach() if detach_metric_scaling_factor_from_geom else scale`
- `geom_raw = geom_metric / scale_geom`

Important subtlety:
- Whether this flag **reduces** or **increases** gradients into `metric_scaling_factor` depends on how the
  model constructs `geom_metric`.

Two common cases:

1) If the model outputs **unscaled** geometry (`geom_metric` does *not* contain `scale`):
- flag OFF: geometry losses backprop into `scale` via the `1/scale` division.
- flag ON: denominator is `stopgrad(scale)` so geometry losses do **not** backprop into `scale`.

2) If the model outputs **scaled** geometry (`geom_metric = geom_base * scale`):
- flag OFF: `(geom_base * scale) / scale = geom_base` (geometry becomes scale-invariant; gradient to `scale`
  cancels out).
- flag ON: `(geom_base * scale) / stopgrad(scale)` breaks the cancellation, so geometry losses can backprop
  into `scale`.

In this repo, both VGGT and MapAnything are typically in case (2) (they multiply depth/points/pose-translation
by `metric_scaling_factor` before returning predictions), so interpret this knob as:
**“Should dense geometry losses be allowed to also push the scale head?”**
not as a universally “detach everything from scale”.

Code reference (current implementation):
- `mapanything/train/losses.py` in `FactoredGeometryScaleRegr3DPlusNormalGMLoss.get_all_info()`:
  - uses `scale_geom = scale.detach() if self.detach_metric_scaling_factor_from_geom else scale`
  - then divides points/depth/pose_trans by `scale_geom`

Config reference (example):
- `configs/loss/opv2v_vggt_pose_scale_loss_recover_direct_detach_geom_scale_w0p4_pose12_vr1p_noconf.yaml`
  sets `detach_metric_scaling_factor_from_geom=True`.

Practical note:
- This knob does **not** “fix scale by itself”; it only changes gradient routing.
  If scale supervision is sparse/noisy (OPV2V often is), you may still need:
  - stronger/more reliable scale supervision, or
  - a dedicated “scale recover” stage after a “pose push” stage.
