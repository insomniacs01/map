# Goal-Based Test Plan (Review-Fix Loop)

Last updated: 2026-02-27

This document defines goal-linked acceptance criteria, reproducibility rules, and current execution status.

## 2026-02-27 Updates

- PredPose det-head (deploy-like) fixalign 后重训（e10 v1）+ composed eval evidence (Test50):
  - det head ckpt:
    - `map-anything/experiments/local_runs/20260227_det_headonly_deploy_predpose_aligngt0_fixalign_from_promotedv2_e10_v1/checkpoint-final.pth`
  - MapAnything / sfix_direct (note: AP higher but geometry quality is much worse; suggests compose eval is sensitive to backbone/feature distribution, not only geometry metrics):
    - `map-anything/eval_runs/det_compose_predpose_aligngt0_test50_20260227j_sfix_direct_dethead_fixalign_e10_seeded/summary_test.json`
    - single `det_ap_iou=1.69e-02`, coop `det_ap_iou=2.56e-03`
  - MapAnything / promoted_v2:
    - `map-anything/eval_runs/det_compose_predpose_aligngt0_test50_20260227g_promoted_v2_dethead_fixalign_e10_seeded_md5fix/summary_test.json`
    - single `det_ap_iou=8.53e-03`, coop `det_ap_iou=5.48e-04`
  - VGGT / e30_best:
    - `map-anything/eval_runs/det_compose_predpose_aligngt0_test50_20260227i_vggt_e30_best_dethead_fixalign_e10_seeded/summary_test.json`
    - single `det_ap_iou=1.36e-02`, coop `det_ap_iou=4.85e-04`
- PredPose det alignment bug fix (critical):
  - `map-anything/mapanything/models/mapanything/det_model.py` now constructs predicted poses from `cam_quats/cam_trans` during `forward()` so `pred_pose_align_to_view0_gt` actually applies.
  - Evidence: `map-anything/eval_runs/det_compose_predpose_aligngt0_test50_20260227_fixalign/summary_test.json` (single AP jumps to `~1.54e-03`).
- Det fairness audit now reports two dimensions:
  - `protocol_fair_*` vs `strict_same_pipeline_*` (allow cross-model vs require same `model_arch`)
  - `*_core` vs `*_compose` (ignore vs require `det_head_ckpt`)
  - Script: `map-anything/scripts/audit_det_eval_summaries.py`
- Det BEV PNG gallery is now re-runnable for old eval runs:
  - `map-anything/scripts/make_det_bev_png_gallery.py` -> `<eval_root>/bev_png/index.html`
- Det eval determinism:
  - `batch_eval.py` now seeds RNG from `--seed` (see Locked T6 baseline update below; evidence: `_det_seedtest5_run1` vs `_det_seedtest5_run2`).
- Delta status doc: `docs/lines_status_review_20260227.md`

## Goals & Success Criteria

1) Stable geometry training (no NaN loops) with reproducible checkpoints.
2) Automated gate to switch from geometry -> det/e2e evaluation.
3) Detection AP exceeds locked baselines under a consistent eval protocol.

### Current user-facing objective (authoritative)

- Geometry first, **coop-first**, under one fixed fair contract.
- Deployment constraints:
  - intrinsics known
  - intra-agent 4-camera rig extrinsics known (relative)
  - **cross-agent (vehicle/world) pose unknown** -> must be predicted
- Primary (deployment) protocol: calibrated intrinsics are provided, but **no cross-agent GT poses are provided as model inputs**.
  - We optimize **coop depth/scale + cross-agent relative pose**.
- Scale pass condition (OPV2V-correct, coop-first):
  - `metrics.<model>.coop.scale_to_gt_mult_err_mean <= 1.3` (1 is best; ~<=30% typical multiplicative error).
  - Also report `metrics.<model>.coop.scale_to_gt_ratio_mean` (1 is best; <1 under-scale, >1 over-scale).
- Depth guardrail (coop-first):
  - no regression >10% vs a locked baseline checkpoint on `metrics.<model>.coop.depth_rel_mean`.
- Cross-agent pose guardrail (coop-first):
  - no regression >10% vs baseline on `metrics.<model>.coop.cross_agent_pose_trans_mean` and `metrics.<model>.coop.cross_agent_pose_rot_mean`.
  - Note: this requires backfilling baseline numbers under the same eval protocol.
- Only after geometry passes, re-open det/e2e and target T6 AP improvements.

### Baseline Policy (must use this policy for pass/fail)

- Locked baseline for T6 (FairLock; canonical Test50 contract + explicit det decode cfg):
  - Canonical det contract: `map-anything/eval_runs/frames_test50_det_e2e_v5.json` (50 frames, hash=`1345e984e7f7c1dae7795c972d70db37`)
    - **Pairing note (important):** this legacy contract was generated from `scripts/batch_eval.py::discover_frames()`'s
      *fallback* heuristic (**agent-id first two**, a.k.a. `pair_policy=id_first2`), **not** from the OPV2V index's
      nearest-neighbor pairing (`pair_agent_policy=nearest`).
      - Empirically, on this contract **38% (19/50)** frames use a *different* coop agent than the index's nearest pair,
        and those mismatch frames have **~3× larger baseline distance** on average (ratio mean≈3.00; see “Test-set design review” below).
      - This makes Test50 a **stress / hard-pair** contract. It is OK for *fixed* comparisons, but it is **not representative**
        of the default training/eval pairing used by `OPV2VCoopDataset(pair_agents=True, pair_agent_policy='nearest')`.
  - Locked baseline summary (FairLock re-run; deterministic seed; includes `meta.det_decode_cfg` / `meta.eval_script_md5`):
    - `map-anything/eval_runs/det_fairlock_test50_20260227h_e2e_v5_170846_final_posed_seeded_evalmd5fix/summary_test.json`
  - Locked baseline metrics (same contract, same eval script, same det decode cfg):
    - single: `metrics.det_model.single.det_ap_iou = 0.19073152129275572`
    - coop: `metrics.det_model.coop.det_ap_iou = 0.1336364940393163`
  - Determinism note (2026-02-27):
    - `batch_eval.py` now seeds RNG from `--seed` to make det AP reproducible even when the BEV rasterizer randomly subsamples points (`max_points`).
    - Older “FairLock” summaries generated before this change should be treated as **legacy** for strict pass/fail.

- Historical reference baselines (legacy summaries; keep for context only, do not use for strict pass/fail):
  - `metrics.e2e_v5.single.det_ap_iou = 0.19915220884189924`
    - Source: `map-anything/eval_runs/det_e2e_v5_eval_t005/summary_test.json`
  - `metrics.e2e_v5.coop.det_ap_iou = 0.1380228069656401`
    - Source: `map-anything/eval_runs/det_e2e_v5_coop_eval_t005/summary_test.json`
- Baseline metadata backfill (2026-02-14):
  - Both locked baseline summaries now contain `run_info.det_decode_cfg` (`iou_thresh=0.5`, `score_thresh=0.05`) for auditability.
  - Added ratio-space scale diagnostics (`scale_log_err_mean`, `scale_eq_rel_err_mean`, `scale_ratio_mean`) via artifact backfill script.
- Historical reference best (not the locked baseline):
  - `metrics.stage2_det_v5_2k.single.det_ap_iou = 0.25`
  - Source: `map-anything/eval_runs/stage2_det_v5_2k_eval_000386/summary_test.json`

## Test-Set Design Review (OPV2V size / buckets / pairing fairness)

This section answers: “det Test50 是否够？” “训练有分桶，测试有没有分桶？” and “OPV2V 体量下合理的测试集合怎么定？”.

### Ground-truth population (OPV2V test split, *index nearest-pair*)

Authoritative source: `map-anything/data/opv2v/opv2v_index/index_test.parquet` (built by `scripts/build_opv2v_index.py`).

- Test rows (main/pair pairs): **2170**
- Baseline distance (main vs nearest pair, XY meters):
  - mean=**28.63** p50=**19.46** p95=**80.47** p99=**159.57** max=**166.30**
- Bucket counts (edges `[0,15,30,60,100,inf)`):
  - B0 [0,15): **651**
  - B1 [15,30): **873**
  - B2 [30,60): **517**
  - B3 [60,100): **39** (rare bucket)
  - B4 [100,inf): **90** (tail bucket)

Implications:
- A *random* 50-frame sample from the nearest-pair population would typically contain only ~**1** frame in B3 and ~**2** frames in B4.
- If you care about “geometry completion vs coop perception” **by bucket**, Test50 is **not enough** (tails are under-sampled).

### Current contracts are “hard-pair” (agent-id first-two), not nearest-pair

Both canonical contracts were created from the legacy `discover_frames()` heuristic (first two agent IDs):
- Det Test50: `map-anything/eval_runs/frames_test50_det_e2e_v5.json`
  - baseline distance stats: mean=**40.03**, p95=**120.52**, max=**132.56**
  - bucket counts: B0=5, B1=25, B2=13, B3=2, B4=5
  - pairing mismatch vs index-nearest: **19/50** frames
    - mismatch baseline ratio (contract / nearest): mean≈**3.00**, p50≈**1.76**, p95≈**7.25**, max≈**12.18**
- Geom Test500: `map-anything/eval_runs/frames_test500_seed42.json`
  - baseline distance stats: mean=**36.04**, p95=**103.38**, p99=**161.59**, max=**166.13**
  - bucket counts: B0=100, B1=167, B2=186, B3=18, B4=29
  - pairing mismatch vs index-nearest: **167/500** frames
    - mismatch baseline ratio (contract / nearest): mean≈**3.29**, p50≈**2.37**, p95≈**10.42**, max≈**13.85**

Interpretation:
- These contracts are valid **fixed stress tests**, but they should not be used as the *only* evidence for
  “OPV2V overall performance” unless you explicitly want the “hard-pair / long-baseline” regime.

### Bucketed testing: do we have it?

- Geometry: yes (contract bucketing already exists)
  - `map-anything/scripts/bucket_frames_by_baseline_distance.py`
  - Example: `map-anything/eval_runs/frames_test500_dist_buckets_20260221/bucket_report.md`
  - No-inference per-bucket aggregation:
    - `map-anything/scripts/report_geom_by_baseline_buckets.py`
- Detection: historically **no**, because AP was only computed on the full contract.
  - New (2026-02-27): `scripts/batch_eval.py --save_det_cache` now writes `det_ap_cache_{single,coop}.npz`
    so we can re-score AP on arbitrary frame subsets (buckets / sample-size sensitivity) without re-running inference.
  - Analysis tool: `map-anything/scripts/report_det_ap_subset_sensitivity.py`
  - Bucket tool: `map-anything/scripts/report_det_ap_by_baseline_buckets.py`
  - Evidence (Test500 posed baseline; no-inference reports):
    - `docs/det_ap_subset_sensitivity_test500_posed_baseline_single_20260227.md`
    - `docs/det_ap_subset_sensitivity_test500_posed_baseline_coop_20260227.md`

### Recommended “reasonable” test sets (given OPV2V size + bucket strategy)

Use *two layers* to balance iteration speed vs fairness/stability:

1) **Quick smoke (debug only)**:
   - det: 50-frame fixed contract (keep existing Test50 for continuity), but do not over-interpret tail behavior.
2) **Stable promotion / paper numbers**:
   - det/e2e: **>=200–500** frames, and report AP **by baseline bucket** + overall.
   - geometry: **>=500** frames + per-bucket report; additionally run full B3/B4 buckets periodically because they are rare (39 / 90 frames).

Evidence for “50 is not enough” (AP sampling variance):
- On `frames_test500_seed42.json` (500 frames), posed baseline det AP(full) is:
  - single: **0.205546**
  - coop: **0.143997**
- If you only evaluate on a *random* 50-frame subset (Monte Carlo; no inference), AP typically varies by ~**±0.03**:
  - single: std≈**0.0186**, p05≈**0.1750**, p95≈**0.2379**
  - coop: std≈**0.0179**, p05≈**0.1176**, p95≈**0.1770**
- At 200 frames, variance is much smaller (~**±0.012**):
  - single: std≈**0.0073**, p05≈**0.1937**, p95≈**0.2171**
  - coop: std≈**0.0076**, p05≈**0.1320**, p95≈**0.1565**

Also strongly recommended:
- Add a *nearest-pair* contract for representativeness (generated from index):
  - `map-anything/scripts/make_opv2v_frames_contract.py --pair_policy index_nearest`
  - Example artifact: `map-anything/eval_runs/frames_test500_nearest_seed42.json`
  - Smaller/cheaper option (often enough for promotion gates): `map-anything/eval_runs/frames_test200_nearest_seed42.json`
  - Note: random 500-frame samples may still under-sample B3; either oversample B3/B4 or evaluate those buckets separately.
  - Optional “nearest but stress” contract (explicit bucket targets to avoid B3 under-sampling):
    - `map-anything/eval_runs/frames_test500_nearest_stress_seed42.json` (B0=150,B1=200,B2=103,B3=18,B4=29)
    - `map-anything/eval_runs/frames_test200_nearest_stress_seed42.json` (B0=60,B1=70,B2=40,B3=10,B4=20)

### Detection Metric Policy (what “AP” means here)

This repo’s detection metric key is `det_ap_iou`:

- Definition: **Average Precision** (area under the precision-recall curve) for BEV detection under a
  **single IoU threshold** (`iou_thresh`, typically 0.5).
- It is **not COCO-style mAP** (no averaging across multiple IoU thresholds).
- Implementation: `map-anything/scripts/batch_eval.py::compute_ap_bev` (greedy per-frame matching with
  oriented BEV IoU, then precision-envelope integration).
- Comparability requirements:
  - `det_decode_cfg.iou_thresh` and `det_decode_cfg.score_thresh` must be fixed for comparisons.
    (Decode filters predictions before AP; AP then sweeps over remaining predictions by score.)
  - Use the same frame contract (usually 50-frame sample for det/e2e; see T6).

### Scale Metric Policy (must use this policy for interpretation)

- Training key `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` is a **loss-space** metric.
  - It is computed after log-space transform + robust criterion + `scale_loss_weight`.
  - Correct optimization direction: **lower -> better -> toward 0**.
  - It is **not** a direct metric scale ratio and should not be interpreted as "near 1 is good".
- On OPV2V with `norm_mode=avg_dis`, scale GT is per-frame normalization factor `g` (not constant 1).
- Human-readable scale quality must therefore use **ratio-to-GT** eval keys:
  - `metrics.<model>.<mode>.scale_to_gt_err_mean = mean(|s/g - 1|)`
  - `metrics.<model>.<mode>.scale_to_gt_log_err_mean = mean(|log(s/g)|)`
  - `metrics.<model>.<mode>.scale_to_gt_eq_rel_err_mean = exp(scale_to_gt_log_err_mean) - 1`
  - (more human-friendly) `metrics.<model>.<mode>.scale_to_gt_mult_err_mean = exp(scale_to_gt_log_err_mean)`
  - `metrics.<model>.<mode>.scale_to_gt_ratio_mean = mean(s/g)`
- Interpretation:
  - `*_err_mean`: **0 is best**
  - `scale_to_gt_ratio_mean`: **1 is best** (`<1` under-scale, `>1` over-scale)
  - `scale_to_gt_mult_err_mean`: **1 is best** (typical multiplicative error; e.g. 1.3 ~= 30% scale error)
- Legacy fields (`scale_err_mean`, `scale_log_err_mean`, `scale_ratio_mean`) are retained only for backward compatibility and must not be used for ranking/pass-fail.
- Cross-run comparison rule:
  - If any of `scale_loss_weight` / `direct_scale_loss` / `loss_in_log` / `norm_mode` changed,
    compare runs by `scale_to_gt_*` first, then pose/depth guardrails.

## Reproducibility Requirements (apply to all tests)

Record the following for every test status update:
- timestamp (`YYYY-MM-DD HH:MM +0800`)
- command (or script path + key args)
- checkpoint path
- summary/log evidence path
- split/sample size/seed
- pass/fail decision and reason

For fairness snapshots across historical eval runs (read-only, no inference):
- same-frame scoreboard:
  - `python map-anything/scripts/scoreboard_snapshot.py`
- pinhole test50 top-k scan:
  - `python map-anything/scripts/scoreboard_scan_test50_pinhole.py --topk 10`
- fixed-contract geometry scan (test500 or other explicit frame contract):
  - `python map-anything/scripts/scoreboard_scan_geom_contract.py --canonical_frames_json map-anything/eval_runs/frames_test500_seed42.json --run_prefix geom_generalize_test500 --model geom_model --topk 20`

### Fairness Contract Rule (mandatory)

- Any cross-run conclusion must include:
  - frame contract source (`frames_json` or canonical summary path),
  - frame count,
  - frame hash:
    - prefer `meta.frames_hash_md5_v2` (hash includes `sequence/frame` **and** `main_agent/coop_agents` order)
    - legacy `meta.frames_hash_md5` only hashes `sequence/frame` (insufficient when pair policy differs).
- If frame contracts differ, treat the comparison as exploratory only (not pass/fail evidence).

## Preflight Checks (run before T3/T5/T6)

- Disk/TMP:
  - `df -h /tmp` and `df -h .`
  - If `/tmp` is near full, set `TMPDIR` and `TORCHELASTIC_TMPDIR` to workspace path.
  - Keep `TMPDIR` a **short path** (e.g. `/tmp/ma_tmp`) to avoid Python multiprocessing
    `OSError: AF_UNIX path too long` on some hosts.
- Env:
  - activate `mapanything-cu121` and set `PYTHONPATH`.
- Determinism knobs for eval:
  - fix sample size and seed (`50`, `42`) unless explicitly testing sensitivity.
- Scale tracking hygiene:
  - when recording T4/T5/T6, always include `scale_to_gt_*` diagnostics.

## Test Standards (Acceptance Criteria)

### T1: OPV2V index usage (data pipeline)
Purpose: Ensure dataset loads via global index (no YAML scan fallback in normal path).
- Command (smoke):
```bash
source /J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/activate
PYTHONPATH=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything python - <<'PY'
from mapanything.datasets.opv2v import OPV2VCoopDataset
root = '/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/data/opv2v_images'
depth = '/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/data/opv2v_depth'
dset = OPV2VCoopDataset(
    ROOT=root, depth_root=depth, split='train',
    camera_ids=(0,1,2,3), pair_agents=True, min_agents=2,
    min_num_views=8, max_num_views=8, max_scenes=2, num_views=8,
    variable_num_views=False, resolution=(448, 252),
    principal_point_centered=False, transform='imgnorm',
    data_norm_type='dinov2', aug_crop=0, seed=1, max_num_retries=1,
)
print('Scenes:', len(dset.scenes))
PY
```
- Pass if:
  - Log includes `OPV2V index hit: ...index_<split>.parquet`.
  - `OPV2V index filtered scenes > 0` for typical coop config.

### T2: Index contents sanity
Purpose: Ensure index artifacts exist and are non-empty.
- Command:
```bash
source /J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/activate
python - <<'PY'
import pandas as pd
from pathlib import Path
base = Path('/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/data/opv2v_images/opv2v_index')
for split in ['train', 'validate', 'test']:
    print(split, len(pd.read_parquet(base / f'index_{split}.parquet')))
PY
```
- Pass if:
  - `index_{train,validate,test}.parquet`, `manifest.json`, `schema.json` all exist.
  - Row counts are all `> 0`.

### T3: Geometry stability smoke (no NaN loops)
Purpose: Ensure geometry training can complete and produce usable artifacts.
- Command:
  - `bash bash_scripts/run_train_opv2v_geom.sh ...` (1 epoch smoke on current geometry config).
- Pass if all conditions hold:
  - training exits normally and summary can be generated.
  - `checkpoint-last.pth` or `checkpoint-best.pth` or `checkpoint-final.pth` exists.
  - gate summary metrics for `validate` contain finite numeric values for:
    - `depth_z_mae_m`, `depth_z_rmse_m`, `pose_trans_l2_m`, `pose_rot_deg`, `scale_to_gt_err_mean` (or legacy `scale_err_mean` if unavailable).
    - (legacy) `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` remains useful for optimization tracking.
  - badloss dumps are bounded by configured cap (if triggered).
- Scale note:
  - `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` here means **scale loss** (not scale ratio).
- Note:
  - Do not use naive `grep nan|inf` over full log for pass/fail; config dumps contain literal `inf` strings.

### T4: Gate check (geometry readiness)
Purpose: Ensure gate thresholds evaluate correctly on generated summary (loss-space gate).
- Command:
  - `python bash_scripts/monitor/summarize_run.py --log <train.log> --out_dir <summary_dir>`
  - `source experiments/long_runs/auto_queue_pinhole_pose_depth_scale_v3.txt.gate.env`
  - `python bash_scripts/monitor/gate_metrics.py --summary <summary_dir>/summary.json`
- Pass if:
  - gate script exits with PASS for the intended split.
- Required reporting (for interpretability, not gate pass/fail):
  - from latest eval summary (`summary_test.json`), record `scale_to_gt_err_mean` for each mode.
  - also record `scale_to_gt_log_err_mean` and `scale_to_gt_eq_rel_err_mean` when available.
  - if `scale_to_gt_*` fields are missing/non-finite, mark run status as "scale interpretability incomplete".

### T5: Auto-eval readiness (det/e2e)
Purpose: Verify eval artifacts are generated, and distinguish script smoke from watchdog e2e.
- T5a manual smoke command:
  - `EVAL_MODES="single coop" bash bash_scripts/utils/opv2v_det_eval_and_html.sh <ckpt> bev_centernet_wide_v4 eval_runs/auto_queue_det_eval_smoke 5 42`
- T5a pass if:
  - `eval_runs/auto_queue_det_eval_smoke/summary_test.json` exists.
  - `eval_runs/auto_queue_det_eval_smoke/html/index.html` exists.
- T5b watchdog e2e pass if:
  - one watchdog cycle triggers auto-eval after gate pass and writes expected output dir.

### T6: Detection AP improvement
Purpose: Confirm candidate checkpoint beats locked baselines.
- Command:
  - `bash map-anything/scripts/eval_det_ckpt_test50.sh <ckpt> <det_head_cfg> map-anything/eval_runs/<run> 0`
- Candidate metrics path (current auto-eval format):
  - `metrics.det_model.single.det_ap_iou`
  - `metrics.det_model.coop.det_ap_iou`
- Pass if either improvement rule holds for each mode:
  - additive: `>= baseline + 0.02`, or
  - relative: `>= 1.10 * baseline`.
- Baselines for this comparison are the locked values in "Baseline Policy".

Fairness requirements (mandatory for any T6 conclusion):
- Candidate and baseline must share the same det contract + protocol:
  - frames hash equals `1345e984e7f7c1dae7795c972d70db37` (see `meta.frames_hash_md5`)
  - `meta.seed` matches (det eval now seeds RNG for deterministic AP; do not mix seeds)
  - `meta.det_decode_cfg` matches (score/NMS/max_dets/IoU/BEV grid)
  - `meta.det_head_cfg` matches
  - `meta.keep_camera_poses` and `meta.model_task` match
- Use the audit script as evidence:
  - `python map-anything/scripts/audit_det_eval_summaries.py <baseline_summary> <candidate_summary>`

### T7: Geometry generalization on large fixed contract
Purpose: Verify “good-looking” geometry checkpoints under a larger, fixed frame set (avoid toy/sampling bias).
- Canonical contract:
  - `map-anything/eval_runs/frames_test500_seed42.json` (500 frames, seed=42)
- Command template (deployment protocol; coop-first; calibrated intrinsics; no cross-agent GT pose inputs):
```bash
PYTHONPATH=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything \
CUDA_VISIBLE_DEVICES=<gpu> \
/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python \
map-anything/scripts/batch_eval.py \
  --split test \
  --frames_json map-anything/eval_runs/frames_test500_seed42.json \
  --modes coop \
  --model_task calibrated_sfm \
  --models geom_model=<ckpt> \
  --model_filter geom_model \
  --pc_metrics \
  --output_root map-anything/eval_runs/<run_name>
```
- Pass if all conditions hold:
  - `summary_test.json` exists and includes `coop`.
  - Contract matches canonical (`frames` length/hash equal).
  - Compared checkpoints are ranked on ratio-to-GT scale (`scale_to_gt_*`) with depth/pc as guardrails, plus cross-agent pose as a guardrail.

### T8: Scale metric vs visualization consistency audit
Purpose: Validate that historical “good-looking” ckpts are checked with both corrected scale metric and representative visual evidence under the same contract.
- Inputs:
  - fixed contract eval summary (`summary_test.json`) from T7
  - GT scale contract JSON (`gt_scale_test500_seed42_avg_dis.json`)
- Command template (per ckpt):
```bash
PYTHONPATH=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything \
CUDA_VISIBLE_DEVICES=<gpu> \
/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python \
map-anything/scripts/scale_viz_audit_from_csv.py \
  --eval_run_dir map-anything/eval_runs/<geom_generalize_run> \
  --model geom_model \
  --ckpt <ckpt_path> \
  --contract_json map-anything/eval_runs/gt_scale_test500_seed42_avg_dis.json \
  --out_dir map-anything/eval_runs/scale_viz_audit_test500_<tag>/<ckpt_tag> \
  --split test \
  --modes single coop

PYTHONPATH=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything \
/J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/python \
map-anything/scripts/make_batch_eval_pcd_html.py \
  --eval_root map-anything/eval_runs/scale_viz_audit_test500_<tag>/<ckpt_tag> \
  --summary_json map-anything/eval_runs/<geom_generalize_run>/summary_test.json \
  --draw_gt_boxes \
  --out_dir map-anything/eval_runs/scale_viz_audit_test500_<tag>/<ckpt_tag>/html_scale
```
- Pass if:
  - for each ckpt, `selected_scale_representatives.json` exists and both modes have finite best/median/worst `scale_to_gt_err`;
  - corresponding HTML files exist (`html_scale/geom_model/{single,coop}/*_{best,median,worst}_scale.html`);
  - report summarizes both metric table and visualization links under the same contract.

## Execution Status (current snapshot)

- [x] Geometry deployment-protocol promotion on fixed Test500 contract -> PASS (2026-02-24 05:10 +0800)
  - Evidence (deployment protocol; coop-first; `model_task=calibrated_sfm`, keep_camera_poses=0):
    - baseline: `eval_runs/geom_generalize_test500_20260223_c20260210_sfix_direct_calibrated_coop/summary_test.json`
      - coop: `mult=2.3243`, `ratio=0.4865`, `cross_trans=11.3128m`, `depth_rel=0.3322`
    - promoted (FairLock recheck): `eval_runs/geom_fairlock_test500_20260226_nm4_base/summary_test.json`
      - coop: `mult=1.2693` (<=1.3 ✅), `ratio=0.9493`, `cross_trans=10.5028m`, `depth_rel=0.3468`

- [x] L1b task ablation (train task images_only vs calibrated_sfm; eval protocol fixed) -> PASS (2026-02-24 05:10 +0800)
  - Evidence:
    - images_only: `eval_runs/geom_ablation_test500_20260223_l1_task_imagesonly_from_nm4_w02_d0_60_lr5e6_e1_r1/quick_report.txt`
    - calibrated_sfm: `eval_runs/geom_ablation_test500_20260223_l1_task_calibrated_sfm_from_nm4_w02_d0_60_lr5e6_e1_r1/quick_report.txt`
  - Result:
    - both keep `mult<=1.3`, but calibrated_sfm does not improve cross-agent pose; continue training defaults to images_only.

- [x] L1c pose sweep (detach geom-scale + stronger cross-agent pose supervision) -> FAIL (2026-02-24 06:12 +0800)
  - Evidence:
    - rpose1: `eval_runs/geom_ablation_test500_20260223_l1_detach_w0p4_pose12_rpose1_from_nm4_e1/quick_report.txt`
    - rpose2: `eval_runs/geom_ablation_test500_20260223_l1_detach_w0p4_pose12_rpose2_from_nm4_e1/quick_report.txt`
  - Result:
    - cross-agent pose does not improve (`cross_trans` regresses) and rpose2 violates `mult<=1.3`; not promotable.

- [x] L2 supervision densify (train-only sparse depth nearest fill; radius sweep) -> FAIL (2026-02-24 23:00 +0000)
  - Evidence (Test500 / coop-only / deployment protocol: `model_task=calibrated_sfm`, keep_camera_poses=0):
    - r=1: `eval_runs/geom_ablation_test500_20260223_l2_densify_r1_from_nm4_w02_e1/quick_report.txt`
      - coop: `mult=1.3195` (fail), `ratio=0.8810`, `cross_trans=10.8479m`, `depth_rel=0.3379`
    - r=2: `eval_runs/geom_ablation_test500_20260223_l2_densify_r2_from_nm4_w02_e1/quick_report.txt`
      - coop: `mult=1.3332` (fail), `ratio=0.8633`, `cross_trans=10.7139m`, `depth_rel=0.3376`
  - Result:
    - Nearest-fill densify v1 increased supervision density but regressed scale (`mult>1.3`) and did not improve cross-agent pose; do not promote.
  - Follow-up confirmation:
    - `eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_l2_densify_r2/quick_report.txt`
      - coop: `mult=1.3046` (fail), `ratio=0.8921`, `cross_trans=10.6268m`, `depth_rel=0.3464`
      - Note: r=2 remains fragile even when the rest of the pipeline succeeded; keep densify v1 off the mainline.

- [ ] Plan1 distance buckets (coop-only; fixed Test500 contract) -> DIAGNOSTIC (2026-02-24 23:00 +0000)
  - Context:
    - Purpose is to diagnose which inter-agent distance regime helps cross-agent pose without sacrificing scale/depth.
    - All results below share the same contract hash (`frames_hash_md5=4affdce1...`) and baseline (`nm4_scaleonly_w02`).
  - Evidence (key runs):
    - near 0–30m: `eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_plan1_near_0_30m/quick_report.txt`
      - coop: `mult=1.2889` (pass), `cross_trans=10.7250m`, `depth_rel=0.3425`
    - mid 30–60m (rerun on valid GPUs): `eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_plan1_mid_30_60m/quick_report.txt`
      - coop: `cross_trans=6.1614m` (large improvement) but `mult=1.3242` (fail)
    - mid 30–60m + detach geom-scale (pose-push follow-up): `eval_runs/geom_ablation_test500_20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1/quick_report.txt`
      - coop: `cross_trans=5.9934m`, `cross_rot=1.433°`, `depth_rel=0.3270` (improves), but `mult=1.3028` (just above 1.3; fail)
    - far 60–100m: `eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_plan1_far_60_100m/quick_report.txt`
      - coop: `cross_trans=9.6473m` (improves) but `depth_rel=0.3985` (fails depth guardrail)
    - tail 100–inf (rerun): `eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_plan1_tail_100_inf/quick_report.txt`
      - coop: `mult=1.4535`, `cross_trans=13.6472m`, `depth_rel=0.5242` (all fail)
    - detach follow-up: `eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_l1_detach_geom_scale_w0p4_pose12/quick_report.txt`
      - coop: `mult=1.2709` (pass), `cross_trans=10.4001m`, `depth_rel=0.3211`
    - detach_rpose2 (rerun): `eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_l1_detach_geom_scale_w0p4_pose12_rpose2__rerun_gpu0123/quick_report.txt`
      - coop: `mult=1.2788` (pass), `cross_trans=10.3329m`, `depth_rel=0.3213`
  - Infra note:
    - The initial pipeline lane1 jobs failed with `CUDA error: invalid device ordinal` because the host only exposed GPUs `0..5` but lane1 was launched with `CUDA_VISIBLE_DEVICES=4,5,6,7`.
      - Evidence: `experiments/long_runs/20260224_193935_plan1_pipeline_8x4090__pipeline/steps/lane1__plan1_mid_30_60m__...log`
    - A targeted rerun on `GPU_LIST=4,5` succeeded for mid/tail: `experiments/long_runs/20260224_210949_lane1_gpu45__rerun_lane1_gpu45/runner.log`.

- [x] T1 index usage -> PASS (2026-02-08 04:37 +0800)
  - Evidence: index hit log and non-empty scenes.
- [x] T2 index files + row counts -> PASS (2026-02-08 04:37 +0800)
  - Evidence: `train=6374`, `validate=1980`, `test=2170`.
- [x] T3 geometry stability smoke -> PASS (2026-02-08 05:16 +0800)
  - Evidence: checkpoints written; summary metrics finite; no unbounded badloss artifacts observed.
- [x] T4 gate check -> FAIL (2026-02-08 05:16 +0800)
  - Evidence: legacy loss-space gate failed (`pose_trans_l2_m=2.337 > 0.45`, `pose_rot_deg=1.3372 > 0.7`, `scale_loss=0.0089 > 0.0065`).
  - Note: gate historically used legacy ratio-to-1 `scale_err_mean` (not OPV2V-correct); gate policy should be migrated to `scale_to_gt_*` when training summaries expose it.
- [x] T5a auto-eval smoke -> PASS (2026-02-09 06:54 +0800)
  - Evidence: `eval_runs/auto_queue_det_eval_smoke/summary_test.json` and `eval_runs/auto_queue_det_eval_smoke/html/index.html` regenerated (`sample=5`, `seed=42`) with `run_info` + new scale diagnostics.
- [x] T5b watchdog e2e auto-eval -> PASS (2026-02-09 06:50 +0800)
  - Evidence: `experiments/local_runs/watchdog_t5b_smoke_20260208_2250_v5/logs/auto_queue_0_20260208_224950.log` shows gate PASS and auto-eval execution; `auto_queue.state=1`; artifacts at `eval_runs/watchdog_t5b_smoke_eval/auto_queue_0_20260208_224950/summary_test.json` + HTML.
- [x] T6 det AP improvement -> FAIL (2026-02-09 06:57 +0800)
  - Evidence: `eval_runs/auto_queue_det_eval_full/summary_test.json` gives candidate AP `single=0.0`, `coop=0.0` on 50-frame eval (still below locked baselines).
- [x] Scale interpretability snapshot (2026-02-09 06:57 +0800)
  - Evidence: full eval summary includes legacy ratio-space diagnostics (`|s-1|`) but OPV2V-correct diagnostics should use `scale_to_gt_*` going forward.
- [x] Fairness/metadata backfill snapshot (2026-02-14 16:22 +0800)
  - Evidence: locked baseline summaries now include `run_info.det_decode_cfg` + ratio-space scale diagnostics; fair same-frame top-k scan script output confirms current best remains `e2e_v5` (single=0.199, coop=0.138).
- [x] T7 geometry generalization scan on fixed contract (Test500) -> PASS (2026-02-16 20:10 +0800)
  - Evidence: canonical contract `eval_runs/frames_test500_seed42.json` (500, hash `4affdce1...`) + summaries under:
    - `eval_runs/geom_generalize_test500_20260216_c20260210_sfix_direct/summary_test.json`
    - `eval_runs/geom_generalize_test500_20260216_c20260202_stage2_strong/summary_test.json`
    - `eval_runs/geom_generalize_test500_20260216_c20260204_8xa800/summary_test.json`
    - `eval_runs/geom_generalize_test500_20260216_c20260205_recover_wd/summary_test.json`
- [x] Geometry baseline selection (OPV2V-correct scale metric, fixed Test500 contract) (2026-02-16 20:10 +0800)
  - Evidence: `docs/geom_generalize_test500_report_20260216.md`
  - Best scale (ratio-to-GT) in both single+coop: `c20260210_sfix_direct`
  - Baseline ckpt path (from summary `model_paths`): `experiments/local_runs/a800_sfix_direct/pinhole_pose_depth_scale/checkpoint-best.pth`
- [x] T8 scale metric-vs-viz consistency audit (2026-02-17 20:55 +0800)
  - Evidence: `eval_runs/scale_viz_audit_test500_20260217/index.html`, per-ckpt `selected_scale_representatives.json`, and `docs/scale_metric_visual_audit_20260217.md`.
- [x] T8b scale metric-vs-viz consistency audit (2026-02-20 20:40 +0800)
  - Evidence: `eval_runs/scale_viz_audit_test500_20260220/index.html`, per-ckpt `selected_scale_representatives.json`, and `docs/scale_metric_visual_audit_20260220.md`.
- [x] T9 scale-tail diagnosis closure on fixed Test500 baseline (2026-02-18 14:24 +0800)
  - Evidence: `eval_runs/_diagnose_scale_tail_test500_20260218_sfix/report.md`
  - Key findings:
    - depth valid-ratio proxy is low (`~0.028`) and narrow;
    - coop worst tail is dominated by a hard segment (`2021_08_18_19_48_05`) with joint pose blow-up (`pose_abs_m~80`) and severe under-scale (`scale_to_gt_ratio~0.12`).
- [x] T10 L1 ablation: detach scale from geometry gradient path (control vs detach, fixed Test500) -> FAIL (2026-02-18 16:51 +0800)
  - Training logs:
    - `experiments/local_runs/20260218_l1_control_v1/pinhole_pose_depth_scale/train.log`
    - `experiments/local_runs/20260218_l1_detach_v1/pinhole_pose_depth_scale/train.log`
  - Evidence:
    - `eval_runs/geom_ablation_test500_20260218_l1_control_v1/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260218_l1_detach_v1/quick_report.txt`
  - Result:
    - both runs significantly improved scale (`scale_to_gt_mult_err_mean`) but single-mode pose exceeded guardrail (>10% regression), so overall gate remained FAIL.
- [x] T11 L3 follow-up (2x4-GPU parallel from sfix/l2 starts with detach configs) -> FAIL (2026-02-18 20:33 +0800)
  - Runs:
    - `experiments/local_runs/20260218_l3a_detach_w0p6_pose12_vr0p75_from_sfix_r2/pinhole_pose_depth_scale`
    - `experiments/local_runs/20260218_l3b_detach_w0p4_pose12_vr1p_noconf_from_l2_r2/pinhole_pose_depth_scale`
  - Evidence:
    - `eval_runs/geom_ablation_test500_20260218_l3a_detach_w0p6_pose12_vr0p75_from_sfix_r2_retry/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260218_l3b_detach_w0p4_pose12_vr1p_noconf_from_l2_r2_retry/quick_report.txt`
  - Result:
    - both checkpoints collapsed (inference singular intrinsics on nearly all frames); no valid single/coop aggregate metrics remained under Test500 contract.
  - Infra fixes landed during this loop:
    - `scripts/run_geom_ablation_test500.sh` now normalizes pretrained ckpt path to absolute path (prevents silent relative-path launch failures).
    - `scripts/batch_eval.py` now skips per-frame inference singularities instead of aborting the whole contract eval.
    - `scripts/eval_geom_ckpt_test500.sh` quick-report now handles missing mode metrics gracefully.
- [x] T12 L4 recover retry (2x4-GPU, low-LR, non-detach recover branch) -> FAIL (2026-02-18 21:15 +0800)
  - Runs:
    - `experiments/local_runs/20260218_l4a_recover_w0p2_vr1p_noconf_from_l2_lr1e6/pinhole_pose_depth_scale`
    - `experiments/local_runs/20260218_l4b_recover_w0p2_vr1p_noconf_from_l1c_lr1e6/pinhole_pose_depth_scale`
  - Evidence:
    - `eval_runs/geom_ablation_test500_20260218_l4a_recover_w0p2_vr1p_noconf_from_l2_lr1e6/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260218_l4b_recover_w0p2_vr1p_noconf_from_l1c_lr1e6/quick_report.txt`
    - both summaries have empty mode metrics (`metrics.geom_model={}` for single/coop); eval logs show `Inference failed ... singular` exactly 1000 times/run (500 frames × 2 modes).
  - Result:
    - catastrophic collapse; no valid aggregate metrics under Test500 contract.
  - Infra fixes landed during this loop:
    - `scripts/run_geom_ablation_test500.sh` quick-report generation now handles missing mode metrics gracefully (no more silent crash when all frames fail).
    - `scripts/run_geom_ablation_test500.sh` now defaults to `RESUME=false` for ablation fine-tuning from pretrained ckpts (avoid inheriting stale optimizer/scheduler states).
    - `scripts/run_geom_ablation_test500.sh` now defaults `MAX_BAD_LOSS_COUNT=50` for fail-fast behavior on NaN storms.
- [x] T13 L8 pose/scale trade-off probes on fixed Test500 contract -> FAIL (2026-02-20 16:35 +0800)
  - Runs:
    - `experiments/local_runs/20260220_l8a_w0p2_vr1p_noconf_from_sfix_rf/pinhole_pose_depth_scale`
    - `experiments/local_runs/20260220_l8b3_w0p2_vr1p_noconf_pose075_from_l2/pinhole_pose_depth_scale`
  - Evidence:
    - `eval_runs/geom_ablation_test500_20260220_l8a_w0p2_vr1p_noconf_from_sfix_rf/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l8b3_w0p2_vr1p_noconf_pose075_from_l2/quick_report.txt`
  - Result:
    - scale improves strongly (`single mult` near `1.12~1.15`, `coop mult` near `1.48~1.60`), but single-mode pose guardrail still fails (`>0.0628`), so overall gate remains FAIL.
- [x] T14 L9 scale-weight sweep from sfix baseline (w0.1/w0.15) -> PARTIAL PASS (2026-02-20 18:22 +0800)
  - Runs:
    - `experiments/local_runs/20260220_l9a2_w0p1_vr1p_noconf_from_sfix/pinhole_pose_depth_scale`
    - `experiments/local_runs/20260220_l9b2_w0p15_vr1p_noconf_from_sfix/pinhole_pose_depth_scale`
  - Evidence:
    - `eval_runs/geom_ablation_test500_20260220_l9a2_w0p1_vr1p_noconf_from_sfix/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l9b2_w0p15_vr1p_noconf_from_sfix/quick_report.txt`
  - Result:
    - `l9a2` passes hard gate on both modes:
      - single: `mult=1.2557`, `pose=0.0587`, `depth=0.3214`
      - coop: `mult=1.9498`, `pose=5.6596`, `depth=0.3280`
    - `l9b2` has stronger scale but fails single pose guardrail (`pose=0.0670 > 0.0628`).
  - Promotion:
    - next-iteration starting ckpt is promoted to:
      - `experiments/local_runs/20260220_l9a2_w0p1_vr1p_noconf_from_sfix/pinhole_pose_depth_scale/checkpoint-best.pth`
- [x] T15 L10 follow-up from promoted L9 checkpoint (w0.12 ladder + pose075 checks) -> PARTIAL PASS (2026-02-20 19:50 +0800)
  - Runs:
    - `experiments/local_runs/20260220_l10a_w0p12_vr1p_noconf_from_l9a2/pinhole_pose_depth_scale`
    - `experiments/local_runs/20260220_l10b_w0p15_vr1p_noconf_pose075_from_l9a2/pinhole_pose_depth_scale`
    - `experiments/local_runs/20260220_l10c_w0p12_vr1p_noconf_pose075_from_l9a2/pinhole_pose_depth_scale`
  - Evidence:
    - `eval_runs/geom_ablation_test500_20260220_l10a_w0p12_vr1p_noconf_from_l9a2/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l10b_w0p15_vr1p_noconf_pose075_from_l9a2/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l10c_w0p12_vr1p_noconf_pose075_from_l9a2/quick_report.txt`
  - Result:
    - `l10a` is the new best passing run:
      - single: `mult=1.1151`, `pose=0.0622`, `depth=0.3161`
      - coop: `mult=1.6359`, `pose=5.8352`, `depth=0.3247`
      - `overall_pass=True`
    - `l10b` collapsed (all frames failed / missing mode metrics).
    - `l10c` preserved scale gains but failed single pose guardrail (`0.0640 > 0.0628`).
  - Promotion:
    - next-iteration starting ckpt is promoted to:
      - `experiments/local_runs/20260220_l10a_w0p12_vr1p_noconf_from_l9a2/pinhole_pose_depth_scale/checkpoint-best.pth`
- [x] T16 L11 continuation from L10-best (lower LR / w0.13 probes) -> FAIL (2026-02-20 20:21 +0800)
  - Runs:
    - `experiments/local_runs/20260220_l11a_w0p12_lr5e6_from_l10a/pinhole_pose_depth_scale`
    - `experiments/local_runs/20260220_l11b_w0p13_from_l10a/pinhole_pose_depth_scale`
  - Evidence:
    - `eval_runs/geom_ablation_test500_20260220_l11a_w0p12_lr5e6_from_l10a/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l11b_w0p13_from_l10a/quick_report.txt`
  - Result:
    - both runs collapsed into all-frame failure regime on fixed Test500 eval (missing single/coop aggregates), so neither is promotable.

- [x] T17 Plan1 distance bucket + scale-weight matrix on fixed Test500 contract -> PARTIAL PASS (2026-02-23 03:35 +0000)
  - Completed quick reports (same Test500 contract; baseline `c20260210_sfix_direct`):
    - Gate-pass (guardrails pass) candidates:
      - `geom_ablation_test500_20260222_remote_nm_w015_r1`:
        - single: `mult=1.1924`, `pose=0.0611`, `depth=0.3136`
        - coop: `mult=1.3575`, `pose=5.4196`, `depth=0.3241`
        - Evidence: `eval_runs/geom_ablation_test500_20260222_remote_nm_w015_r1/quick_report.txt`
      - `geom_ablation_test500_20260222_remote_near_w015_r1`:
        - single: `mult=1.1682`, `pose=0.0602`, `depth=0.3109`
        - coop: `mult=1.4343`, `pose=6.2021`, `depth=0.3197`
        - Evidence: `eval_runs/geom_ablation_test500_20260222_remote_near_w015_r1/quick_report.txt`
      - `geom_ablation_test500_20260222_plan1_near_sgpu_r1`:
        - single: `mult=1.1383`, `pose=0.0625`, `depth=0.3105`
        - coop: `mult=1.4667`, `pose=6.5327`, `depth=0.3204`
        - Evidence: `eval_runs/geom_ablation_test500_20260222_plan1_near_sgpu_r1/quick_report.txt`
    - Near-miss (legacy note; close to 1.3 but still > 1.3):
      - `geom_ablation_test500_20260222_local_full_w02_r1_detach`:
        - coop: `mult=1.3204` (> 1.3), single pose `0.0706` (legacy guardrail fail)
        - Evidence: `eval_runs/geom_ablation_test500_20260222_local_full_w02_r1_detach/quick_report.txt`
  - Interpretation:
    - (Legacy) Plan1 confirms coop mult can be pushed into `~1.32-1.45`.
    - (Update after T18) Promotion/gating migrated to the deployment protocol (`model_task=calibrated_sfm`, coop-first, no cross-agent GT poses as inputs). Under this protocol:
      - single pose is a **rig sanity** metric only and is no longer a hard guardrail for coop promotion;
      - the relevant guardrails are `cross_agent_pose_*` and `depth_rel_mean`, while keeping `scale_to_gt_mult_err_mean <= 1.3` (coop).
    - Therefore, the key next step for Plan1 is no longer “protect single pose”, but:
      - start from the promoted `nm4_scaleonly_w02` checkpoint and test whether bucket/curriculum/tail-reweight can improve far/tail coop performance (especially `cross_agent_pose_*`) without breaking the coop scale gate.
  - Status tracker: `docs/master_plan.md` (Section 2.3).

- [x] T18 Deployment-protocol recheck (calibrated_sfm, coop-first) + promotion -> PASS (2026-02-23 16:45 +0000)
  - Purpose:
    - Migrate comparisons away from legacy `model_task=posed_sfm` summaries and lock the deployment protocol:
      `model_task=calibrated_sfm` + `keep_camera_poses=0` + fixed Test500 contract.
  - Locked baseline (same protocol/contract):
    - `eval_runs/geom_generalize_test500_20260223_c20260210_sfix_direct_calibrated_coop/summary_test.json`
    - coop: `mult=2.3243`, `ratio=0.4865`, `cross_trans=11.3128m`, `cross_rot=2.5428deg`, `depth_rel=0.3322`
  - Promoted ckpt (passes coop scale target + guardrails):
    - ckpt: `experiments/local_runs/20260223_nm4_scaleonly_w02_lr5e6_e1_fix/pinhole_pose_depth_scale/checkpoint-best.pth`
    - evidence (FairLock recheck): `eval_runs/geom_fairlock_test500_20260226_nm4_base/summary_test.json`
    - coop: `mult=1.2693` (<=1.3 ✅), `ratio=0.9493`, `cross_trans=10.5028m`, `cross_rot=2.5052deg`, `depth_rel=0.3468` (<= baseline*1.10 ✅)

- [x] T19 Plan1-v2 distance bucket follow-ups (deployment protocol; calibrated_sfm; coop-first) -> NEAR PASS (2026-02-25 07:10 +0800)
  - Purpose:
    - Verify whether cross-agent pose is “learnable” via targeted distance-distribution finetune (Plan1 buckets),
      under the same deployment protocol + fixed Test500 contract.
  - Evidence (Test500 / coop / calibrated_sfm / keep_camera_poses=0):
    - near(0–30m): `eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_plan1_near_0_30m/quick_report.txt`
      - `mult=1.2798` ✅, `cross_trans=10.6467m` (≈baseline), `depth_rel=0.3414`
    - mid(30–60m) scaleonly: `eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_plan1_mid_30_60m/quick_report.txt`
      - `cross_trans=6.1614m` (large improvement) but `mult=1.3242` ❌
    - mid(30–60m) detach+pose12: `eval_runs/geom_ablation_test500_20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1/quick_report.txt`
      - `cross_trans=5.9934m`, `cross_rot=1.433°`, `depth_rel=0.3270` (all improved)
      - but `mult=1.3028` (just over the `1.3` gate) ❌
  - Conclusion:
    - Plan1 provides the strongest current evidence that cross-agent pose is not a hard limit (10m -> ~6m),
      but it must be paired with a **scale recover** stage to re-pass `mult<=1.3` on full Test500.

- [x] T20 Plan2 VGGT metric pose+depth long train (single+coop e30) -> DONE (2026-02-24) / Test500 eval integration -> DONE (2026-02-26)
  - Training artifacts:
    - single: `experiments/vggt/training/20260224_plan2_single_metric_e30_full/` (checkpoint-best/final)
    - coop: `experiments/vggt/training/20260224_plan2_coop_metric_e30_full/` (checkpoint-best/final)
  - Eval integration:
    - `scripts/batch_eval.py` now supports `--model_arch vggt` + `data_norm_type=identity`, and backfills
      `scale_to_gt_*` via an implied `avg_dis` scale factor when `metric_scaling_factor` is absent.
  - Evidence (Test500 / coop / deployment protocol; FairLock):
    - `eval_runs/geom_fairlock_test500_20260226_vggt_coop_e30_best/summary_test.json`
  - Result (coop mean; note: `scale_to_gt_*` is implied-scale fallback since VGGT has no scale head here):
    - `mult=1.1695`, `ratio=0.9484`, `cross_trans=9.4655m`, `cross_rot=3.4375deg`, `depth_rel=0.1148`
  - Interpretation:
    - Scale + depth are very strong under the fixed contract, and translation improves vs the MapAnything promoted baseline,
      but cross-agent rotation regresses materially; do not treat this as “deployment-ready” until rotation error is reduced.

- [x] T21 Plan1 scale recover stage from the best pose-push ckpt (mid(30–60m) detach+pose12) -> PASS (2026-02-26 10:12 +0000)
  - Goal:
    - Keep `cross_trans ~ 6m` from the mid-bucket pose-push signal, while re-passing the coop scale gate:
      `scale_to_gt_mult_err_mean <= 1.3` on the full fixed Test500 contract.
  - Attempt A (tail-only scale-only warmup; >=100m) -> FAIL (2026-02-26)
    - Evidence:
      - `eval_runs/geom_ablation_test500_20260226_srecover_tail_from_midpose12/quick_report.txt`
      - coop: `mult=2.7894`, `ratio=0.4057`, `cross_trans=23.2432m`, `depth_rel=0.7198`
    - Interpretation:
      - Tail-only scale warmup does not generalize and collapses near/mid scale; do not promote.
  - Attempt B (near-distribution scale-only warmup; <=32m) -> PASS (2026-02-26 10:12 +0000)
    - run_tag: `20260226_srecover_near_from_midpose12_lr1e5`
    - Evidence (Test500 / coop / calibrated_sfm / keep_camera_poses=0):
      - FairLock summary: `eval_runs/geom_fairlock_test500_20260226_promoted_v2/summary_test.json`
      - (history) ablation summary: `eval_runs/geom_ablation_test500_20260226_srecover_near_from_midpose12_lr1e5/summary_test.json`
      - `eval_runs/geom_ablation_test500_20260226_srecover_near_from_midpose12_lr1e5/quick_report.txt`
    - Result (coop mean):
      - `scale_to_gt_mult_err_mean=1.2984` (<=1.3 ✅)
      - `scale_to_gt_ratio_mean=1.0647`
      - `cross_agent_pose_trans_mean=5.9300m`, `cross_agent_pose_rot_mean=1.432°` (large improvement vs promoted nm4 baseline)
      - `depth_rel_mean=0.3205` (improves vs baseline)
    - Interpretation:
      - This closes the Plan1 two-stage strategy under the deployment protocol: pose can be pushed (~6m),
        then scale can be recovered without collapsing pose/depth on the full contract.
      - This checkpoint is a promotion candidate for re-opening det/e2e (T6) under a consistent protocol.

## Open Gaps / Next Iteration

- Under the deployment protocol + fixed Test500 contract, we now have a **promotion candidate that closes both**:
  - coop scale gate: `scale_to_gt_mult_err_mean <= 1.3` (T18 baseline passes; T21 candidate also passes), and
  - cross-agent pose: `cross_agent_pose_trans_mean` is pushed into the `~6m` range (T21).
- Remaining geometry gaps (before calling it “deployment-ready” for coop fusion):
  - stabilize a safer margin on scale (current candidate is `mult=1.2984`, close to the `1.3` gate);
  - understand whether the remaining pose error (~6m / ~1.43deg) is sufficient for downstream det/e2e, or whether it still dominates alignment.
- Scale supervision density is still low (`scale_valid_ratio_avg` around ~0.038 in train/test logs); structural improvements (supervision density or tail-robust weighting) are still required.
- Keep dual-track reporting in every status update: `scale_loss` (optimization) + `scale_to_gt_*` (quality/ranking), with `scale_to_gt_mult_err_mean` as primary ranking key.
- Detection T6 is now **unblocked**: re-open det/e2e using the new geometry promotion candidate and the locked test50 protocol.

## Current Execution Plan (Integrated)

Objective: Start from the latest geometry promotion candidate (T21) and re-open det/e2e, while keeping Test500/coop `scale_to_gt_mult_err_mean <= 1.3` and monitoring cross-agent pose + depth guardrails.

### Phase 0: Lock contract + metrics (non-negotiable)

- Geometry contract (for ranking/pass-fail):
  - `eval_runs/frames_test500_seed42.json` (500, hash `4affdce1...`)
  - primary: `scale_to_gt_mult_err_mean` (1 is best), `scale_to_gt_ratio_mean` (1 is best)
  - guardrails (coop-first): `depth_rel_mean`, `cross_agent_pose_trans_mean`, `cross_agent_pose_rot_mean`
- Detection contract (for final goal on det):
  - locked test50 baselines in this doc (T6).

### Phase 1: Baseline (fixed starting point; deployment protocol)

- Locked baseline (historical reference, calibrated_sfm protocol):
  - `experiments/local_runs/a800_sfix_direct/pinhole_pose_depth_scale/checkpoint-best.pth`
  - `eval_runs/geom_generalize_test500_20260223_c20260210_sfix_direct_calibrated_coop/summary_test.json`
- Previous promoted checkpoint (passes scale gate, but cross-agent pose ~10m):
  - `experiments/local_runs/20260223_nm4_scaleonly_w02_lr5e6_e1_fix/pinhole_pose_depth_scale/checkpoint-best.pth`
  - `eval_runs/geom_fairlock_test500_20260226_nm4_base/summary_test.json`
- **Current promoted checkpoint for next geometry + det/e2e iteration (start from here):**
  - `experiments/local_runs/20260226_srecover_near_from_midpose12_lr1e5/pinhole_pose_depth_scale/checkpoint-best.pth`
  - `eval_runs/geom_fairlock_test500_20260226_promoted_v2/summary_test.json`

### Phase 2: Diagnose (completed on baseline; rerun after each structural change)

- Build an evidence-backed explanation for why scale is still under-scale and why coop has tail failures:
  1) Depth supervision density on the contract:
     - compute per-frame depth valid ratio (depth>0) and summarize distribution;
     - correlate worst `scale_to_gt_err` frames with depth valid ratio.
  2) Outlier analysis:
     - list top-N worst coop frames by `cross_agent_pose_trans_m` and `scale_to_gt_err`;
     - confirm if outliers are driven by pose blow-ups (many are).
- Current evidence:
  - `eval_runs/_diagnose_scale_tail_test500_20260218_sfix/report.md`

### Phase 3: Fix ladder (minimal → structural)

Rule: one change at a time, and every change must be evaluated on the fixed Test500 contract.

1) **Loss-only** (fast iteration; low chance to fully solve but useful):
   - keep `direct_scale_loss=True`, `loss_in_log=True` as default (aligned with ratio error).
   - try scale bias correction term (batch-level mean log-ratio → 0) if under-scale persists.
2) **Increase supervision density** (expected to be required):
   - build `opv2v_depth_dense_v1` from existing sparse depth (safe inpainting/fill) and re-train from baseline.
   - if still insufficient, escalate to a stronger pseudo-depth pipeline (explicitly recorded as “noisy label”).
3) **Outlier robustness / coop tail**:
   - reweight/skip samples by depth valid ratio (`valid_ratio`) and/or by pose consistency;
   - add cross-agent scale consistency (same timestamp: scales should match) to stabilize coop.

### Phase 4: Re-open det/e2e (only after geometry gate)

- Once geometry target is met on Test500:
  - run det/e2e training and evaluate under locked T6 protocol.

## Resource-Allocated Execution Plan (Actionable)

### Compute policy

- Always prefer parallelism for *ablation* (short runs), and reserve “long runs” only after we have a winning direction.

Available options (pick what is actually available):
- Single node 8xA800: run 1 experiment at a time.
- If 4x4090 is available elsewhere: use it for fast ablations + eval, while A800 runs longer jobs.
- If 2x8 A800 is available (Volcano): run 2 experiments in parallel.

### Geometry pass/fail gate (Test500 contract)

Pass (geometry ready for det) when all hold on Test500 under the **deployment protocol** (coop-first):
- `metrics.<model>.coop.scale_to_gt_mult_err_mean <= 1.3`
- `metrics.<model>.coop.scale_to_gt_ratio_mean` close to 1 (report bias explicitly; do not “hide” it behind mult_err)
- coop guardrails vs a locked baseline under the same protocol:
  - `metrics.<model>.coop.depth_rel_mean <= baseline * 1.10`
  - `metrics.<model>.coop.cross_agent_pose_trans_mean <= baseline * 1.10`
  - `metrics.<model>.coop.cross_agent_pose_rot_mean <= baseline * 1.10`
- single-mode metrics are **sanity-only** (in deployment, intra-agent rig is known and can be provided/locked).

### Evaluation standard (mandatory)

Run eval on the fixed Test500 contract (deployment protocol; coop-first; calibrated intrinsics; no cross-agent GT pose inputs):
```
source /J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/activate
PYTHONPATH=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything \
python map-anything/scripts/batch_eval.py \
  --split test --frames_json map-anything/eval_runs/frames_test500_seed42.json --modes coop \
  --model_task calibrated_sfm \
  --models geom_model=<CKPT> --model_filter geom_model \
  --output_root <eval_runs/NAME>
```

Optional: single rig sanity check (not used for coop ranking/pass-fail; only for debugging depth/scale stability when rig extrinsics are provided):
```
source /J6P-perception/yijinxiong_workspace/venvs/mapanything-cu121/bin/activate
PYTHONPATH=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything \
python map-anything/scripts/batch_eval.py \
  --split test --frames_json map-anything/eval_runs/frames_test500_seed42.json --modes single \
  --keep_camera_poses --model_task calib_posed_sfm \
  --models geom_model=<CKPT> --model_filter geom_model \
  --output_root <eval_runs/NAME>_single_rig
```

For “quick iteration” (optional), you may also run a smaller contract (e.g. 50/100 frames), but **must** re-check on Test500 before any promotion.

### Reporting / logging (mandatory)

For each run, record:
- `train.log`, `launch.log`, `summary_test.json`
- checkpoint path used for eval
- final metrics (coop-first): `scale_to_gt_mult_err_mean`, `scale_to_gt_ratio_mean`, `cross_agent_pose_trans_mean`, `cross_agent_pose_rot_mean`, `depth_rel_mean`
