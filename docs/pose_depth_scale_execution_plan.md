# Pose / Depth / Scale Improvement Execution Plan (OPV2V)

Last updated: 2026-02-23

> Status:
> - This file keeps detailed historical stage notes (S4/S5/S7...).
> - For the current authoritative objective + pass/fail + resource plan, use:
>   - `map-anything/docs/goal_based_test_plan.md`
>   - `docs/master_plan.md`
>   - `docs/scale_metric_visual_audit_20260217.md`

> Important (OPV2V scale metric policy, 2026-02-17):
>
> - If training/eval uses `norm_mode=avg_dis`, OPV2V “GT scale” is a **per-frame factor** `g` (not constant 1).
> - Therefore **do not** treat `scale_err_mean=mean(|s-1|)` as the primary quality metric on OPV2V. It is kept only as a legacy field.
> - For pass/fail, ranking, and “does scale improve”, use **ratio-to-GT** metrics from `scripts/batch_eval.py`:
>   - `scale_to_gt_err_mean = mean(|s/g - 1|)` (0 is best)
>   - `scale_to_gt_log_err_mean = mean(|log(s/g)|)` (0 is best; symmetric for 0.5x and 2x)
>   - More intuitive: `scale_to_gt_mult_err_mean = exp(scale_to_gt_log_err_mean)` (1 is best)

## Goals and Constraints
- Goal: Reduce pose/depth/scale errors on OPV2V Coop to a clearly lower and stable level.
- Target for scale (OPV2V-correct): `scale_to_gt_mult_err_mean <= 1.3` on a fixed-contract eval (see “Evaluation Protocol”).
- Use all available compute; if only one node is available, run variants sequentially.
- Every stage must define pass criteria. If not met, iterate with changes until it passes.
- Prefer minimal, reversible changes first; only then move to heavier data/label changes.
- Stop condition: only declare completion after the **OPV2V-correct** Stage 3 gate passes (`scale_to_gt_*` + pose/depth guardrails).

## Baselines (Current Evidence)
Current promoted checkpoint (deployment protocol: `calibrated_sfm`, coop-first, fixed Test500 contract):
- ckpt: `experiments/local_runs/20260223_nm4_scaleonly_w02_lr5e6_e1_fix/pinhole_pose_depth_scale/checkpoint-best.pth`
- evidence: `eval_runs/geom_recheck_test500_20260223_nm4_scaleonly_w02_calibrated/summary_test.json`
- key metrics (Test500, coop mean):
  - scale: `scale_to_gt_mult_err_mean=1.2693` (<=1.3 ✅), `scale_to_gt_ratio_mean=0.9493`
  - cross-agent pose: `cross_agent_pose_trans_mean=10.5028m`, `cross_agent_pose_rot_mean=2.5052deg`
  - depth: `depth_rel_mean=0.3468`

Locked historical baseline for guardrails (same protocol/contract):
- `eval_runs/geom_generalize_test500_20260223_c20260210_sfix_direct_calibrated_coop/summary_test.json`

Historical stability scan notes (legacy / pre-metric-correction; do not use for pass/fail):
- S4 (w=0.1, epoch1 test summary)
  - scale_err_mean: 8.3872
  - scale_log_err_mean: 2.2360
  - scale_ratio_mean: 9.3872
  - pose_trans_l2_m: 0.6870
  - pose_rot_deg: 1.4039
  - depth_z_mae_m: 1.6784
  - depth_z_rmse_m: 5.5995
  - ckpt: .../20260210_a800_s4_direct_w0p1/pinhole_pose_depth_scale/checkpoint-best.pth
- S5 (w=0.2, epoch2 test summary)
  - scale_err_mean: 8.6981
  - scale_log_err_mean: 2.2689
  - scale_ratio_mean: 9.6981
  - pose_trans_l2_m: 0.6600
  - pose_rot_deg: 1.4368
  - depth_z_mae_m: 1.6575
  - depth_z_rmse_m: 5.5737
  - ckpt: .../20260210_a800_s5_direct_w0p2/pinhole_pose_depth_scale/checkpoint-best.pth

Regression guardrails (vs current baseline on Test500):
- pose_abs_mean <= baseline * 1.10 (both single+coop)
- depth_rel_mean <= baseline * 1.05 (both single+coop)
If any exceed these, mark regression even if scale improves.

## Evaluation Protocol
Primary evaluation source (recommended, reproducible, OPV2V-correct):
- Fixed-contract eval via `scripts/batch_eval.py` (so scale can be interpreted as `s/g`):
  - Example contract: `eval_runs/frames_test500_seed42.json`
  - Output: `eval_runs/<run_name>/summary_test.json` with `scale_to_gt_*` keys.

Quick gate (optional):
- 1_000 sample eval (logs: "Testing on 1_000 @ OPV2VCoopDataset:test ...").

Metrics to track:
- scale (primary, OPV2V-correct): `scale_to_gt_err_mean`, `scale_to_gt_log_err_mean`
  - recommended to report: `scale_to_gt_mult_err_mean = exp(scale_to_gt_log_err_mean)` (1 is best)
- scale (legacy, compatibility only): `scale_err_mean`, `scale_log_err_mean`, `scale_ratio_mean`
- pose_trans_l2_m, pose_rot_deg
- depth_z_mae_m, depth_z_rmse_m

Invalid run criteria (do NOT trust metrics):
- Any NaN/Inf loss during train or eval.
- `scale_to_gt_*` missing/non-finite (or all-zero together with `loss=nan`) on the eval summary.

Metric extraction (recommended, fixed-contract eval; deployment protocol):
- Run eval (example: test500 contract, coop-first):
  - `PYTHONPATH=$(pwd) python scripts/batch_eval.py --split test --frames_json eval_runs/frames_test500_seed42.json --modes coop --model_task calibrated_sfm --models geom_model=<CKPT> --model_filter geom_model --output_root eval_runs/<RUN_NAME>`
- Extract key scale numbers from `eval_runs/<RUN_NAME>/summary_test.json`:
  - `metrics.geom_model.coop.scale_to_gt_log_err_mean`
  - (and report `scale_to_gt_mult_err_mean = exp(scale_to_gt_log_err_mean)`)

Metric extraction (legacy, from training full-test logs):
- Find test block: `grep -n \"Testing on full/OPV2VCoopDataset:test\" <log>`
- Extract last `Test Epoch` line: `grep \"Test Epoch: \\[.*\\]\" <log> | tail -n 1`

## Pass Criteria (Stage Gates)
Stage 0 (Stability gate):
- No NaN/Inf, training completes configured epochs.

Stage 1 (Scale improvement gate):
- scale_to_gt_mult_err_mean <= 2.0 (i.e., typical multiplicative error <=2x)
- pose/depth must not regress by >10% vs current baseline (Test500 contract)

Stage 2 (Intermediate gate):
- scale_to_gt_mult_err_mean <= 1.6 (i.e., typical multiplicative error <=1.6x)
- pose/depth must not regress by >10% vs current baseline (Test500 contract)

Stage 3 (Final target gate):
- coop-first:
  - `metrics.geom_model.coop.scale_to_gt_mult_err_mean <= 1.3`
  - plus guardrails under the same protocol/contract (see `map-anything/docs/goal_based_test_plan.md`):
    - `cross_agent_pose_*` does not regress >10% vs baseline
    - `depth_rel_mean` does not regress >10% vs baseline

Note: single-mode metrics are sanity-only (deployment can provide intra-agent rig extrinsics).

Note: If these thresholds are too strict or too loose, update once and propagate to all stages.

## Resource Allocation
- Node A (local): main candidate run.
- Node B (if available): parallel variant run.
- If only one node is available, queue variants sequentially in priority order.

Artifacts and logs:
- Run dir: `map-anything/experiments/local_runs/<run_tag>/pinhole_pose_depth_scale/`
- Log dir: `/J6P-perception/yijinxiong_workspace/log/mlp_entry/entry_<timestamp>_<host>_rank0.log`
- Best/last ckpt: `checkpoint-best.pth`, `checkpoint-last.pth`, `checkpoint-final.pth`

Checkpoint selection rule:
- Primary: lowest `scale_to_gt_mult_err_mean = exp(scale_to_gt_log_err_mean)` on fixed-contract eval.
- Tie-breaker: lower `scale_to_gt_err_mean`, then lower pose_trans_l2_m, then depth_z_mae_m.

Preflight checklist (before each run):
- Base ckpt exists and is readable.
- Dataset roots exist: `${machine.opv2v_images_root}`, `${machine.opv2v_depth_root}`.
- Index dir exists: `${machine.opv2v_images_root}/opv2v_index`.
- Log dir writable: `/J6P-perception/yijinxiong_workspace/log/mlp_entry/`.
- TMPDIR short path set (e.g., `/tmp/ma_tmp_4090`) to avoid `AF_UNIX path too long` on 4090.

## Stage Plan (Execute in Order)

### Stage 0: Diagnose & stabilize (mandatory)
0.1 Clean badloss artifacts + cap future dumps
- If a run emits badloss_* dumps, delete them before next run to avoid disk pressure.
- Ensure badloss dump cap is enabled in train params (bad_loss_dump_max/bad_loss_dump_stride).
Pass: no exploding disk usage; no repeated badloss dumps in a single run.

0.2 Verify scale supervision density
- Confirm `*_scale_valid_ratio_avg` is logged in train logs.
- If missing, patch logger to emit the key and re-run a 1000-sample quick gate.
Pass: scale_valid_ratio_avg visible and >1% on average; if <1%, prioritize Stage 2.2.

0.3 Baseline sanity check (short eval)
- Run 1_000-sample eval on S5/S7a to anchor `scale_to_gt_*` and avoid regression drift.
Pass: baseline metrics logged and comparable.

### Stage 1: Scale-only warmup -> full loss (S6)
Purpose: stabilize scale head before full geometry loss coupling.

Variant A (Node A):
- Warmup: scale-only, 1 epoch
- Then: full loss, 2 epochs
- Base ckpt: S5 best
- Loss (no gating, noconf): `opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2_noconf.yaml` -> `opv2v_vggt_pose_scale_loss_recover_direct_w0p2_noconf.yaml`

Variant B (Node B):
- Warmup: scale-only, 2 epochs (or 1 epoch + lower LR)
- Then: full loss, 2 epochs
- Base ckpt: S5 best

Pass criteria: Stage 1 gate above.
If Stage 1 passes but Stage 2 (final target) is not met, proceed to Stage 2.
If Stage 1 fails, proceed to Stage 2.

### Stage 2: Increase effective scale supervision
2.1 Mask gating for scale loss
- Skip scale loss for samples with valid depth ratio below a threshold (e.g., <1%).
- Goal: reduce noisy gradients when supervision is too sparse.
Implementation target:
- Add `min_scale_valid_ratio` to scale loss in `mapanything/train/losses.py` (direct scale loss path).
- Compute valid_ratio from `valid_mask` (depth>0) and skip scale loss if below threshold.
- Log per-batch valid_ratio to confirm behavior.

2.2 Densify depth supervision
- Add or enhance pseudo depth (e.g., smoothing/denoising opv2v_depth or depth completion).
- Recompute valid_mask with denser depth maps.
Implementation target:
- Generate dense depth via `map-anything/depth/batch_depth.py`.
- Store at `map-anything/data/opv2v_depth_dense`.
- Switch depth root in `map-anything/configs/machine/local_a800.yaml` (opv2v_depth_root)
  and re-run with the same dataset config `map-anything/configs/dataset/opv2v_coop_ft_2a8v_full.yaml`.

Pass criteria: Stage 1 gate then Stage 2 gate.
If fail: proceed to Stage 3.

### Stage 3: Loss shaping / representation
- Explore loss_in_log / scale_ratio vs scale_err, or clamp ranges.
- Tune scale_loss_weight (0.2 -> 1.0) with best warmup recipe.
- Keep pose/depth stable (no regression >10%).
Implementation target:
- Edit loss configs in `map-anything/configs/loss/` (direct_w0p1/w0p2 or new variants).
- If using log-scale loss, keep `loss_in_log=True`; if using ratio, adjust scale head mode.

Pass criteria: Stage 3 gate.
If fail: proceed to Stage 4.

### Stage 4: Scale 1-3 target (calibration + consistency)
4.1 Scale calibration (dataset-level)
- Compute global scale factor between predicted depth and lidar depth.
- Apply correction factor to scale target or depth map.
Implementation target:
- Add a calibration script under `map-anything/scripts/` to compute scale factor on 1_000 samples.
- Store factor and apply in loss (scale_gt *= factor) or in depth preprocessing.

4.2 Cross-agent scale consistency
- Add a loss term that penalizes scale variance across cooperative agents in the same scene.
Implementation target:
- Implement in `mapanything/train/losses.py` near direct scale loss.
- Use per-sample scales and add L2 on pairwise log-ratio.

4.3 Sample reweighting by valid_ratio
- Upweight samples with valid_ratio >= 5%, downweight <1%.
- Goal: ensure scale loss dominated by reliable supervision.
Implementation target:
- Add weights in loss computation using valid_ratio buckets.

Pass criteria: Stage 3 gate (final target).
If fail: iterate Stage 4 with smallest change first.

## Current Status (2026-02-18)
- Baseline ckpt is now fixed by the OPV2V-correct scale metric under the Test500 contract:
  - ckpt: `experiments/local_runs/a800_sfix_direct/pinhole_pose_depth_scale/checkpoint-best.pth`
  - evidence: `eval_runs/geom_generalize_test500_20260216_c20260210_sfix_direct/summary_test.json`
- Tail diagnosis on the same fixed contract is completed:
  - `eval_runs/_diagnose_scale_tail_test500_20260218_sfix/report.md`
  - coop worst tail is dominated by a hard segment (`2021_08_18_19_48_05`) with joint pose blow-up (~80m) and severe under-scale (~0.12x).

Operational note:
- Old auto-loop scripts (Stage1→Stage3) historically gated on legacy train-log `scale_err_mean`. They are not suitable for OPV2V scale pass/fail.
- Current gating/ranking must use fixed-contract `batch_eval.py` and `scale_to_gt_*` keys (see `docs/master_plan.md` / `docs/goal_based_test_plan.md`).

Notes:
- New loss configs for Stage 2: `opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2_vr1p.yaml` and `opv2v_vggt_pose_scale_loss_recover_direct_w0p2_vr1p.yaml`.
- Added noconf scale-only variant: `opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2_vr1p_noconf.yaml`.
- Added noconf variant to avoid NaN from conf loss: `opv2v_vggt_pose_scale_loss_recover_direct_w0p2_vr1p_noconf.yaml`.
- Added higher scale-weight variant for Stage3: `opv2v_vggt_pose_scale_loss_recover_direct_w1p0_vr1p_noconf.yaml`.
- Added `min_scale_valid_ratio` and `*_scale_valid_ratio_avg` logging in `map-anything/mapanything/train/losses.py`.
- S7a final full test (epoch2, [203/204]): scale_err_mean=11.7760, scale_log_err_mean=2.5458, pose_trans_l2_m=0.7389, pose_rot_deg=1.6028, depth_z_mae_m=1.6678.
- Added no-gating noconf Stage1 configs: `opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2_noconf.yaml`, `opv2v_vggt_pose_scale_loss_recover_direct_w0p2_noconf.yaml`.

## Execution Commands (Template)
Common:
- Entry: map-anything/bash_scripts/train/mlp_entry_pinhole_pose_depth_scale_recover.sh
- Loss configs:
  - scale-only warmup: opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2
  - full loss: opv2v_vggt_pose_scale_loss_recover_direct_w0p2

Example (Node A, warmup 1 epoch):
- LOSS_NAME=opv2v_vggt_pose_scale_loss_recover_direct_scaleonly_w0p2
- PRETRAINED_CKPT=<S5-best>
- EPOCHS=1

Example (Node A, full loss):
- LOSS_NAME=opv2v_vggt_pose_scale_loss_recover_direct_w0p2
- PRETRAINED_CKPT=<warmup-best>
- EPOCHS=2

Monitoring and recovery:
- Check GPU load:
  - Local: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
  - Remote: `ssh -p 12222 10.199.10.207 \"nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader\"`
- If a run exits before all epochs:
  - Resume with same run_tag (train_params.resume=true is already enabled).
  - If resume fails, re-run with a new run_tag and record the reason in experiment_logbook.

## Iteration Rule
After each stage:
1) Extract full test metrics from logs.
2) Compare to pass criteria.
3) If fail, apply the next minimal change and rerun on both nodes.
4) Repeat until Stage 3 gate is satisfied.

Logbook update:
- Record each run (run_tag, ckpt path, full-test metrics, pass/fail) in `map-anything/docs/experiment_logbook.md`.
