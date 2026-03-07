# Experiment Logbook (MapAnything)

Purpose: Keep a structured, time-stamped record of training issues, fixes, and outcomes.
This log is designed to be human-readable and machine-parseable for automation.

Formatting rules (must follow):
- Use ISO timestamps with timezone, e.g., "2026-02-03 08:10 +0800".
- Record both MICRO (event-level) and MACRO (summary-level) entries.
- Every MICRO entry must include: Context, Symptom, Root Cause (best guess), Fix, Validation, Next.
- Every MACRO entry must include: Scope, Key wins, Key risks, Next experiments.

---

## MACRO SUMMARY (rolling)

### 2026-03-06 17:05 +0800
- 用 GSD Phase 1 锁定新的 **authoritative success surface**：最终 success 只能定义为 `full2170 + deploy-like coop + det_ap_iou >= 0.20`。
- 新文档：`.planning/phases/01-success-surface-lock/01-success-surface.md`、`.planning/phases/01-success-surface-lock/01-baseline-scoreboard.md`、`.planning/phases/01-success-surface-lock/01-promotion-checklist.md`。
- 明确规定：Test50/Test500/Test2000 仅作 quick/gate，cylindrical 高 AP 仅作 high-upside evidence，不得冒充 final success。

### 2026-03-06 16:04 +0800
Scope: 把本地 OPV2V / MapAnything / VGGT 评测结果从“散落的 `summary_test.json` + 人工文档”收敛为统一评测标准、curated runs、以及面向 GitHub Project 的友好展示链路。
Key wins:
- 远端 `coopVGGT` Project V2 字段集已从本机只读核验，`sync_benchmark_project_v2.py` 能与当前字段模型对齐。
- 首批 canonical contracts（Geometry/Test500 deployment、Det/Test50 FairLock）和 curated runs 已冻结到 `configs/unified_eval_project_registry.json`。
- 新增 `scripts/build_unified_eval_board.py`，可从 registry 一次性生成本地 overview + per-run manifests，并为后续 Project 同步做 dry-run。
Key risks:
- 仍有大量历史 `eval_runs/**/summary_test.json` 尚未进入 curated 层；不能把“全仓扫描”误当成“都应上板”。
- PredPose det 仍然噪声较大，更适合作为 regression / sanity 信号，而不是精细几何排序器。
Next experiments:
- 先审阅第一版 `unified_eval_board` 本地 overview，再决定首批写远端的 runs。
- 只有当本地展示仍有关键缺口时，才针对固定合同发起最小 rerun / inference / training。

### 2026-02-25 12:44 +0800
Scope: 收敛 20260224 Plan1 距离桶 pipeline（`20260224_193935_plan1_pipeline_8x4090`）在“新机不一定是 8 卡”的现实下的实际执行结果：补齐 lane1（mid/tail/densify_r2/detach_rpose2）并产出最终 scoreboard；同时梳理与 20260222 旧 Plan1 bucket-only 试跑（`20260222_plan1_*_sgpu_r1`）的设置/结果差异。
Key wins:
- 最终产出统一 scoreboard：`experiments/long_runs/20260224_193935_plan1_pipeline_8x4090__pipeline/artifacts/scoreboard_20260224_193935_plan1_pipeline_8x4090.md`（matched_rows=5），并在 deploy 口径（Test500 / coop-first / calibrated_sfm）下回填关键指标。
- detach follow-up 在当前 baseline（`nm4_scaleonly_w02`）上属于“安全改动”：`l1_detach_geom_scale_w0p4_pose12` 与 `..._rpose2` 都能保持 `mult<=1.3` 且 cross/depth guardrails 通过。
- mid 桶依然有“cross-agent pose 明显变好”的信号，但 scale `mult` 仍轻微越线（见 `20260224_210949_lane1_gpu45_plan1_mid_30_60m`）。
Key risks:
- tail（>=100m baseline）仍是主要失败源：scale/depth/cross-agent pose 会一起崩（`20260224_210949_lane1_gpu45_plan1_tail_100_inf`）。
- far 桶出现“scale ok 但 depth_rel 退化越线”的新 failure mode（需要单独定位：是否是 bucket-only finetune 的分布偏置/过拟合）。
Next experiments:
- 以 mid 桶 pose-improved ckpt 为起点，设计一个 “scale recover” 阶段（更小 LR + 混合 near 桶/或更强 scale 约束）把 `mult` 拉回 <=1.3，同时验证 `cross_trans` 不反弹。
- 把 Plan1 pipeline runner 做成“GPU 数自适应 + 不覆盖旧 run_tag”的版本，避免重复执行/覆盖导致的结果混淆。

### 2026-02-24 23:00 +0000
Scope: 完成 L2 densify 半径 sweep 的 fixed Test500（部署口径）结论回填；并在 2026-02-24 启动 Plan1 距离桶诊断流水线（near/mid/far/tail + follow-ups）以定位 cross-agent pose 的可学习区间与退化边界。
Key wins:
- L2 densify v1（最近邻填充）在 Test500 / coop-first / calibrated_sfm 口径下 **未达标**：`mult>1.3`，且 cross-agent pose 无改善；可以明确不走这条“简单 densify”路线。
- Plan1 mid 桶（30–60m）出现了**显著的 cross-agent pose 改善信号**（`cross_trans` 从 ~10.50m 降到 `~6.16m`），说明跨车位姿并非不可学，而是强依赖采样/距离分布。
- detach follow-up（w0.4 + pose12）在更大训练集（full coop train）下保持 `mult≈baseline` 且 pose/depth 有小幅改善，证明该改动在当前配置下至少是“安全的”。
Key risks:
- mid 桶虽然把 `cross_trans` 打下来，但 scale `mult` 轻微越线（`1.3242 > 1.3`），存在“pose 变好但 scale 变坏”的耦合风险，必须设计后续阶段把 scale 拉回 <=1.3 并验证不反弹。
- tail 桶（100–inf）训练会显著拉崩 scale/depth（严重过拟合/分布偏置/噪声），不适合作为单独 finetune 阶段。
Next experiments:
- 以 mid 桶 pose-improved ckpt 为起点，加一个“scale recover”阶段（更强 scale 约束/更小 LR/混合 near 桶）以把 `mult` 拉回 <=1.3，同时守住 `cross_trans` 的收益。
- 继续完成 lane1 的 densify_r2 / detach_rpose2（在正确 GPU 列表下）并统一用同合同复评后再决定是否保留该分支。

### 2026-02-23 16:50 +0000
Scope: 完成“部署口径（`calibrated_sfm`）”迁移并在 fixed Test500 合同下重评测；确认新的可 promotion 起点（coop scale 过线）并同步修正文档/脚本口径。
Key wins:
- 锁定部署评测口径：`model_task=calibrated_sfm` + `keep_camera_poses=0` + `frames_test500_seed42.json`。
- 在该口径下重评测并促成 promotion：
  - baseline（`c20260210_sfix_direct`）：coop `mult=2.3243`, `ratio=0.4865`；evidence: `eval_runs/geom_generalize_test500_20260223_c20260210_sfix_direct_calibrated_coop/summary_test.json`
  - promoted（`nm4_scaleonly_w02`）：coop `mult=1.2693` (<=1.3 ✅), `ratio=0.9493`；evidence: `eval_runs/geom_recheck_test500_20260223_nm4_scaleonly_w02_calibrated/summary_test.json`
- 同步工程化输出（避免再“口径混用”）：
  - 更新 `docs/master_plan.md` / `docs/goal_based_test_plan.md`（coop-first + calibrated_sfm）
  - `scripts/scoreboard_scan_geom_contract.py` 增加 cross-agent/rig pose 列
  - `scripts/run_geom_ablation_test500.sh` 默认切到 coop-first + `TARGET_SCALE_MULT=1.3`，并支持 cross-agent pose guardrails
Key risks:
- 跨车相对位姿误差仍是 10m 级别（baseline `cross_trans≈11.31m` -> promoted `≈10.50m`），对协同融合/BEV det 仍是硬瓶颈。
- scale 已基本回正，但如果继续追求 pose，需防止 scale 回退到 >1.3。
Next experiments:
- 以 promoted ckpt 为起点做 coop pose-focused finetune（更强 cross-agent pose supervision / 更鲁棒的 outlier 处理），每轮必须回到 Test500 合同评测。
- 在 cross-agent pose 进入可用区间后，再重开 det/e2e（T6）。

### 2026-02-23 03:35 +0000
Scope: 20260222 Plan1（距离桶 + scale_loss_weight 矩阵）在 fixed Test500 合同下全部收敛完结；汇总结果并明确下一步修正方向（继续迭代直到 coop `mult<=1.3`）。
Key wins:
- 本轮矩阵已全部落盘 `quick_report.txt`，并能稳定复现“coop scale 显著改善但单车 pose 容易退化”的核心矛盾。
- 当前最优的 gate-pass 候选（guardrails pass）已更新为：
  - `20260222_remote_nm_w015_r1`（0-60m, w0.15）：single mult=1.1924 pose=0.0611；coop mult=1.3575
  - Evidence: `eval_runs/geom_ablation_test500_20260222_remote_nm_w015_r1/quick_report.txt`
Key risks:
- 仍未达到最终验收：single+coop `scale_to_gt_mult_err_mean <= 1.3`；且“压 coop mult”会显著推高 single pose（最典型是 full bucket）。
- 有 near-miss（coop mult 已接近 1.3）但仍被 single pose guardrail 卡住：
  - `20260222_local_full_w02_r1_detach`：coop mult=1.3204，但 single pose=0.0706（fail）
Next experiments:
- 以 `20260222_remote_nm_w015_r1` 的 ckpt 为起点做稳定性优先的续训/微调（更小 LR / 更强 outlier 抑制 / 更严格 pose 保护），每轮必须回到 fixed Test500 合同评测；
- 若两轮内仍无法把 coop mult 压到 <=1.3，则按 `docs/master_plan.md` 升级 Fix Ladder（L1/L2）。

### 2026-02-22 19:26 +0000
Scope: 单机 8xA800（`10.199.96.61`）并行推进“距离桶 + loss 权重矩阵”的 fixed Test500 合同评测；尝试补齐桶组合（0-60 / 30-100）以诊断退化边界。
Key wins:
- 已产出 20260222 批次 12 个 `quick_report.txt`（同一 Test500 合同口径）并形成可比对的第一批结论：
  - 当前唯一 `overall_pass=True`：`geom_ablation_test500_20260222_plan1_near_sgpu_r1`
    - single: `mult=1.1383`, `pose=0.0625`, `depth=0.3105`
    - coop: `mult=1.4667`, `pose=6.5327`, `depth=0.3204`
    - evidence: `eval_runs/geom_ablation_test500_20260222_plan1_near_sgpu_r1/quick_report.txt`
- 明确了本轮主现象：**coop scale 能大幅改善到 ~1.39-1.49，但 single pose guardrail 常被击穿**（多数 run 不可 promotion）。
Key risks:
- 最终目标仍未达成：`coop scale_to_gt_mult_err_mean <= 1.3` 仍未出现；full 桶与 w0.15 sweep 结果决定是否要升级 Fix Ladder。
- 另一台 8xA800（`10.199.96.16:12222`）SSH 鉴权失败，算力暂不可调度，影响吞吐。
Next experiments:
- 等待 in-flight 的 `remote_full_w02 / remote_full_w012 / w0.15*` 产出 `quick_report.txt`，统一入表决策下一轮主线；
- 若 full 桶仍无法兼顾（coop scale 改善 + single pose guardrail），按 `docs/master_plan.md` 的 Fix Ladder 优先升级 L1（scale 与几何分支解耦），再考虑 supervision 密度增强；
- 修复 `10.199.96.16` 的 trusted SSH key 安装，恢复双机并行。

### 2026-02-22 16:50 +0000
Scope: 双开发机（2x8 A800）资源拉满，推进方案一分桶/权重矩阵并行验证。
Key wins:
- 两台机器均已恢复并行训练状态，`python -m torch.distributed.launch` 主进程已覆盖远端 8 路 + 本机 8 路（含新增 `w0.15/full` 补位）。
- 本机后台拉起方式完成修正：由普通 `nohup` 切换为 `setsid -f bash -lc`，避免子进程异常回收导致“看似启动但实际退出”。
- 关键 run 全部进入训练/测试循环并持续写入 `launch.log`，未再出现此前的空转状态。
Key risks:
- 部分 run 处于 test/eval 阶段时 GPU util 会短时接近 0（显存仍占用），需要用日志进度判定活性，不能只看瞬时利用率。
- full 桶（`6333` iter）耗时明显长于 near/mid/far，出结果节奏不一致。
Next experiments:
- 等待各 run 产出 `eval_runs/geom_ablation_test500_<run_tag>/quick_report.txt` 后统一入表；
- 先做同桶 `w0.12 vs w0.2` 比较，再做 full 桶 `w0.12/w0.15/w0.2` 比较；
- 依据同合同 Test500 结果选下一轮主线 ckpt 与 loss 配置，继续保持双机满载。

### 2026-02-20 20:25 +0800
Scope: Post-L9 optimization sweep from promoted ckpt (L10/L11) under fixed Test500 contract.
Key wins:
- Found a stronger passing checkpoint than `l9a2`:
  - `l10a` (`w0.12`, from `l9a2`) with `overall_pass=True`.
  - Metrics:
    - single: `mult=1.1151`, `pose=0.0622`, `depth=0.3161`
    - coop: `mult=1.6359`, `pose=5.8352`, `depth=0.3247`
- Verified two important boundaries quickly with parallel ablations:
  - stronger scale push + pose075 (`l10b`) can fully collapse;
  - continuation from the stronger pass ckpt (`l11a/l11b`) can also collapse if stability margin is insufficient.
- Promoted new start checkpoint:
  - `experiments/local_runs/20260220_l10a_w0p12_vr1p_noconf_from_l9a2/pinhole_pose_depth_scale/checkpoint-best.pth`.
Key risks:
- Final coop target remains open (`mult=1.6359`, target `<=1.3`).
- Single pose is near guardrail on best pass run (`0.0622` vs threshold `0.0628`), so naive extra scale pressure has little headroom.
Next experiments:
- Keep `l10a` as base and move to stability-first improvements (tail/outlier handling or supervision-density) instead of stronger direct-scale weights.
- Require fail-fast checks for NaN/bad-loss and immediate demotion of runs with missing mode metrics.

### 2026-02-20 18:25 +0800
Scope: L9 dual-run rerun on fixed Test500 contract (w0.1 / w0.15 direct-scale sweep from sfix baseline) and gate closure check.
Key wins:
- Fixed loss-config eval hazard for new ablation configs (`norm_mode` / `depth_type_for_loss` quoted as strings), avoiding `NameError` at criterion construction.
- Completed two parallel 4-GPU runs end-to-end with identical contract and baseline:
  - `20260220_l9a2_w0p1_vr1p_noconf_from_sfix`
  - `20260220_l9b2_w0p15_vr1p_noconf_from_sfix`
- Reached first fixed-contract hard-gate pass in this branch:
  - `l9a2`: `overall_pass=True` (`single mult=1.2557`, `coop mult=1.9498`, pose/depth guardrails both pass).
- Promoted next-iteration start checkpoint:
  - `experiments/local_runs/20260220_l9a2_w0p1_vr1p_noconf_from_sfix/pinhole_pose_depth_scale/checkpoint-best.pth`.
Key risks:
- Final scale objective is still open in coop mode (`mult=1.9498`, target `<=1.3`).
- `l9b2` confirms stronger scale is reachable (`single mult=1.1221`, `coop mult=1.7011`) but single pose guardrail regresses (`0.0670 > 0.0628`), so pose-scale trade-off remains the blocker.
Next experiments:
- From promoted `l9a2`, run pose-preserving scale push (intermediate scale weight / stronger pose metric term) under same Test500 contract.
- If two more loss-only rounds cannot push coop below ~1.6, escalate to supervision-density and tail-robustness changes.

### 2026-02-18 21:20 +0800
Scope: L4 low-LR recover branch closure on fixed Test500 contract + fail-safe hardening.
Key wins:
- Completed two parallel L4 runs (2x4 GPU) and finished full fixed-contract eval artifacts for both:
  - `eval_runs/geom_ablation_test500_20260218_l4a_recover_w0p2_vr1p_noconf_from_l2_lr1e6/summary_test.json`
  - `eval_runs/geom_ablation_test500_20260218_l4b_recover_w0p2_vr1p_noconf_from_l1c_lr1e6/summary_test.json`
- Quantified collapse severity objectively: both eval logs report exactly `1000` inference singular failures (`500 frames × single/coop`), leaving empty mode metrics.
- Hardened `run_geom_ablation_test500.sh` to avoid silent post-eval crashes and stale-state contamination:
  - quick-report now handles missing mode metrics;
  - default `RESUME=false` for pretrained-ckpt ablations;
  - default `MAX_BAD_LOSS_COUNT=50` for fail-fast on NaN storms.
Key risks:
- L4 branch is non-promotable (no valid single/coop aggregates under Test500).
- NaN storm starts late in epoch (iter ~960) and then dominates remaining train/test loops, so checkpoint-last becomes unusable.
Next experiments:
- Re-run safer branch with new fail-safe defaults (no optimizer resume) from stable pre-L4 checkpoints, then gate on Test500.
- Keep detach-heavy branch de-prioritized until non-collapse behavior is restored on short runs.

### 2026-02-18 14:40 +0800
Scope: Plan consolidation + fixed-contract scale tail diagnosis on the current geometry baseline.
Key wins:
- Added a single entry-point delivery plan: `docs/master_plan.md` (goal → status → gaps → executable next steps).
- Updated doc navigation to reflect both pinhole geometry/det and cylindrical routes: `docs/README.md`.
- Closed the missing diagnosis loop with an evidence-backed tail report on Test500:
  - `eval_runs/_diagnose_scale_tail_test500_20260218_sfix/report.md`
  - includes both official metrics (from `summary_test.json`) and per-frame tail listings.
- Updated acceptance plan status with the new diagnosis artifact (T9): `docs/goal_based_test_plan.md`.
Key risks:
- Baseline still under-scales systematically (`scale_to_gt_ratio_mean<1`) and coop tail contains joint pose+scale blow-ups (max pose_abs ~80m).
Next experiments:
- Implement and test "detach metric_scaling_factor from geometry losses when direct_scale_loss=True" to prevent scale drift.
- If median improves but tail remains, escalate to supervision density (dense depth v1) + coop-tail robustness (reweight/consistency).

### 2026-02-17 20:55 +0800
Scope: Unified metric + visualization re-validation for historical “good-looking” pinhole geometry ckpts.
Key wins:
- Completed fixed-contract (`test500`, hash `4affdce1133580fe79d7522c2892e359`) re-check on four candidate ckpts with OPV2V-correct scale metrics (`scale_to_gt_*`).
- Produced one-click visualization index across all audited ckpts:
  - `eval_runs/scale_viz_audit_test500_20260217/index.html`
  - per-ckpt offline html under `.../c*/html_scale/index.html`.
- Added consolidated report with quantitative + qualitative consistency analysis:
  - `docs/scale_metric_visual_audit_20260217.md`.
Key risks:
- All four ckpts still show systematic under-scale on average (`scale_to_gt_ratio_mean < 1`), especially coop tail frames.
- Coop worst cases remain coupled pose+scale failures (same hard sequence segment, `pose_abs_m≈80`).
Next experiments:
- Keep `c20260210_sfix_direct` as geometry start point (best `scale_to_gt_err_mean` under fixed contract).
- Prioritize reducing coop tail failures and pulling median `s/g` toward 1 without regressing pose/depth.

### 2026-02-16 17:35 +0800
Scope: Scale metric definition correction on OPV2V + test500 fairness re-ranking.
Key wins:
- Confirmed that legacy `scale_err_mean=mean(|s-1|)` is not valid as the primary quality metric for OPV2V (`norm_mode=avg_dis`), because GT scale is per-frame factor `g`, not constant 1.
- Added ratio-to-GT backfill pipeline:
  - `scripts/compute_opv2v_gt_scale_contract.py`
  - `scripts/backfill_scale_to_gt_contract.py`
  - upgraded `scripts/scoreboard_scan_geom_contract.py` to sort/report by `scale_to_gt_err_mean`.
- Recomputed test500 contract scale ranking under corrected metric:
  - best scale ckpt flips from `recover_wd` (legacy metric) to `sfix_direct` (ratio-to-GT metric), both single and coop.
Key risks:
- Many historical docs/log lines still mention legacy `scale_err_mean` thresholds (1–3 band), which can mislead unless explicitly marked as legacy.
- Training-time gate/log metrics still expose legacy ratio-to-1 keys by default.
Next experiments:
- Use `scale_to_gt_*` as the only pass/fail/ranking scale metric in new reports.
- Keep legacy `scale_err_mean` in outputs only for backward compatibility/audit.

### 2026-02-14 16:22 +0800
Scope: Fairness-first quantitative scoreboard refinement (same test50 frame set), plus metadata backfill for locked pinhole baselines.
Key wins:
- Backfilled `run_info.det_decode_cfg` and ratio-space scale diagnostics into locked baseline summaries (`det_e2e_v5_eval_t005`, `det_e2e_v5_coop_eval_t005`) without rerunning inference.
- Added reproducible scripts for fairness snapshots:
  - `scripts/scoreboard_snapshot.py` (grouped scoreboard + fairness checks),
  - `scripts/scoreboard_scan_test50_pinhole.py` (same-frames Top-K scan across pinhole runs),
  - `scripts/backfill_pinhole_test50_summaries.py` (artifact metadata/metric backfill).
- Updated clean summary doc with explicit same-frames Top-K conclusions: current det decode/e2e tuning did not beat locked baselines on the canonical test50 split.
Key risks:
- Backfilled `scale_log_err_mean/scale_ratio_mean` for historical summaries are derived from `scale_err` CSV and use a one-sided assumption when `scale_err>=1`; this is robust for current large-error runs but should be treated as derived metadata.
- Cylindrical coop test50 result remains train-on-test leakage; still not generalization evidence.
Next experiments:
- Build a leak-free cylindrical held-out benchmark and re-run det/e2e under identical decode config recording.
- Continue geometry-side fixes for scale supervision density; detection-side tuning is near ceiling under current geometry.

### 2026-02-12 23:16 +0800
Scope: Documentation consolidation for BEV head attempts + representative ckpts (pinhole & cylindrical).
Key wins:
- Workspace overview now documents BEV/det-head attempts with purpose/results/meaning, anchored to eval artifacts.
- Added representative ckpts for pinhole/cyl routes with interpretation (pose vs geometry bottlenecks).
- Deep index for experiments/eval_runs enables traceability from summaries to artifacts.
Key risks:
- Metrics still dispersed across multiple JSON/CSV files; manual cross-run comparison is time-consuming.
- Detection baselines unchanged; coop AP remains limited by pose/scale drift.
Next experiments:
- If needed, auto-extract key metrics (scale_err_mean/pose/depth/AP) into a compact table per run.

### 2026-02-12 10:35 +0800
Scope: 4x4090 sequential scale plan (S1 warm/full → S2 gating → S3 high scale weight), watchdog/loop monitoring.
Key wins:
- Completed S1 warm/full and S2/S3 passes at least once; checkpoints saved for each stage.
- Valid-ratio diagnostics visible in full-test logs (scale_valid_ratio_avg ~0.037).
Key risks:
- Full-test scale_err_mean worsened to ~18 (epoch1) vs baseline ~8.7; still far from target 1–3.
- Full-test epoch2 lines show NaN loss + scale_err_mean=0 (invalid metrics); runs must be treated as invalid.
- Scale supervision remains sparse (~3.7% valid), limiting effective scale learning.
Next experiments:
- Finish current S2 epoch2, advance S3 (w1.0) and re-check full-test metrics.
- If scale still >3, move to densified depth/pseudo-depth + calibration and/or scale-consistency loss.

### 2026-02-11 05:33 +0800
Scope: Stage2 geometry recovery (min_scale_valid_ratio gating + w0p2 direct scale) with dual A800 runs.
Key wins:
- S7a full run completed epoch0 + full validate test with ratio-space scale diagnostics logged.
- S7b re-run launched on second A800 with lower LR (1e-5) to avoid NaN instability.
Key risks:
- Ratio-space `scale_err_mean` remains high (~11.9) after S7a epoch0 full test.
- Potential stall after eval; S7b may still hit NaNs without further loss tuning.
Next experiments:
- Let S7a epoch1 finish + collect final metrics; monitor S7b for NaNs and full test.
- If scale stays >3, advance to Stage3 (log-scale loss, stricter valid-ratio gating, or reduce conf loss).

### 2026-02-09 06:58 +0800
Scope: Review-fix closure loop for watchdog e2e verification, eval backfill, and status synchronization.
Key wins:
- Verified one full watchdog path: gate PASS -> auto-eval trigger -> summary/HTML artifacts (`watchdog_t5b_smoke_20260208_2250_v5`).
- Backfilled key eval runs (`auto_queue_det_eval_smoke`, `auto_queue_det_eval_full`) with `run_info` and ratio-space scale diagnostics (`scale_log_err_mean`, `scale_eq_rel_err_mean`, `scale_ratio_*`).
- Eliminated a real watchdog parser bug (`auto_queue_watchdog.sh` heredoc regex) and revalidated runtime with a successful end-to-end cycle.
- TMP defaults are now workspace-first in core train/launch scripts, reducing `/tmp` exhaustion risk.
Key risks:
- Geometry gate on real training still fails current thresholds.
- Detection AP remains `0.0/0.0` on current checkpoint; T6 objective still not met.
Next experiments:
- Continue geometry recovery until real gate pass (not synthetic log smoke), then retrain det/e2e.
- Keep tracking both `scale_loss` and ratio-space scale diagnostics in every milestone eval.

### 2026-02-09 04:45 +0800
Scope: Scale semantics clarification loop (loss-space vs ratio-space) and document propagation.
Key wins:
- Unified interpretation: `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` is scale loss (target 0), not scale ratio (target 1).
- Goal-based test plan now requires dual-track scale reporting (`scale_loss` + `scale_err_mean`).
- Gate env comments are updated to prevent ratio/loss misinterpretation.
Key risks:
- Gate still thresholds loss-space scale only; run-to-run comparability is sensitive to loss config changes.
- Historical auto-eval outputs need backfill to expose the new ratio-space scale diagnostics.
Next experiments:
- Keep current gate for continuity, but always pair it with eval `scale_err_mean`.
- Re-run/backfill representative eval runs so `scale_log_err_mean` and `scale_ratio_*` are available.

### 2026-02-08 06:58 +0800
Scope: Documentation optimization loop for logbook + goal-based test plan.
Key wins:
- Unified baseline policy in test plan (locked baseline vs historical reference best).
- Strengthened acceptance criteria (especially T3/T5) with reproducibility and preflight rules.
- Added explicit status split for T5a manual smoke vs T5b watchdog end-to-end.
Key risks:
- Watchdog full auto gate->eval cycle still not verified in one closed-loop run.
- Eval summary still lacks explicit checkpoint metadata.
- /tmp remains full; runs are fragile without TMPDIR override.
Next experiments:
- Run one watchdog closed-loop cycle and capture evidence.
- Add checkpoint path metadata to eval summary artifacts.
- Keep training geometry until gate thresholds pass, then rerun T6.

### 2026-02-08 05:16 +0800
Scope: Execute T3–T6 tests (geometry smoke, gate, auto-eval, det AP).
Key wins:
- Geometry smoke run completed (1 epoch, checkpoints saved, no NaN/badloss).
- Auto-eval smoke + full det eval generated summary JSON and HTML outputs.
Key risks:
- Gate still fails (pose/scale_loss above thresholds).
- Det AP on current checkpoint is 0.0 (single/coop) for 50-sample eval.
- /tmp on root FS is full; tmp-based runs can fail unless TMPDIR is redirected.
Next experiments:
- Improve geometry to pass gate (pose_trans/rot + scale_loss threshold).
- Train det head/e2e before re-running T6 (current ckpt not det-trained).

### 2026-02-08 04:37 +0800
Scope: Review-fix loop iteration focused on index loader verification + test standards.
Key wins:
- Fixed OPV2V index loader to accept numpy-array agent lists and to fail fast on empty index results.
- Added goal-based test plan with acceptance criteria; ran index smoke tests (index hit + non-empty scenes).
Key risks:
- Geometry stability smoke, gate check, auto-eval, and det AP improvement remain unverified.
Next experiments:
- Run geometry stability smoke + gate check (T3/T4 in goal_based_test_plan).
- Run auto-eval smoke + det AP eval (T5/T6).

### 2026-03-06 16:20 +0800
Context: Standardize the local evaluation contract and GitHub Project V2 presentation for canonical geometry / detection evidence.
Symptom: Repo backup continuity existed, but the Project-facing evaluation workflow was still fragmented; single-only reference runs were also getting mislabeled as gate blockers.
Root Cause (best guess): The planning scope was too narrow, and `sync_benchmark_project_v2.py` assumed a geometry-style gate for every run.
Fix: Audited the live `coopVGGT` Project V2 schema; added `render_project_eval_overview.py` and `sync_project_eval_canonical.py`; added `--gate_status_override none` support for single-only reference runs; synced the curated canonical items into the remote Project.
Validation: `docs/project_v2_eval_overview.md` exists; `sync_project_eval_canonical.py --live` updated the canonical Project items for geometry baseline/promoted and detection single/coop evidence.
Next: Review remote item readability and decide the smallest validation experiment needed for the next uncertainty gap.

### 2026-02-08 01:47 +0800
Scope: Implement index build, dataset wiring, badloss cap, gate thresholds, and auto-eval hook.
Key wins:
- OPV2V global index built and dataset now prefers index over YAML scan.
- Badloss debug dumps capped; gate thresholds aligned to stage2+20%.
- Auto-eval hook wired into watchdog with det/e2e defaults.
Key risks:
- Wiring changes not yet exercised in a fresh run (index hit + auto-eval artifacts unverified).
- Gate thresholds may need tuning if too strict under new configs.
Next experiments:
- Run a short training start to confirm index hit log + no YAML scan.
- Trigger one watchdog cycle to confirm auto-eval `summary_test.json` output.

### 2026-02-07 23:23 +0800
Scope: Session audit, status snapshot, and cleanup after NaN/badloss runs.
Key wins:
- OPV2V coop scene cache is implemented and in use; scale-only direct run completed with checkpoints.
- Badloss debug artifacts cleaned (3.47 TB -> 0 files) from a single run directory.
Key risks:
- Global OPV2V index is only schema+script; not built or wired into dataset path yet.
- Badloss debug dump remains unbounded in code; NaN can re-fill disk quickly.
- Watchdog auto-fix can timeout without patch; gate/auto-eval not yet closed-loop.
Next experiments:
- Build OPV2V global index and integrate dataset loading path to prefer index.
- Add cap/sampling for badloss debug dump and revisit NaN root-cause.
- Define geometry gate and automate switch to det/e2e evaluation.

### 2026-02-03 08:10 +0800
Scope: Pinhole coop geometry + det training; focus on stabilizing pose/depth/scale.
Key wins:
- Identified a stronger geometry checkpoint (20260130_*_dist30_m32_cache) with pose 0.x on val/test.
- Found stage3 "superstrong" config regressing geometry vs stage2 best.
Key risks:
- Stage3 loss weights and squared relative pose can destabilize geometry.
- Dataset filtered to max_agent_distance=32 yields small train set (121 scenes), higher variance.
Next experiments:
- Recover geometry from 20260130 best with weaker pose/scale loss.
- Then train det head only; then joint fine-tune with low geometry LR.

---

## PROGRESS SNAPSHOT (updateable)

- Data cache: OPV2V index is built (`opv2v_index/index_{train,validate,test}.parquet`) and dataset prefers index before YAML scan.
- Training stability: badloss debug dumps are capped (`bad_loss_dump_max`, `bad_loss_dump_stride`); geometry smoke run writes checkpoints.
- Gate readiness (fixed Test500 hard gate): multiple PASS candidates exist.
  - best pass run: `20260220_l10a_w0p12_vr1p_noconf_from_l9a2` (`overall_pass=True`)
  - key metrics: single `mult=1.1151`, coop `mult=1.6359`, pose/depth guardrails both pass.
- Scale interpretability: primary ranking now uses `scale_to_gt_*` under fixed Test500 contract; latest promoted run still has coop gap vs final target (`1.6359` vs `<=1.3`).
- Detection status: locked baselines are single `0.19915220884189924` and coop `0.1380228069656401`; current candidate AP remains `0.0/0.0`.
- Test plan status: T1/T2/T3/T5a/T5b/T7/T8/T9 PASS; T10/T11/T12/T13 FAIL; T14/T15 PARTIAL PASS; T16 FAIL.
- Ops risk: `/tmp` remains full at host level, but train/launch scripts now default TMPDIR to workspace paths, reducing fragility.
- Next: reduce coop `scale_to_gt_mult_err_mean` from `1.6359` to `<=1.3` without breaking pose/depth guardrails, then re-open det/e2e and rerun T6.

---

## GOALS & GAPS (current)

Goals:
- Stable geometry training (no NaN loops) with reproducible checkpoints.
- Automated gate to switch from geometry -> det/e2e evaluation.
- Det AP exceeds locked baselines under a consistent eval protocol.

Baseline policy:
- Locked baseline for pass/fail: single `0.19915220884189924` (`metrics.e2e_v5.single.det_ap_iou`) and coop `0.1380228069656401` (`metrics.e2e_v5.coop.det_ap_iou`).
- Historical reference best (context only): single `0.25` from `stage2_det_v5_2k_eval_000386`.

Gaps:
- Gate still fails on latest geometry smoke run (legacy loss-space); ratio-space gate pending rerun.
- Ratio-space scale error is still high (`scale_err_mean` in latest eval summary is far from 0).
- Det AP improvement is still unmet (current candidate AP remains 0.0/0.0).
- Watchdog has a validated smoke loop, but production reliability still needs confirmation on real training logs.

---

## TODO BACKLOG (updateable)

- [x] Build OPV2V global index (requires pandas).
  - Output: `/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/data/opv2v_images/opv2v_index/index_{train,validate,test}.parquet` + `manifest.json` + `schema.json`.
- [x] Wire dataset to prefer index before YAML scan.
  - Code: `map-anything/mapanything/datasets/opv2v.py`; config: `configs/dataset/opv2v/default.yaml` (`index_dir`).
- [x] Cap badloss debug dumps (max N per run) + optional sampling.
  - Code: `map-anything/mapanything/train/training.py` (`bad_loss_dump_max`, `bad_loss_dump_stride`).
- [x] Define geometry gate metrics for auto switch to det.
  - Env: `auto_queue_pinhole_pose_depth_scale_v3.txt.gate.env` + `auto_queue_pinhole_geom_det_e2e_v1.txt.gate.env`.
- [x] Add auto-eval step for det/e2e.
  - Code: `bash_scripts/monitor/auto_queue_watchdog.sh` with auto-eval defaults.
- [x] Create/maintain goal-based test plan with acceptance criteria.
  - Doc: `docs/goal_based_test_plan.md`.
- [x] Align scale metric semantics across plans and gate docs.
  - Clarified that gate scale key is loss-space (`...GMLoss_scale`, target 0), and eval `scale_err_mean` is ratio-space (target 0).
- [x] Verify one watchdog closed-loop run (gate PASS -> auto-eval trigger -> artifacts).
  - Evidence: `experiments/local_runs/watchdog_t5b_smoke_20260208_2250_v5/logs/auto_queue_0_20260208_224950.log`, `auto_queue.state=1`, and `eval_runs/watchdog_t5b_smoke_eval/auto_queue_0_20260208_224950/summary_test.json`.
- [x] Add checkpoint path metadata into eval summary output (or mirrored run-registry entry).
  - Code: `bash_scripts/utils/opv2v_det_eval_and_html.sh` now patches `run_info` into `summary_test.json`.
- [x] Eliminate `/tmp` dependency in run scripts (use workspace TMPDIR by default).
  - Code: core train/launch scripts now export workspace-first `TMPDIR`/`TORCHELASTIC_TMPDIR`/`TMP`/`TEMP`.
- [x] Add ratio-space scale diagnostics to eval summary (`scale_log_err_mean`, `scale_eq_rel_err_mean`, `scale_ratio_*`).
  - Code: `scripts/batch_eval.py` and `scripts/recompute_pc_metrics.py`.
- [x] Backfill/re-run key eval summaries so new scale diagnostics are present in comparable artifacts.
  - Evidence: `eval_runs/auto_queue_det_eval_smoke/summary_test.json` and `eval_runs/auto_queue_det_eval_full/summary_test.json` regenerated with `scale_log_err_mean`, `scale_eq_rel_err_mean`, `scale_ratio_*`, and `run_info`.

---

## REVIEW FIX LOOP (current)

Goals & Success Criteria:
- Documentation and status tracking are consistent, reproducible, and unambiguous.
- Every open gap maps to one executable test or action item.
- Baseline and pass/fail policy are stable across logbook and test plan.

Evidence Reviewed:
- `map-anything/docs/experiment_logbook.md`
- `map-anything/docs/goal_based_test_plan.md`
- `map-anything/scripts/batch_eval.py`
- `map-anything/summary_smoke/20260207_smoke_geom_gate/summary.json`
- `map-anything/eval_runs/auto_queue_det_eval_smoke/summary_test.json`
- `map-anything/eval_runs/auto_queue_det_eval_full/summary_test.json`
- `map-anything/eval_runs/det_e2e_v5_eval_t005/summary_test.json`
- `map-anything/eval_runs/det_e2e_v5_coop_eval_t005/summary_test.json`
- `map-anything/eval_runs/stage2_det_v5_2k_eval_000386/summary_test.json`
- `map-anything/experiments/long_runs/auto_queue_pinhole_pose_depth_scale_v3.txt.gate.env`

Findings (severity order):
- Fixed: scale metric semantics are now explicit across plans (`scale_loss` is optimization-space, `scale_err_mean` is interpretation-space).
- Fixed: baseline ambiguity resolved by introducing locked-baseline policy and historical reference split.
- Fixed: T3 acceptance no longer depends on noisy `grep nan|inf`; now based on finite structured metrics + artifacts.
- Fixed: T5 is split into T5a (manual smoke) and T5b (watchdog e2e) to avoid false closure.
- Fixed: reproducibility requirements and preflight checks are now explicit in the test plan.
- Fixed: key historical eval summaries were backfilled and now include ratio-space scale diagnostics.
- Fixed: watchdog closed-loop automation was verified in a dedicated smoke run.
- Fixed: eval summaries now include explicit checkpoint metadata via `run_info`.
- Fixed: run scripts now default temp paths to workspace, reducing `/tmp` dependency.
- Remaining: geometry gate (ratio-space) and AP objective are still unmet and require model-side improvements.

Fix Plan:
- Keep docs synced to one baseline policy and one status vocabulary.
- Close remaining engineering gaps with targeted execution evidence (watchdog run, metadata patch, TMPDIR default).

Changes Made:
- Rewrote `goal_based_test_plan.md` for baseline clarity, stricter acceptance criteria, and reproducibility metadata.
- Updated logbook progress/goals/todo sections to align with latest pass/fail evidence.
- Added a scale semantics policy and dual-track reporting rule (`scale_loss` + `scale_err_mean`) in planning docs.
- Added fresh macro+micro entries documenting this documentation optimization loop.

Verification:
- Checked index artifacts exist and row counts are non-zero.
- Re-ran index usage smoke; observed `OPV2V index hit` and `filtered scenes=2`.
- Re-ran gate check; confirmed FAIL with metrics over thresholds.
- Confirmed auto-eval smoke/full artifacts exist.
- Confirmed `run_info` metadata exists in regenerated smoke/full summaries.
- Confirmed new scale diagnostics (`scale_log_err_mean`, `scale_eq_rel_err_mean`, `scale_ratio_*`) exist in regenerated smoke/full summaries.
- Verified watchdog smoke run reached gate PASS and produced auto-eval artifacts with `auto_queue.state=1`.
- Confirmed T6 candidate AP values are still zero in full eval summary.

Remaining Gaps / Next Iteration:
- Documentation/tooling closure is complete for this loop.
- Continue loop on model-quality gaps (real gate pass and AP improvement beyond locked baselines).

---

## MICRO TIMELINE (latest-first; backfill allowed)

### 2026-03-01 14:00 +0800
Context: 在“跨车外参未知 + baseline distance heavy-tail”前提下，为保证 3D foundation model（MapAnything / VGGT）评测**严格公平可比**，对主线 ckpt 做一次固定合同（nearest + stress）+ baseline-distance bucket breakdown 的全量复评，并产出可横向对比的汇总报告。
Symptom:
- 需要把“现有主要有效线”的多个 ckpt 在同一套 bucket-suite 合同下跑全并做详尽对比；同时发现 VGGT 的 per-bucket scale 表格出现 NA（无法用于 bucket 对比）。
Root Cause (best guess):
- `report_geom_by_baseline_buckets.py` 早期实现依赖 `scale_ratio_mean`（预测 `metric_scaling_factor`）重算 `scale_to_gt_*`；但 VGGT 输出缺失该字段（CSV 为 nan），导致 per-bucket scale 统计为 NA。
- `compare_geom_bucket_suite.py` 初版将 `logs/` 误识别为模型目录，导致自动 discover contracts 失败。
Fix:
- `scripts/report_geom_by_baseline_buckets.py`：改为直接聚合 `*_metrics.csv` 里 per-frame 的 `scale_to_gt_ratio_mean / scale_to_gt_log_err / scale_to_gt_err`（与 `summary_test.json` 口径一致），从而支持 VGGT 的 bucket 对比。
- `scripts/compare_geom_bucket_suite.py`：只把“包含 contract 子目录且有 summary_test.json 的目录”识别为模型目录，跳过 `logs/`。
- 复用 `SKIP_BATCH_EVAL_IF_EXISTS=1` 对所有已跑出的 outputs 快速再生成 bucket 报告与 compare 汇总。
Validation:
- 主线 bucket-suite outputs：`eval_runs/geom_bucket_suite_mainlines_20260301/`
- 汇总对比（overall + per-bucket）：`eval_runs/geom_bucket_suite_mainlines_20260301/compare.md`
  - 合同一致性：两份合同下所有模型 `frames_hash_md5_v2` 均一致（compare.md 内有显式列）。
  - VGGT per-bucket scale 表格已不再是 NA（compare.md 已覆盖）。
Next:
- 以 `compare.md` 为准锁定“代表性合同 + stress 合同”下的 geometry baseline/guardrails（尤其关注 tail>=100m 桶），再把 det/e2e 的 bucket 化评测接到同一套合同体系里做公平对照。

### 2026-02-25 14:05 +0800
Context: Plan1 距离桶诊断显示 mid(30–60m) 可把 `cross_agent_pose_trans_mean` 压到 ~6m，但会把 scale `mult` 推到 >1.3；需要验证是否可以通过更稳的 loss（detach geom-scale）在 mid 桶上同时拿到 pose 改善 + scale 过线。
Symptom:
- 现有最强 pose 信号（mid 桶）对应的 run 仍因 `mult>1.3` 不可 promotion；需要一个“只改 loss”的对照实验来判断 scale 是否可控。
Root Cause (best guess):
- mid 桶子分布下 pose/scale 强耦合；原配置可能通过“scale 偏移”来降低跨车平移误差，导致 `mult` 越线。
Fix:
- 启动 in-flight 实验：保持 mid(30–60m) 数据过滤不变，仅改 loss 为 detach geom-scale（w0.4 pose12），从 promoted nm4 起点短训 1 epoch，并回到 fixed Test500 合同评测。
  - run_tag：`20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1`
  - train log：`experiments/local_runs/20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1/pinhole_pose_depth_scale/train.log`
  - expected eval：`eval_runs/geom_ablation_test500_20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1/quick_report.txt`
Validation:
- 训练与 Test500 合同评测已完成：
  - `eval_runs/geom_ablation_test500_20260225_mid_30_60_detach_w0p4_pose12_from_nm4_e1_r1/quick_report.txt`
  - coop：`mult=1.3028`（>1.3 ❌；仅轻微越线），`ratio=1.0849`，`cross_trans=5.9934m`，`cross_rot=1.433°`，`depth_rel=0.3270`
  - 结论：mid 桶的 “pose push” 信号非常强，但 scale 仍需要一个 recover 步骤把 `mult` 拉回 <=1.3。
Next:
- 进入两阶段闭环（同一 Test500 合同评测）：
  1) pose push：保留该 ckpt 作为候选起点（当前 best `cross_trans~6m`）
  2) scale recover：从该 ckpt 出发，设计“尽量不伤 pose 的 scale 校准”实验，把 `mult` 拉回 <=1.3（例如：只训 scale 分支 / 更偏重 scale 的短训 / 数据分布回混）

### 2026-02-25 12:50 +0800
Context: 用户反馈“方案一似乎跑了两次”，需要把两批 Plan1（距离桶）实验的设置差异与结果差异讲清楚，并收敛 20260224 Plan1 pipeline 在非 8 卡新机上的实际落盘结果（避免入口退出导致机器回收、队列被杀）。
Symptom:
- `experiments/long_runs/20260224_193935_plan1_pipeline_8x4090__pipeline/status.tsv` 显示同一批 job（near/far/densify_r1/detach_v1 + mid/tail/densify_r2/detach_rpose2）被记录了两轮执行；且 lane1 两轮都快速失败。
- lane1 失败表现为 `CUDA error: invalid device ordinal`（GPU 列表包含不存在的 6/7 号卡）。
- Plan1 的 lane1（mid/tail/densify_r2/detach_rpose2）在主 pipeline 内未产出有效结果，需要补跑；同时入口命令退出会触发开发机回收，必须确保“所有补跑任务完成后再退出”。
Root Cause (best guess):
- pipeline runner 以 8 卡假设硬编码 lane1 `CUDA_VISIBLE_DEVICES=4,5,6,7`，但宿主机实际只暴露 `0..5`，导致 lane1 直接报错退出。
- 平台 entrypoint 侧存在“同一条入口命令被重复拉起/重复记录”的现象（同一 `PIPELINE_ID` 两轮写入同一 `pipeline.log/status.tsv`），导致 run_tag 复用并覆盖了第一次 lane0 的 eval artifacts（以文件 mtime 可验证）。
Fix:
- 补跑 lane1（在不改主 pipeline ID 的前提下）：
  - rerun 队列：`experiments/long_runs/20260224_210949_lane1_gpu45__rerun_lane1_gpu45/run.sh`（GPU_LIST=4,5）顺序补齐 mid/tail/densify_r2。
  - 将剩余 detach_rpose2 调度到空闲 0-3 卡并写回同一 `status.tsv`：`experiments/long_runs/20260224_210949_lane1_gpu45__rerun_lane1_gpu45/dispatch_detach_gpu0123.sh`（同时在 rerun 队列启动其自带 detach 时 kill 掉，避免重复占用）。
- 为防止入口脚本提前退出导致补跑任务被平台回收：在 `scripts/scoreboard_scan_geom_contract.py` 增加可选 hold hook（读取 `.../__pipeline/hold_until.json`，等待外部队列 `status.tsv` 行数满足期望后再继续）。
Validation (fixed Test500 / coop-first / calibrated_sfm / keep_camera_poses=0):
- Plan1 pipeline 最终 scoreboard（包含 5 个 matched rows）：`experiments/long_runs/20260224_193935_plan1_pipeline_8x4090__pipeline/artifacts/scoreboard_20260224_193935_plan1_pipeline_8x4090.md`
- 新 Plan1（以 promoted nm4 作为 init，loss=scaleonly_w0.2 为主）关键结果：
  - near：`eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_plan1_near_0_30m/quick_report.txt`（PASS；mult=1.2798）
  - far：`eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_plan1_far_60_100m/quick_report.txt`（FAIL；depth_rel=0.3984 越线，但 mult=1.2787 是 ok 的）
  - densify_r1：`eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_l2_densify_r1_full/quick_report.txt`（PASS；mult=1.2933）
  - detach_v1：`eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_l1_detach_geom_scale_w0p4_pose12/quick_report.txt`（PASS；mult=1.2715）
  - detach_rpose2（补跑）：`eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_l1_detach_geom_scale_w0p4_pose12_rpose2__rerun_gpu0123/quick_report.txt`（PASS；mult=1.2788）
- lane1 rerun（同 init/loss，只是补齐 mid/tail/densify_r2）：
  - mid：`eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_plan1_mid_30_60m/quick_report.txt`（FAIL；mult=1.3242，但 cross_trans=6.1614m 有明显改善信号）
  - tail：`eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_plan1_tail_100_inf/quick_report.txt`（FAIL；mult=1.4535 且 depth/cross 也退化）
  - densify_r2：`eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_l2_densify_r2/quick_report.txt`（FAIL；mult=1.3046，接近阈值）
- 与旧 Plan1 bucket-only（从 l10a 起点，loss w0.12）对比：在当前 gate 下 near/mid/far 都明显更差（mult 约 1.42~1.50）：
  - near：`eval_runs/geom_ablation_test500_20260222_plan1_near_sgpu_r1/quick_report.txt`（mult=1.4667）
  - mid：`eval_runs/geom_ablation_test500_20260222_plan1_mid_sgpu_r1/quick_report.txt`（mult=1.4958）
  - far：`eval_runs/geom_ablation_test500_20260222_plan1_far_sgpu_r1/quick_report.txt`（mult=1.4215）
Next:
- 把 Plan1 pipeline runner 做成“GPU 数自适应/显式 lane GPU 参数化 + run_tag 不复用”的稳定入口，避免再次出现 lane1 无效 GPU 列表与 artifacts 覆盖。
- 算法侧：tail 仍需要 mixture/curriculum/tail reweight（bucket-only tail finetune 会明显崩）；mid 的 pose 改善需要一个专门的 scale recover 阶段把 mult 拉回 <=1.3。

### 2026-02-24 23:00 +0000
Context: 需要确认“2/24 启动的新策略（Plan1 距离桶 + follow-ups）”与既定主计划（固定 Test500 合同 + coop-first + calibrated_sfm 评测）是否一致，并把 L2 densify 的 in-flight 状态收敛为最终结论。
Symptom:
- L2 densify r=1/2 的 quick_report 已落盘，但计划文档仍标记为 IN-FLIGHT。
- Plan1 pipeline 的 lane1 在启动后快速失败（mid/tail/densify_r2/detach_rpose2 均未产出 summary/quick_report），导致“只跑了一半”的错觉。
Root Cause (best guess):
- lane1 失败的根因是 **GPU 列表与宿主机可见 GPU 不匹配**：宿主机仅暴露 `0..5`，但 lane1 使用了 `CUDA_VISIBLE_DEVICES=4,5,6,7`，触发 `CUDA error: invalid device ordinal`。
- L2 densify v1 的根因更偏“监督噪声/错误传播”：最近邻填充虽然提高有效像素，但把错误深度扩散进 supervision，导致 scale 退化。
Fix:
- 将 L2 densify r=1/2 状态更新为 FAIL（不 promotion）。
- 对 Plan1 的 lane1 做针对性 rerun（GPU_LIST=4,5），确保 mid/tail 能完整 train->eval 落盘；并更新 pipeline runner，避免再次生成无效 GPU 列表。
Validation (fixed Test500 / coop-only / `model_task=calibrated_sfm` / keep_camera_poses=0):
- L2 densify（结论：FAIL）：
  - r=1: `eval_runs/geom_ablation_test500_20260223_l2_densify_r1_from_nm4_w02_e1/quick_report.txt`（coop `mult=1.3195`）
  - r=2: `eval_runs/geom_ablation_test500_20260223_l2_densify_r2_from_nm4_w02_e1/quick_report.txt`（coop `mult=1.3332`）
- Plan1（诊断信号）：
  - mid 30–60m（rerun）：`eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_plan1_mid_30_60m/quick_report.txt`
    - coop: `cross_trans=6.1614m`（显著改善）但 `mult=1.3242`（越线）
  - tail 100–inf（rerun）：`eval_runs/geom_ablation_test500_20260224_210949_lane1_gpu45_plan1_tail_100_inf/quick_report.txt`（多指标崩）
  - detach follow-up：`eval_runs/geom_ablation_test500_20260224_193935_plan1_pipeline_8x4090_l1_detach_geom_scale_w0p4_pose12/quick_report.txt`（总体 pass）
Next:
- 用 Plan1 的证据更新主线策略：把“cross-agent pose”作为第一目标，但必须设计后续阶段把 scale `mult` 拉回 <=1.3（否则不可部署）。
- 等 `20260224_210949_lane1_gpu45_l2_densify_r2` / `..._l1_detach_rpose2` 产出 Test500 quick_report 后，决定是否保留 densify/pose-sweep 分支。

### 2026-02-24 06:12 +0800
Context: L1c（detach geom-scale + 强化 cross-agent pose 监督）两路并行已完成 Test500 合同评测；目标是压 `cross_agent_pose_trans_mean` 同时保持 `mult<=1.3`。若失败则按 Fix Ladder 进入 L2（supervision 密度增强）。
Symptom:
- 两路都未带来 cross-agent pose 改善（`cross_trans` 变差），且更强 rpose2 触发 `mult<=1.3` gate 失败。
Root Cause (best guess):
- 当前 cross-agent pose 的主要瓶颈不在“相对位姿 loss 权重”，更可能在监督密度/噪声（稀疏 depth）或 tail 稳定性（需要升级到 L2/L3）。
Fix:
- 不 promotion L1c；回退到 promoted baseline 作为起点。
- 进入 L2：对稀疏深度做“安全 densify v1”（半径最近邻填充），只在训练集启用，val/test 保持不变。
- 工程实现（默认关闭）：
  - `mapanything/utils/depth_densify.py`：`densify_sparse_depthmap_nearest(depthmap, radius)`
  - `datasets/base/base_dataset.py`：新增 `sparse_depth_densify_radius`（pointmap 生成前 densify `depthmap`）
  - `configs/dataset/opv2v_coop_ft_2a8v_full.yaml`：`dataset.sparse_depth_densify_radius`（仅传给 train_dataset）
Validation (Test500 / coop-only / `model_task=calibrated_sfm` / keep_camera_poses=0):
- L1c rpose1：
  - `eval_runs/geom_ablation_test500_20260223_l1_detach_w0p4_pose12_rpose1_from_nm4_e1/quick_report.txt`
  - coop：`mult=1.2980`, `cross_trans=10.8069m`, `depth_rel=0.3210`
- L1c rpose2：
  - `eval_runs/geom_ablation_test500_20260223_l1_detach_w0p4_pose12_rpose2_from_nm4_e1/quick_report.txt`
  - coop：`mult=1.3027`（fail）, `cross_trans=10.7991m`, `depth_rel=0.3226`
Runs (L2; 2×4GPU 并行; EPOCHS=1; LR=5e-6; init=promoted nm4 ckpt; loss=scaleonly_w0.2):
- densify r=1：`20260223_l2_densify_r1_from_nm4_w02_e1`
- densify r=2：`20260223_l2_densify_r2_from_nm4_w02_e1`
Infra note:
- L2 已在 tmux session `ma_l2_densify` 后台执行（0-3 / 4-7）。
Next:
- 等两路产出 `eval_runs/geom_ablation_test500_20260223_l2_densify_r*/quick_report.txt`：
  - 以 `mult<=1.3` 为硬门槛；
  - 观察 `cross_trans` 是否出现 >=10% 改善；若仍无改善，进入 L3（tail 稳定化 / reweight / consistency）或重新审视 pose 指标定义与监督口径。

### 2026-02-24 05:10 +0800
Context: L1b（训练期 `model/task=calibrated_sfm` vs `images_only`）两路并行已完成 Test500 合同评测，需要决定后续训练口径，并继续推进 L1（压 cross-agent pose，保持 `mult<=1.3`）。
Symptom:
- 预期：训练期喂 ray dirs（calibrated_sfm）能显著降低 `cross_agent_pose_*`。
- 观测：两路都保持 `mult<=1.3`，但 calibrated_sfm 并未改善 cross-agent pose（`cross_trans` 略变差）。
Root Cause (best guess):
- “训练期是否喂 ray dirs”不是当前 cross-agent pose 的主要瓶颈；更可能受限于监督密度/噪声或 loss 结构（需要按 Fix Ladder 升级策略）。
Fix:
- 后续训练默认回到 `model/task=images_only`（只在评测期使用部署口径 `model_task=calibrated_sfm`）。
- 启动 L1c 两路并行：detach geom-scale + 强化 cross-agent pose 监督（只改 loss 核心因素）。
Validation (Test500 / coop-only / `model_task=calibrated_sfm` / keep_camera_poses=0):
- images_only：
  - `eval_runs/geom_ablation_test500_20260223_l1_task_imagesonly_from_nm4_w02_d0_60_lr5e6_e1_r1/quick_report.txt`
  - coop：`mult=1.2656`, `cross_trans=10.4535m`, `cross_rot=2.562°`, `depth_rel=0.3507`
- calibrated_sfm：
  - `eval_runs/geom_ablation_test500_20260223_l1_task_calibrated_sfm_from_nm4_w02_d0_60_lr5e6_e1_r1/quick_report.txt`
  - coop：`mult=1.2700`, `cross_trans=10.6196m`, `cross_rot=2.719°`, `depth_rel=0.3590`
Runs (L1c; 2×4GPU 并行; EPOCHS=1; LR=5e-6; init=promoted nm4 ckpt):
- rpose1：`experiments/local_runs/20260223_l1_detach_w0p4_pose12_rpose1_from_nm4_e1/pinhole_pose_depth_scale/train.log`
- rpose2：`experiments/local_runs/20260223_l1_detach_w0p4_pose12_rpose2_from_nm4_e1/pinhole_pose_depth_scale/train.log`
Infra note:
- L1c 已在 tmux session `ma_l1_pose_sweep` 后台执行（避免 CLI tool-call 回收子进程）。
Next:
- 等 L1c 两路产出 `eval_runs/geom_ablation_test500_20260223_l1_detach_w0p4_pose12_*/quick_report.txt`：
  - 若出现 `cross_trans` 明显下降（>=10%）且 `mult<=1.3`，进入长训；
  - 否则升级到 L2（supervision 密度增强 / 稳定 tail）。

### 2026-02-24 04:26 +0800
Context: L1（加 RelativePoseMetricLoss）未能改善 cross-agent pose，按主计划升级策略：验证训练期启用 `model/task=calibrated_sfm`（ray dirs）是否能带来跨车 pose 改善。为隔离“多训 1 epoch 本身”的影响，做严格对照两路并行（同 init ckpt / 同 loss / 同数据过滤 / 同 LR，仅改 task）。
Runs (2×4GPU 并行; EPOCHS=1; LR=5e-6; train data filter: `coop_max_agent_distance=60m`):
- ctrl-run（images_only）：
  - train：`experiments/local_runs/20260223_l1_task_imagesonly_from_nm4_w02_d0_60_lr5e6_e1_r1/pinhole_pose_depth_scale/train.log`
- task-run（calibrated_sfm）：
  - train：`experiments/local_runs/20260223_l1_task_calibrated_sfm_from_nm4_w02_d0_60_lr5e6_e1_r1/pinhole_pose_depth_scale/train.log`
Infra note:
- 由于 CLI tool-call 可能清理后台子进程，长跑任务统一放到 tmux session `ma_l1_task` 内执行，避免“nohup 起了但很快被回收”的假启动。
Next:
- 两路训练完成后自动跑 Test500 合同 eval 并生成 `quick_report.txt`；对比 `cross_agent_pose_trans_mean` 与 `mult<=1.3` 决定是否 promotion 进入长训。

### 2026-02-24 04:17 +0800
Context: L1 “coop pose-focused” 两路 finetune（ctrl vs rpose2）已完成同合同 Test500（部署口径）评测，需要决定是否 promotion/继续长训，或升级策略。
Results (Test500 / coop-only / `model_task=calibrated_sfm` / 不喂跨车 GT pose):
- baseline（promoted 起点）：
  - `eval_runs/geom_recheck_test500_20260223_nm4_scaleonly_w02_calibrated/summary_test.json`
  - coop：`mult=1.2693`, `ratio=0.9493`, `cross_trans=10.5028m`, `depth_rel=0.3468`
- ctrl（RelativePoseMetricLoss=1.0）：
  - `eval_runs/geom_ablation_test500_20260223_l1_posefocus_ctrl_from_nm4_v4_cachefix/quick_report.txt`
  - coop：`mult=1.2942`（<=1.3 ✅ 但更差），`ratio=0.9168`（更 under-scale），`cross_trans=10.7229m`（更差），`depth_rel=0.3219`（更好）
- rpose2（2.0× RelativePoseMetricLoss）：
  - `eval_runs/geom_ablation_test500_20260223_l1_posefocus_rpose2_from_nm4_v4_cachefix/quick_report.txt`
  - coop：`mult=1.3037`（>1.3 ❌），`cross_trans=10.7572m`（更差），`depth_rel=0.3237`（更好）
Decision:
- 不 promotion L1 两路；继续以 baseline promoted ckpt 作为主线起点。
- 结论更像 “depth 变好但 pose/scale 没收益/退化”，说明单纯加 RelativePoseMetricLoss（在当前训练 task=images_only）不足以压 `cross_agent_pose_*`。
Next (upgrade strategy; one-factor change):
- 验证训练期启用 `model/task=calibrated_sfm`（ray dirs）是否能降低跨车 pose（当前训练默认 `images_only`，但评测口径已迁移到 `calibrated_sfm`）。
- 为隔离“多训 1 epoch 本身”的影响，做严格对照：同 init ckpt / 同 loss / 同数据过滤 / 同 LR，仅改 task。

### 2026-02-24 04:01 +0800
Context: L1 “coop pose-focused” 两路 finetune（ctrl vs rpose2）训练已完成，正在按同一固定 Test500 合同跑部署口径评测以选择下一轮起点（coop-first）。
Status:
- train 已完成（checkpoint 已落盘）：
  - ctrl：`experiments/local_runs/20260223_l1_posefocus_ctrl_from_nm4_v4_cachefix/pinhole_pose_depth_scale/checkpoint-best.pth`
  - rpose2：`experiments/local_runs/20260223_l1_posefocus_rpose2_from_nm4_v4_cachefix/pinhole_pose_depth_scale/checkpoint-best.pth`
- Test500 合同评测进行中（coop-only；`model_task=calibrated_sfm`；不喂跨车 GT pose）：
  - ctrl：`eval_runs/geom_ablation_test500_20260223_l1_posefocus_ctrl_from_nm4_v4_cachefix/eval.log`
  - rpose2：`eval_runs/geom_ablation_test500_20260223_l1_posefocus_rpose2_from_nm4_v4_cachefix/eval.log`
  - 进度快照（2026-02-24 04:00 +0800）：两路均已处理约 373/500 帧（按 eval.log 中 “Stripped external pose inputs” 行数估算）。
Next:
- 等两路 eval 落盘后读取并对比（同合同 / 同协议）：
  - `.../summary_test.json` + `.../quick_report.txt`
- 依据：保持 `scale_to_gt_mult_err_mean<=1.3`，并优先选择 `cross_agent_pose_trans_mean` 改善更明显的一路作为下一轮起点；若改善 <10%，升级策略（仍“一次只改一个核心因素”）。

### 2026-02-24 03:06 +0800
Context: 需要在 promoted ckpt（coop scale 已过线）基础上做 L1 “coop pose-focused” finetune（2×4GPU 并行），但多卡训练出现启动期卡死，导致算力空转、计划无法执行。
Symptom:
- 2×4GPU 训练启动后，rank0 会打印到 “Creating criterion / before DDP wrap”，随后不再进入训练循环；GPU 仅有少量显存占用但 util 很低。
- 强制打点后发现：rank0 能完成 train dataloader 构建，但 rank!=0 会卡在 dataloader 构建阶段（训练入口处“看起来像 DDP 卡死”，实则是 dataset init 卡死/超时等待）。
Root Cause (best guess):
- `OPV2VCoopDataset._load_data()` 在启用距离过滤（`max_agent_distance/min_agent_distance`）时，会启用 scene cache 协议：
  - rank!=0 先等待 `*_scene_cache.json`（默认等待 1800s）
  - rank0 如果走 parquet index 命中分支，会直接 return（未写 scene cache）
  - 结果：rank!=0 永远等不到 cache，训练在 dataloader 构建阶段“假死”。
Fix:
- 修复数据集分布式启动死锁：
  - `map-anything/mapanything/datasets/opv2v.py`：当 parquet index 命中且 `cache_file!=None` 时，由 rank0 写 scene cache（并按需 sync fast/primary），确保 rank!=0 立即可读。
- 同步减少启动期日志/卡顿：
  - `map-anything/mapanything/utils/train_tools.py`：`get_parameter_groups()` 不再 dump 全量参数名列表（只打印每组 num_params/numel + examples）。
- 追加定位用的跨 rank 强制打点（后续稳定后可移除）：
  - `map-anything/mapanything/train/training.py`：关键阶段 `force=True` 打点（init_distributed/dataloader/model/DDP/optimizer）。
Validation:
- 复现 -> 修复 -> 验证链路已跑通：多卡能够完成 dataloader 构建、DDP wrap、进入训练迭代（日志出现 `Epoch: [0] [50/971] ...`）。
- 已重启并进入稳定训练（1 epoch + Test500 eval）：
  - ctrl：`experiments/local_runs/20260223_l1_posefocus_ctrl_from_nm4_v4_cachefix/pinhole_pose_depth_scale/train.log`
  - rpose2：`experiments/local_runs/20260223_l1_posefocus_rpose2_from_nm4_v4_cachefix/pinhole_pose_depth_scale/train.log`
Next:
- 等两路完成后，读取 fixed Test500 合同评测：
  - `eval_runs/geom_ablation_test500_20260223_l1_posefocus_ctrl_from_nm4_v4_cachefix/quick_report.txt`
  - `eval_runs/geom_ablation_test500_20260223_l1_posefocus_rpose2_from_nm4_v4_cachefix/quick_report.txt`
- 选择保持 `scale_to_gt_mult_err_mean<=1.3` 且显著改善 `cross_agent_pose_trans_mean` 的配置作为下一轮起点；若两路均无明显改善（<10%），升级到下一层策略（仍遵守“一次只改一个核心因素”）。

### 2026-02-24 01:38 +0800
Context: 已完成部署口径 promotion（coop `scale_to_gt_mult_err_mean<=1.3`）；用户再次明确部署约束：单车内 4 相机 rig 外参已知、跨车 pose 未知，需 coop-first 学跨车相对位姿 + depth + metric scale。
Symptom:
- promoted ckpt 下 cross-agent translation error 仍为 10m 级别（Test500/coop mean：`cross_agent_pose_trans_mean≈10.50m`），协同融合/BEV det 仍被硬卡住。
Root Cause (best guess):
- cross-agent 相对位姿监督力度不足（scale 修正后仍未能把跨车平移误差显著压低）。
Fix:
- 以 promoted ckpt 为 init，在同一训练/数据设置下做 2×4GPU 并行对照，只改“cross-agent pose 直接监督强度”（RelativePoseMetricLoss）：
  - ctrl：`20260223_l1_posefocus_ctrl_from_nm4`（loss=`opv2v_vggt_pose_scale_loss_recover_direct_detach_geom_scale_w0p4_pose12_vr1p_noconf`）
  - rpose2：`20260223_l1_posefocus_rpose2_from_nm4`（loss=`..._noconf_rpose2`，2.0× RelativePoseMetricLoss）
  - init ckpt：`experiments/local_runs/20260223_nm4_scaleonly_w02_lr5e6_e1_fix/pinhole_pose_depth_scale/checkpoint-best.pth`
  - driver：`scripts/run_geom_ablation_test500.sh`（EPOCHS=1, LR=5e-6；完成后自动跑 Test500 合同评测：`model_task=calibrated_sfm`, `modes=coop`）
Validation:
- In-flight train logs：
  - `experiments/local_runs/20260223_l1_posefocus_ctrl_from_nm4/pinhole_pose_depth_scale/train.log`
  - `experiments/local_runs/20260223_l1_posefocus_rpose2_from_nm4/pinhole_pose_depth_scale/train.log`
- 预期评测证据（完成后生成）：
  - `eval_runs/geom_ablation_test500_20260223_l1_posefocus_ctrl_from_nm4/quick_report.txt`
  - `eval_runs/geom_ablation_test500_20260223_l1_posefocus_rpose2_from_nm4/quick_report.txt`
Next:
- 两路都完成后按同合同 Test500 的 coop-first 指标决策：必须保持 `mult<=1.3`，并让 `cross_agent_pose_trans_mean` 至少改善 10%（depth guardrail 不退化 >10%）。
- 若两路均无显著改善，下一轮优先尝试：训练 task 切到 `calibrated_sfm`（显式使用 ray dirs），或进一步提高/重排 cross-agent pose supervision（仍遵守“一次只改一个核心因素”）。

### 2026-02-23 03:45 +0000
Context: 在 Plan1 全量结果汇总后，继续推进“coop mult <= 1.3”目标；优先从最稳的 gate-pass ckpt 出发做小 LR / weight 微调 sweep。
Symptom:
- 上一轮矩阵结束后 GPU 会完全空闲；如果不主动发起下一轮就会停在“已有结论但未达标”的状态。
Root Cause (best guess):
- 当前瓶颈是“压 coop mult 时 single pose 容易越 guardrail”，需要在同一训练桶下系统探索更稳的更新幅度（LR/epoch/scale_weight）。
Fix:
- 以 `20260222_remote_nm_w015_r1` 的 ckpt 作为起点，固定训练桶 `0-60m`，启动 8 个并行 sweep：
  - `20260223_nm_ft_w{015,018,02}_lr{5e-6,3e-6}_e1`（6 个 1-epoch 探针）
  - `20260223_nm_ft_w015_lr5e6_e2`、`20260223_nm_ft_w018_lr3e6_e2`（2 个 2-epoch 探针）
Validation:
- 运行证据：
  - driver logs: `map-anything/experiments/local_runs/20260223_nm_ft_*_driver.log`
  - 每个 run 完成后都会生成：`map-anything/eval_runs/geom_ablation_test500_20260223_<run_tag>/quick_report.txt`
Next:
- 等 6 个 1-epoch run 先出 `quick_report`，挑出最有希望接近/突破 coop `mult<=1.3` 且不破 guardrail 的配置，再决定是否继续加 epoch 或进入结构性 Fix Ladder。

### 2026-02-23 03:35 +0000
Context: 用户问“现在什么情况”，需要给出明确的当前状态 + 已产出结果 + 下一步动作，而不是只报进程状态。
Symptom:
- GPU 全空闲（`nvidia-smi` 显存=1MiB），意味着上一轮矩阵训练/评测已全部结束，若不主动续跑会“停在结果汇总”阶段。
Root Cause (best guess):
- `run_geom_ablation_test500.sh` 的设计是“1 epoch + Test500 eval”，到点自然退出；需要下一轮策略才能继续向最终目标推进。
Fix:
- 汇总并点名 full bucket / w0.15 / 桶组合的最终 `quick_report.txt`，并更新主计划实时状态到“已完成 + 结论 + 下一步触发条件”。
- 明确本轮最佳 gate-pass 候选与 near-miss（coop 更接近 1.3 但 pose fail）的差异，为下一轮“保护 pose 的同时压 coop mult”提供起点。
Validation:
- 关键证据：
  - best gate-pass: `eval_runs/geom_ablation_test500_20260222_remote_nm_w015_r1/quick_report.txt`
  - best coop but pose fail: `eval_runs/geom_ablation_test500_20260222_local_full_w02_r1_detach/quick_report.txt`
Next:
- 启动下一轮（以 `remote_nm_w015_r1` 为 base）并继续跑到达标为止；若仍不达标，升级 Fix Ladder（L1/L2）。

### 2026-02-22 19:26 +0000
Context: 用户追问“已经有产出是什么情况”，并要求资源用满 + 指标口径客观可比。
Symptom:
- 汇报口径容易混淆：同一个 run 会经历 train -> internal test -> fixed-contract Test500 eval（只有最后一步才会写 `eval_runs/.../quick_report.txt`）。
- 单机 8 卡存在短暂空闲（桶矩阵跑完后留下 GPU 空位）。
Root Cause (best guess):
- 产出物分散在两处：训练在 `experiments/local_runs/<tag>`，合同评测在 `eval_runs/geom_ablation_test500_<tag>`；若只看 `nvidia-smi` 或只看某一目录，会误判“没产出/空转”。
Fix:
- 汇总并点名 20260222 已完成的 12 份 `quick_report.txt`，直接给出 single/coop 的 mult/pose/depth 结果与 `overall_pass`。
- 更新 `docs/master_plan.md` 的 `2.3 实时执行状态`（含当前可用机器、已产出列表、in-flight 进度、下一步触发条件）。
- 为补满单机 8 卡，新增启动桶组合 run：
  - `20260222_remote_nm_w015_r1`（0-60m）
  - `20260222_remote_mf_w015_r1`（30-100m）
Validation:
- 已产出证据目录：`map-anything/eval_runs/geom_ablation_test500_20260222_*/quick_report.txt`（示例：`..._plan1_near_sgpu_r1/quick_report.txt`）。
- 新增 run 已进入训练并持续写入：`map-anything/experiments/local_runs/20260222_remote_{nm,mf}_w015_r1/pinhole_pose_depth_scale/train.log`。
Next:
- 等待 full 桶与 w0.15 sweep 的 `quick_report.txt` 落盘后，统一做表格对比并推进 Fix Ladder（不满足 gate 则升级策略）。

### 2026-02-22 16:50 +0000
Context: 用户要求“两台开发机资源都用满”，并明确反馈另一台看起来空闲。
Symptom:
- 本机最初仅 3 卡活跃（其余 GPU 空置）；远端虽有任务，但活性展示不稳定，易被误判为空闲。
- 本机新加后台任务在普通 `nohup` 启动方式下出现“日志停在 launch 命令行、进程很快消失”的现象。
Root Cause (best guess):
- 主要是后台拉起方式不稳（会话退出后子进程被回收），导致任务未真正驻留；不是配置本身错误（同配置前台可稳定训练）。
Fix:
- 本机改为 `setsid -f bash -lc` 启动新增 run，并逐个校验 `master_port` 对应 launch 进程是否存在。
- 对已空闲的 GPU 及时补位新增 run（含 `full + w0.15`），保证 8 卡不空转。
- 远端维持 8 路矩阵并持续核查 `launch.log` 进度与 `nvidia-smi` 显存占用。
Validation:
- 本机活跃 run（示例）：`20260222_local_*_detach`、`20260222_local_*_dbg*`、`20260222_plan1_*`；均有持续 epoch/test 进度行。
- 远端活跃 run：`20260222_remote_{near,mid,far,full}_{w012,w02}_r1` 全部在 train/test 阶段推进。
- 双机当前均为 8 卡显存占用态（含 train/test 切换期的短时低 util）。
Next:
- 持续监听所有 run 完成，第一时间收集 `quick_report.txt`；
- 产出统一对比表并决定下一轮主线配置（不再手工逐条追问）。

### 2026-02-22 23:10 +0800
Context: 用户回到原开发机，要求继续落实方案二（远端）并排查方案一退出原因。
Symptom:
- 方案一四个分桶 run 全部退出，未产出可用 quick_report。
- 方案二初次远端启动失败在权重下载阶段（网络慢 + 客户端异常）。
Root Cause (best guess):
- 方案一：
  - near/mid/far 在分布式 early-stage 出现 NCCL allreduce watchdog 超时（30min）；
  - tail 在 `min_agent_distance>=100m` 条件下 test split 无样本，数据集构建直接失败。
- 方案二：
  - 远端通过公网慢速拉权重导致流程不稳定。
Fix:
- 方案一：完成失败归因并记录证据行号，标记“本轮无有效指标”。
- 方案二：
  - 先修复远端 SSH 通路并确认可调度；
  - 重启为本地 warm-start 路径：
    - `load_pretrained_weights=false`
    - `pretrained_checkpoint_path=.../opv2v_single_vggt_pose_metric_vggt1k/.../checkpoint-best.pth`
  - 设置共享缓存：`TORCH_HOME=/J6P-perception/yijinxiong_workspace/.cache/torch`，绕开慢速下载。
Validation:
- 方案一失败证据：
  - near/mid/far timeout：`.../launch.log:1789`
  - tail no-scene：`.../launch.log:1795`
- 方案二当前 run：`20260222_plan2_single_metric_remote_e1_r4_torchcache`
  - warm ckpt 加载：`launch.log:1769`
  - 进入训练循环：`launch.log:3901`（`Epoch [0] [0/2500]`）
  - 8 卡显存均进入训练占用（~43GB）。
Next:
- 持续监控方案二直到本轮 1 epoch + eval 完成并回收指标；
- 针对方案一准备“可复现重跑修复包”（先解决 tail 空样本与分布式超时后再重跑）。

### 2026-02-22 22:30 +0800
Context: 用户切到新开发机（`10.199.96.61`），要求让原开发机（`10.199.96.16`）反向连通。
Symptom:
- 新开发机 `sshd` 使用只读 `/root/.ssh/authorized_keys`，`ws trust install-ssh` 直接失败（`Read-only file system`）。
Root Cause (best guess):
- 容器入口将 `/root/.ssh/authorized_keys` 挂载为只读 tmpfs，导致常规“追加 key”路径不可用。
Fix:
- 在新开发机执行 `ws trust init --yes` 生成本机 trust key。
- 修改 `/root/.ssh/sshd_config`，新增：
  - `AuthorizedKeysFile /root/.ssh/authorized_keys /J6P-perception/yijinxiong_workspace/.wsops/security/trusted_ssh_keys.pub`
- 发送 `SIGHUP` reload `sshd`，让服务直接读取共享盘 trusted keys。
Validation:
- 本机回环认证成功：
  - `ssh -i /root/.ssh/ws_trust_ed25519 -p 12222 root@127.0.0.1 'hostname && whoami'`
- 说明 `sshd` 已正确接受 `trusted_ssh_keys.pub` 中的 key。
Next:
- 在原开发机执行反向复测：
  - `ssh -i /root/.ssh/ws_trust_ed25519 -p 12222 root@10.199.96.61 'hostname && whoami'`
- 复测通过后，立即在 `10.199.96.61` 启动方案二 MVP。

### 2026-02-22 22:11 +0800
Context: 用户反馈已在目标机执行解锁动作（done），继续推进方案二远端接入。
Symptom:
- 复测后仍无法 SSH 认证，方案二无法在 `10.199.96.61` 启动。
Root Cause (best guess):
- 目标机 `authorized_keys` 未正确包含当前 trusted key（或权限导致 key 被 sshd 忽略）。
Fix:
- 执行三路复测，确认同一阻塞：
  - `ssh -p 12222 root@10.199.96.61`
  - `ssh -i /root/.ssh/ws_trust_ed25519 -p 12222 root@10.199.96.61`
  - `ws add root@10.199.96.61:12222`
- 同步更新主计划与方案评估文档，补充目标机侧需要核对的具体命令（authorized_keys 内容与权限）。
Validation:
- 三路复测均返回 `Permission denied (publickey,password/publickey)`，阻塞定位稳定。
- 文档状态已与最新阻塞点同步。
Next:
- 等用户回传目标机 `authorized_keys` 检查结果后，立即复测并继续远端落地方案二。

### 2026-02-22 21:53 +0800
Context: 用户要求把方案二放到内网独立开发机 `10.199.96.61` 并行落地，先做可达性与认证检查。
Symptom:
- 远端主机可握手，但无法认证登录，导致方案二无法直接启动。
Root Cause (best guess):
- 目标机尚未安装当前工作区 trusted SSH 公钥（`ws trust` 链路未完成目标机侧一次性安装）。
Fix:
- 本机完成 trusted-host 初始化：`ws trust init --yes`。
- 执行远端探测：
  - `ssh -p 12222 root@10.199.96.61`（失败：`Permission denied (publickey,password)`）
  - `ws add root@10.199.96.61:12222`（失败：`Permission denied (publickey)`）
- 给出目标机解锁动作：
  - `/J6P-perception/yijinxiong_workspace/luca --doctor`（推荐）或 `ws trust install-ssh`；
  - 或手动安装 `/J6P-perception/yijinxiong_workspace/.wsops/security/trusted_ssh_keys.pub` 到 `/root/.ssh/authorized_keys`。
Validation:
- 认证前置条件已在本机满足；阻塞点明确定位在目标机授权缺失。
- `docs/master_plan.md` 与 `docs/coop_scale_strategy_options_eval.md` 已同步“方案二接入阻塞 + 解锁条件”。
Next:
- 等目标机完成一次性 key 安装后立即复测 SSH；
- 复测通过即在该机启动方案二 MVP（single metric-depth）并回收 Test500 single 对照结果。

### 2026-02-22 21:28 +0800
Context: User要求“每次梳理都同步计划文档与实时进度”，同时继续执行方案一分桶训练验证（near/mid/far/tail）。
Symptom:
- 诊断结论已完成，但主计划文档未显式标注“实时执行状态”和“更新规则”，容易造成口径漂移。
- 分桶训练 near/mid 已启动，但尚未产出最终 quick_report，状态需要持续同步。
Root Cause (best guess):
- 之前文档侧重静态方案与历史结论，缺少“进行中 run_tag + 下一步触发条件”的状态栏。
Fix:
- 更新 `docs/master_plan.md`：
  - `Last updated` 改为 2026-02-22；
  - 新增 `2.3 实时执行状态`（记录诊断已完成、训练进行中、产物路径）；
  - 在 `5. 追踪与留存` 增加“每次梳理必须同步状态栏”的规则。
- 更新 `docs/coop_scale_strategy_options_eval.md`：
  - 增加 `1.9 实时进度`（明确“诊断完成、训练在跑”）；
  - 增加 `2.6 与方案一/三关系`（澄清方案二训练实现独立，但评测验收口径不独立）。
Validation:
- 方案一诊断证据保持不变且可直接引用：
  - `docs/coop_scale_bucket_eval_test500_20260221.md`
  - 关键结论：`corr(distance, g_coop)=0.987`，coop 随距离桶显著退化，tail 主导失败。
- 进行中 run（2026-02-22 13:26 UTC）：
  - near: `20260222_plan1_debug_near`
  - mid: `20260222_plan1_debug_mid`
  - phase2 自动接续脚本：`/tmp/plan1_phase2_after_near_mid.sh`
  - 运行健康：`stalled_suspected`（进程在跑，但 15+ 分钟无 `Train Epoch` 新行）
Next:
- 继续盯 near/mid -> far/tail 全链路完成并回收 `quick_report.txt`；
- 产出统一四桶对比结论（full + bucket）后再决定是否升级到方案三主线。

### 2026-02-21 08:48 +0800
Context: Plan-1 verification for coop scale difficulty by bucketing a fixed Test500 contract by cross-agent baseline distance (no new inference).
Symptom:
- Coop `scale_to_gt_*` and `pose_abs_mean` show heavy tails; unclear whether this is a model issue vs a GT/normalization-definition effect.
- Need an evidence-backed answer for “is coop scale hard because baseline distance pushes `g_coop` into a heavy tail?”
Root Cause (best guess):
- Under `norm_mode=avg_dis`, coop GT scale `g_coop` is computed as mean distance-to-origin over all valid points after transforming both agents into ego view0; this directly incorporates cross-agent baseline distance.
Fix:
- Implemented bucket tooling and report generation:
  - `scripts/bucket_frames_by_baseline_distance.py` (generates bucketed frame contracts + report)
  - `scripts/report_geom_by_baseline_buckets.py` (re-aggregates existing `*_metrics.csv` per bucket; computes corr(distance, g_coop))
- Produced plan-1 verification report (fixed contract, exact re-aggregation):
  - `docs/coop_scale_bucket_eval_test500_20260221.md`
Validation:
- Bucket report created:
  - `eval_runs/frames_test500_dist_buckets_20260221/bucket_report.md`
- Verified strong correlation on Test500:
  - corr(distance, g_coop) ≈ 0.987 (reported in `docs/coop_scale_bucket_eval_test500_20260221.md`)
- Verified error growth with distance buckets (coop):
  - near (<=30m) `mult` ~1.17–1.22, tail (>=100m) `mult` ~3.45–5.60 depending on ckpt; pose mean also explodes (~44–48m in tail).
Next:
- Use bucket-level metrics to drive tail-focused interventions (reweight/curriculum or redefine scale target to be ego-centric / per-agent local).
- Treat tail bucket metrics as an explicit regression test for any future scale/pose changes.

### 2026-02-20 20:40 +0800
Context: Resolve user confusion on OPV2V scale semantics by re-validating “best ckpt” candidates with metrics + representative visualizations under the same fixed Test500 contract.
Symptom:
- Users observe “scale gets worse” when reading legacy `scale_err_mean=|s-1|` (which is meaningless under `norm_mode=avg_dis`) or reading loss-space `FactoredGeometryScale..._scale`.
- New promoted ckpt (`l10a`) improved Test500 metrics, but there was no updated best/median/worst visualization audit to match the corrected metric (`scale_to_gt_*`).
Root Cause (best guess):
- Old viz audit (`2026-02-17`) only covered earlier baselines; later ablations changed scale distribution materially.
- Metric naming (`scale_err` vs `scale_to_gt_err`) invites misinterpretation without a paired visual check.
Fix:
- Generated a new scale visualization audit set (baseline vs promoted vs scale-strong-but-pose-worse):
  - `map-anything/eval_runs/scale_viz_audit_test500_20260220/index.html`
  - per-ckpt: `l10a_w0p12`, `l8b3_pose075`, `sfix_direct`
- Wrote a compact metric+viz report with the exact fixed-contract numbers:
  - `docs/scale_metric_visual_audit_20260220.md`
- Synced planning docs to avoid wrong CLI flags (`--out_dir` vs `--output_dir`) and to include the `--summary_json` requirement for HTML generation.
Validation:
- Verified each ckpt has 6 HTML files (single/coop × best/median/worst) under:
  - `map-anything/eval_runs/scale_viz_audit_test500_20260220/<ckpt_tag>/html_scale/index.html`
- Representative stats show the expected ranking on coop:
  - baseline median coop ratio ~0.48, worst ratio ~0.12 + pose_abs ~80m
  - `l10a` improves coop median ratio ~0.68, worst ratio ~0.18 + pose_abs still ~80m
  - `l8b3` improves coop further (median ratio ~0.79) but pose regresses (not promotable)
Next:
- Use the worst-representative frames to drive tail-focused fixes (supervision density + coop-tail robustness) instead of further increasing direct-scale weights.

### 2026-02-20 20:21 +0800
Context: Continue from `l9a2` to close coop scale gap while preserving pose/depth guardrails; executed L10 and L11 loops in parallel.
Symptom:
- `l9a2` already passes hard gate, but coop `mult` still too high (`1.9498` vs final `<=1.3`).
- Stronger scale variants historically tend to break pose or collapse.
Root Cause (best guess):
- Scale/pose trade-off is tight near current boundary; once scale pressure goes beyond stability margin, training enters NaN/bad-loss region and downstream eval loses valid aggregates.
Fix:
- Added and tested incremental variants:
  - `l10a`: `w0.12` from `l9a2`
  - `l10b`: `w0.15 + pose075` from `l9a2`
  - `l10c`: `w0.12 + pose075` from `l9a2`
  - `l11a`: `w0.12`, lower LR (`5e-6`) from `l10a`
  - `l11b`: `w0.13` from `l10a`
- Kept all runs on the same fixed Test500 contract and baseline guardrail policy.
Validation:
- Best run: `l10a` (`overall_pass=True`)
  - `eval_runs/geom_ablation_test500_20260220_l10a_w0p12_vr1p_noconf_from_l9a2/quick_report.txt`
  - single `mult=1.1151`, coop `mult=1.6359`, pose/depth guardrails pass.
- Failures:
  - `l10b`: all-frame failure (missing single/coop metrics).
  - `l10c`: single pose guardrail fail (`0.0640 > 0.0628`).
  - `l11a` / `l11b`: all-frame failure (missing single/coop metrics).
Next:
- Promote `l10a` as new base.
- Shift next loop toward stability-first structural improvements (supervision density / tail robustness), not larger scale weights.

### 2026-02-20 18:22 +0800
Context: Continue fixed-contract geometry loop with two parallel L9 reruns to resolve “scale improves but pose guardrail fails” conflict.
Symptom:
- Previous L8 branch improved scale strongly but still failed single-pose guardrail.
- Fresh L9 launch attempts got stuck early with no usable training artifacts due config/runtime hygiene issues.
Root Cause (best guess):
- New loss YAMLs had unquoted string literals (`norm_mode=avg_dis`, `depth_type_for_loss=depth_along_ray`), which can break `eval(args.loss.train_criterion)` in training startup.
- Prior asynchronous launch attempts mixed stale monitor/log processes and made run-state diagnosis noisy.
Fix:
- Patched loss configs:
  - `configs/loss/opv2v_vggt_pose_scale_loss_recover_direct_w0p1_vr1p_noconf.yaml`
  - `configs/loss/opv2v_vggt_pose_scale_loss_recover_direct_w0p15_vr1p_noconf.yaml`
  - quoted `norm_mode` and `depth_type_for_loss`.
- Relaunched two clean parallel runs with fixed tags (`l9a2`, `l9b2`) and completed full train+eval pipeline on Test500.
Validation:
- `eval_runs/geom_ablation_test500_20260220_l9a2_w0p1_vr1p_noconf_from_sfix/quick_report.txt`
  - `overall_pass=True`
  - single: `mult=1.2557`, `pose=0.0587`, `depth=0.3214`
  - coop: `mult=1.9498`, `pose=5.6596`, `depth=0.3280`
- `eval_runs/geom_ablation_test500_20260220_l9b2_w0p15_vr1p_noconf_from_sfix/quick_report.txt`
  - stronger scale but `overall_pass=False` due single pose guardrail (`0.0670 > 0.0628`).
Next:
- Promote `l9a2` ckpt as next-iteration startpoint and run targeted follow-up to further reduce coop `scale_to_gt_mult_err_mean` while preserving single-pose guardrail.

### 2026-02-18 21:15 +0800
Context: L4 twin runs finished training and entered fixed-contract Test500 evaluation; quick-report generation failed despite summary outputs.
Symptom:
- both L4 runs produced `summary_test.json` but no `quick_report.txt`;
- evaluation logs contained only singular-inference warnings, with no valid single/coop aggregate metrics.
Root Cause (best guess):
- embedded quick-report snippet in `run_geom_ablation_test500.sh` assumed `metrics.geom_model.{single,coop}` always exist and crashed on empty metrics;
- ablation launcher defaulted to `RESUME=true`, likely inheriting unstable optimizer/scheduler states; combined with late-epoch NaN storm, this yielded unusable checkpoints.
Fix:
- patched `scripts/run_geom_ablation_test500.sh`:
  - quick-report now supports missing mode metrics;
  - default `RESUME=false`;
  - default `MAX_BAD_LOSS_COUNT=50` (fail fast).
- backfilled quick reports for completed L4 outputs from existing summaries.
Validation:
- `eval_runs/geom_ablation_test500_20260218_l4a_recover_w0p2_vr1p_noconf_from_l2_lr1e6/quick_report.txt` => overall FAIL, both modes missing.
- `eval_runs/geom_ablation_test500_20260218_l4b_recover_w0p2_vr1p_noconf_from_l1c_lr1e6/quick_report.txt` => overall FAIL, both modes missing.
- `eval.log` for both runs shows `Inference failed ... singular` exactly 1000 times (500 frames × 2 modes).
Next:
- do not promote L4 checkpoints; rerun safer non-resume ablations from stable checkpoints and gate on Test500.

### 2026-02-18 20:35 +0800
Context: Completed L3 parallel runs (8xA800 split into 2x4) and retried fixed-contract Test500 eval.
Symptom:
- Both L3 checkpoints produced widespread inference failures on Test500 (`torch.linalg.solve` singular in intrinsics recovery).
- Original eval wrapper crashed before report generation when mode metrics were missing.
Root Cause (best guess):
- L3 configs pushed model into degenerate output regime (ray/intrinsics collapse), so most frames became non-evaluable.
- Eval/report scripts assumed single+coop metrics always exist.
Fix:
- Added inference-level skip guard in `scripts/batch_eval.py` (skip bad frame/mode instead of aborting entire eval).
- Hardened report generation in `scripts/eval_geom_ckpt_test500.sh` for missing mode metrics.
- Re-ran both L3 checkpoints on Test500 (retry outputs).
Validation:
- Retry eval writes `summary_test.json` for both runs:
  - `eval_runs/geom_ablation_test500_20260218_l3a_detach_w0p6_pose12_vr0p75_from_sfix_r2_retry/summary_test.json`
  - `eval_runs/geom_ablation_test500_20260218_l3b_detach_w0p4_pose12_vr1p_noconf_from_l2_r2_retry/summary_test.json`
- Both summaries have no usable single/coop aggregates (all-frame failure regime); quick reports marked FAIL with missing metrics.
Next:
- Drop this unstable branch from promotion candidates.
- Resume from stable branch (L1/L2 neighborhood) with stricter anti-collapse safeguards before next 2x4 sweep.

### 2026-02-18 19:58 +0800
Context: User reported “CPU满/GPU空” behavior and asked for immediate anomaly check while continuing.
Symptom:
- First relaunched L3 jobs appeared idle on GPU; run then failed early due missing pretrained ckpt path.
- Eval pipeline on non-pc-metrics contract still spent excess CPU on unnecessary point-cloud conversion.
Root Cause (best guess):
- Relative ckpt path passed into hydra train became invalid after working-directory switch.
- `batch_eval.py` converted predictions to point clouds even when `--pc_metrics` and `--pc_save_dir` were both off.
Fix:
- `scripts/run_geom_ablation_test500.sh`: normalize pretrained ckpt to absolute path with early existence check.
- `scripts/batch_eval.py`: skip point-cloud conversion unless pc metrics/dump is explicitly requested.
- Relaunched L3 jobs after fix.
Validation:
- Relaunch consumed all 8 GPUs as expected (2x4 split), training progressed through epoch loops.
- Path-related immediate failures no longer reproduced.
Next:
- Complete this L3 cycle and gate via Test500 quick reports.

### 2026-02-18 15:14 +0800
Context: Launch first L1 ablation pair from the new master plan (detach-scale vs control, same baseline ckpt, same epochs).
Symptom: First attempt failed immediately in dataloader workers with repeated `OSError: AF_UNIX path too long`.
Root Cause (best guess): TMP path used by multiprocessing resource_sharer was too long under workspace path composition.
Fix:
- Changed training entry TMP default to short path `/tmp/ma_tmp` and added a path-length guard.
- Re-launched two comparable runs:
  - `20260218_l1_detach_v1` (`opv2v_vggt_pose_scale_loss_direct_detach_geom_scale`)
  - `20260218_l1_control_v1` (`opv2v_vggt_pose_scale_loss_direct`)
Validation:
- Both runs now pass distributed init and start training steps.
- Initial logs show expected metrics including `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale_valid_ratio_avg`.
Next:
- Wait for epoch completion + Test500 eval auto-generated by run script.
- Compare quick reports (`.../quick_report.txt`) and decide whether to promote detach-scale config to longer run.

### 2026-02-14 16:22 +0800
Context: User要求“继续优化到找不到优化点”，并强调量化公平对比与路线点名。
Symptom: 现有看板仍缺少两点：1) 针孔锁定 baseline summary 缺少 `run_info`；2) 同帧集 Top-K 结果缺少一键复现脚本与明确结论。
Root Cause (best guess): 历史评测产物生成时间较早，summary schema 不统一（缺 run_info / 新 scale diagnostics）。
Fix: 新增 `backfill_pinhole_test50_summaries.py` 回填 baseline summary；新增 `scoreboard_scan_test50_pinhole.py` 做同帧集 Top-K；更新 `workspace_clean_summary.md` 写明同帧集结论与证据。
Validation: 
- `det_e2e_v5_eval_t005/summary_test.json` 与 `det_e2e_v5_coop_eval_t005/summary_test.json` 已含 `run_info.det_decode_cfg={iou:0.5, score:0.05}` 与 ratio-space scale 字段；
- `python scripts/scoreboard_scan_test50_pinhole.py --topk 8` 输出同帧集排行榜并显示 baseline 仍为最佳。
Next: 若要继续提高客观性，下一步应补齐“无泄漏 cylindrical held-out”评测，不再新增同口径文档层改动。

### 2026-02-12 23:49 +0800
Context: Reviewed codex_threads prompts/meta to补齐“做了什么/为何失败”的记录缺口。
Symptom: 工作区总览缺少自动化失败原因与重试模式（仅列目录）。
Root Cause (best guess): 未纳入 codex_threads 里 `meta.txt/prompt.md` 的失败原因与任务统计。
Fix: 汇总 codex_threads（59 条）按 reason/task 统计并提炼典型异常（retry_failure、bad loss nan、gate_fail）。
Validation: `docs/workspace_overview.md` 新增“Codex 调试记录补充”小节，列出 counts 与代表性日志。
Next: 若需要更细粒度，可按任务自动抽取 `prompt.md` 的 Last Command 与 log tail 形成更细表格。

### 2026-02-12 23:16 +0800
Context: Workspace overview was missing BEV head attempts and “good-looking” ckpts (pinhole/cyl) with purpose/results/meaning.
Symptom: Summary was overly coarse and only listed paths, lacking interpretation of outcomes and project significance.
Root Cause (best guess): Consolidation pass relied on directory inventory but did not incorporate `results_overview.md` and eval artifacts.
Fix: Expanded workspace overview with BEV/det-head attempt summary, representative ckpt list (pinhole & cyl), and direct evidence paths.
Validation: `docs/workspace_overview.md` now includes sections 2.7–2.8 with BEV head attempts, ckpt meaning, and links to eval/visualization assets.
Next: If further detail is required, add an automated metric extraction table for key runs.

### 2026-02-12 10:35 +0800
Context: 4x4090 sequential scale plan on OPV2V Coop (S1 warm/full → S2 gating → S3 high scale weight).
Symptom: Full-test scale_err_mean ~18 (epoch1) and epoch2 full-test lines show NaN loss + scale_err_mean=0 (invalid).
Root Cause (best guess): Scale supervision sparse (~3.7% valid); late-epoch instability/NaNs contaminate full-test metrics.
Fix: Keep noconf + valid-ratio gating, enforce invalid-run filtering, and parse last available Test Epoch line.
Validation: S2 epoch1 full-test shows scale_err_mean ~18.07 with scale_valid_ratio_avg ~0.037; invalid epoch2 lines flagged.
Next: Let S2 epoch2 finish, then run S3 (w1.0). If still >3, move to depth densification + calibration or scale-consistency loss.

### 2026-02-11 05:45 +0800
Context: Start Stage3 variant on Node A with higher scale weight and noconf wrapper.
Symptom: Stage2 S7a finished with scale_err_mean ~11.78; still far from target 1–3.
Root Cause (best guess): Scale loss weight too weak to correct large ratio bias.
Fix: Launch S8a with `scale_loss_weight=1.0`, noconf loss wrapper, base ckpt from S7a best.
Validation: Run tag `20260210_s8a_full_w1p0_noconf` started on port 29572.
Next: Monitor for NaN/instability and compare scale_err_mean vs Stage2.

### 2026-02-11 05:41 +0800
Context: S7a full run (vr1p gating) finished 2 epochs with full test.
Symptom: Final ratio-space scale error still >> target.
Root Cause (best guess): Scale supervision remains sparse; direct scale loss not sufficient without denser depth or loss shaping.
Fix: Proceed to parallel noconf S7b run; prepare Stage3 loss-shaping options if S7b fails.
Validation: Full test (epoch2, [203/204]) shows pose_trans_l2_m=0.7389, pose_rot_deg=1.6028, depth_z_mae_m=1.6678, depth_z_rmse_m=5.6192, scale_err_mean=11.7760, scale_log_err_mean=2.5458, scale_ratio_mean=12.7760.
Next: Compare against S7b noconf results; if scale still >5, move to Stage3.

### 2026-02-11 05:40 +0800
Context: S7b full run (LR=1e-5) still produced NaN in conf loss at epoch0 iter12.
Symptom: Repeated `[WARN] Bad loss=nan` with conf loss components = nan.
Root Cause (best guess): Conf loss path is unstable on some real batches; NaNs propagate before scale gating.
Fix: Stop S7b run and relaunch with noconf loss wrapper (`ExcludeTopNPercentPixelLoss` for train).
Validation: New run tag `20260210_s7b_full_from_warm_vr1p_lr1e5_noconf` launched on port 29563.
Next: Watch for NaNs; if stable, record full test metrics and compare to S7a.

### 2026-02-11 05:32 +0800
Context: Re-run Stage2 full training on second A800 to avoid NaN/instability.
Symptom: Prior S7b full run hit NaN/bad_loss in epoch1 and aborted.
Root Cause (best guess): LR too high for direct scale loss + conf term; unstable batches early.
Fix: Relaunched full run with LR=1e-5 from warm ckpt (`20260210_s7b_warm_vr1p`), loss `opv2v_vggt_pose_scale_loss_recover_direct_w0p2_vr1p`.
Validation: Distributed init logged; run dir created at `experiments/local_runs/20260210_s7b_full_from_warm_vr1p_lr1e5_retry/pinhole_pose_depth_scale`.
Next: Monitor for NaN/bad_loss; record full test metrics when finished.

### 2026-02-11 05:29 +0800
Context: Stage2 full run S7a from warm ckpt with min_scale_valid_ratio gating (0.01).
Symptom: Ratio-space scale error still high after epoch0 full test.
Root Cause (best guess): Scale supervision still sparse/biased; direct scale loss not pulling ratio to 1.
Fix: Continue training to epoch1; launch parallel lower-LR run on second A800.
Validation: Full validate test (200) shows pose_trans_l2_m=0.7010, pose_rot_deg=0.6334, depth_z_mae_m=2.0191, scale_err_mean=11.8655, scale_log_err_mean=2.5538, scale_ratio_mean=12.8655.
Next: Wait for epoch1 completion and compare with S7b; if still >3, advance to Stage3 loss shaping.

### 2026-02-09 06:58 +0800
Context: Close all open review-fix engineering items (watchdog e2e, eval metadata/schema backfill, tmp robustness).
Symptom: T5b and eval-comparability tasks were still open in docs; watchdog smoke previously failed due parser/runtime issues.
Root Cause (best guess): Queue heredoc regex in watchdog script had a bash parse bug; previous smoke runs also mixed incompatible runtime env and malformed queue command.
Fix: Patched `auto_queue_watchdog.sh` heredoc detection for bash compatibility, executed a controlled watchdog smoke (`watchdog_t5b_smoke_20260208_2250_v5`) to PASS gate and auto-trigger eval, and regenerated key eval summaries (`auto_queue_det_eval_smoke`, `auto_queue_det_eval_full`) with new schema + `run_info`.
Validation: `auto_queue.state=1`, watchdog log shows gate PASS + auto-eval execution, smoke/full summaries now include `run_info` and new scale fields.
Next: Focus on model-quality loop (real gate pass and AP lift); documentation/tooling closure is complete for this iteration.

### 2026-02-09 04:45 +0800
Context: User confusion on scale metric semantics (`0.0006`-like values) and planning interpretation mismatch.
Symptom: Scale values in gate/logs were interpreted as if they should be near 1 instead of near 0.
Root Cause (best guess): Mixed use of loss-space scale key and ratio-space scale quality without explicit naming convention.
Fix: Updated plan/logbook semantics globally: use `scale_loss` for `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale`, and pair it with eval `scale_err_mean` for human interpretation.
Validation: Goal-based test plan and logbook now consistently define optimization-space vs interpretation-space scale metrics.
Next: Keep dual-track reporting in every gate/eval status update; backfill key eval summaries with the new scale diagnostics.

### 2026-02-08 06:58 +0800
Context: Documentation optimization pass for experiment logbook + goal-based test plan.
Symptom: Baseline wording was ambiguous; T3/T5 acceptance criteria were not strict enough for reproducibility.
Root Cause (best guess): Plan evolved quickly and documentation mixed milestone notes with strict pass/fail policy.
Fix: Unified baseline policy (locked vs reference), added reproducibility/preflight requirements, split T5a/T5b, and synced snapshot/goals/todo sections.
Validation: Logbook and test plan now agree on baseline values, status vocabulary, and unresolved gaps.
Next: Run watchdog closed-loop verification and add checkpoint metadata to eval summaries.

### 2026-02-08 05:16 +0800
Context: Run geometry smoke + auto-eval tests for T3–T6.
Symptom: Gate failed; det AP 0.0 on current checkpoint.
Root Cause (best guess): Geometry still not at gate thresholds; checkpoint not det-trained.
Fix: Complete 1-epoch smoke; generate gate summary; run auto-eval smoke + full eval.
Validation: Checkpoints saved; gate FAIL; auto-eval outputs generated; det AP single/coop = 0.0.
Next: Improve geometry metrics to pass gate; train det head/e2e before re-eval.

### 2026-02-08 05:02 +0800
Context: Geometry smoke run attempt (T3) using tmp output.
Symptom: Training aborted while saving checkpoint with "file write failed".
Root Cause (best guess): Root FS (/tmp) is 100% full; torch elastic/temp writes failed.
Fix: Rerun with TMPDIR/TORCHELASTIC_TMPDIR pointing to workspace and outputs under experiments/local_runs.
Validation: Initial run failed; cleanup required for retry.
Next: Rerun smoke with TMPDIR override.

### 2026-02-08 04:45 +0800
Context: Gate check on recent scale-only run summary.
Symptom: Gate failed with depth/pose metrics above thresholds.
Root Cause (best guess): Geometry quality not yet at gate readiness on scale-only run.
Fix: None (expected failure confirms gate behavior).
Validation: `gate_metrics.py` reports FAIL with metrics above GATE_MAX.
Next: Improve geometry metrics (training/recovery) before det/e2e stages.

### 2026-02-08 04:37 +0800
Context: Validate index loader and define goal-based test standards.
Symptom: Index-based dataset load returned zero scenes due to agent list decoding.
Root Cause (best guess): Parquet decodes agents as numpy arrays; loader only accepted list/tuple.
Fix: Accept numpy arrays in index loader; raise explicit error on empty index results; add goal-based test plan doc.
Validation: OPV2V index hit log appears and filtered scenes > 0 in smoke test.
Next: Run geometry stability + gate + auto-eval + det AP checks (T3-T6).

### 2026-02-08 01:47 +0800
Context: Close all remaining review-fix-loop TODOs (index, wiring, badloss cap, gates, auto-eval).
Symptom: Index missing; dataset still scanning YAML; badloss dumps unbounded; gates/auto-eval incomplete.
Root Cause (best guess): Tasks not yet implemented.
Fix: Built OPV2V index; added dataset index loader + config; capped badloss dumps; tightened gate thresholds; wired auto-eval in watchdog.
Validation: `opv2v_index/index_{train,validate,test}.parquet` + `manifest.json` + `schema.json` exist; config + code patched.
Next: Run a short training start to confirm index hit and auto-eval artifact generation.

### 2026-02-08 00:53 +0800
Context: Apply review-fix-loop to refine documentation.
Symptom: Missing review loop summary + skill install status in logbook.
Root Cause (best guess): Skill did not exist at the time of prior doc pass.
Fix: Created review-fix-loop skill and added review summary + checked install item.
Validation: Skill validated; logbook updated with review section.
Next: Execute index build and dataset wiring tasks.

### 2026-02-08 00:02 +0800
Context: Expand TODOs into executable steps and attempt review-fix skill install.
Symptom: TODO items lacked concrete commands/params; skill listing failed (HTTP 403).
Root Cause (best guess): Missing pandas/index inputs; GitHub API rate limit or auth required.
Fix: Added command-level steps + verify criteria to TODO backlog; started proxy and attempted list-skills.
Validation: TODO backlog now contains commands/verification; skill list still blocked (HTTP 403).
Next: Provide OPV2V root/depth paths; add pandas; rerun index build; provide GitHub token or skill path.

### 2026-02-07 23:23 +0800
Context: Session audit + progress doc + cleanup of badloss artifacts.
Symptom: 3.47 TB badloss debug dumps and unclear task progress.
Root Cause (best guess): Unbounded badloss dump logic + long-running NaN loops.
Fix: Deleted all badloss_* artifacts from `/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/experiments/local_runs/20260206_155329_pinhole_pose_depth_scale_recover_nanfix/pinhole_pose_depth_scale`; recorded status snapshot + key metrics.
Validation: badloss files=0; run dir size now 8.5G.
Next: Add dump cap; build OPV2V index; integrate index path; define gate to switch to det.

### 2026-02-03 08:45 +0800
Context: Relaunch recovery run with corrected loss config.
Symptom: Previous run (v5) still hit NameError (stale config).
Root Cause (best guess): Loss config not updated before launch.
Fix: Patch loss config in repo and start new run (recover_v6).
Validation: DDP init succeeds; waiting for criterion creation.
Next: Check first eval metrics and confirm no NameError.

### 2026-02-03 08:40 +0800
Context: Recovery run launch failed with NameError in loss config.
Symptom: "NameError: name avg_dis is not defined" during loss eval.
Root Cause (best guess): Missing quotes around string args in loss config.
Fix: Add quotes for norm_mode and depth_type_for_loss; restart run (recover_v5).
Validation: DDP init succeeds; training proceeds to dataset build.
Next: Monitor first eval metrics and stability for 5+ minutes.

### 2026-02-03 08:36 +0800
Context: Start geometry recovery run (8xGPU) from 20260130 checkpoint.
Symptom: Previous stage3 run regressed geometry; needed stable restart.
Root Cause (best guess): Over-strong loss weights; squared relative pose in loss.
Fix: Launch recovery config with lower pose/scale weights and squared=False.
Validation: Monitor first eval (epoch 0/1) for pose_trans < 1m and pose_rot < 2deg trend.
Next: If stable, keep 6 epochs; then switch to det-head-only.

### 2026-02-03 08:20 +0800
Context: Create recovery training plan from 20260130 checkpoint (better pose).
Symptom: Stage3 superstrong run regressed geometry.
Root Cause (best guess): Over-strong loss weights; small 32m dataset.
Fix: Add new loss config opv2v_vggt_pose_scale_loss_recover with lower pose/scale weights and squared=False; create recovery entry script.
Validation: Use recover script with 20260130 ckpt; compare test metrics after 1-2 epochs.
Next: Start recovery run, then det-head-only, then joint fine-tune.

### 2026-02-03 08:10 +0800
Context: Stage3 pinhole pose+depth+scale superstrong run (max_agent_distance=32).
Symptom: Geometry regressed badly; depth MAE and pose errors much worse than stage2 start.
Root Cause (best guess): Loss weights too strong + squared relative pose; small dataset (121 scenes) amplifies instability.
Fix: Stop stage3 run; create "recover" loss config with lower pose/scale weights and squared=False.
Validation: Compare metrics before/after on test split; ensure pose_trans < 1m, pose_rot < 2deg.
Next: Start new run from 20260130 best checkpoint using recover config.

### 2026-02-03 07:40 +0800
Context: Resume training from stage3 checkpoint.
Symptom: Resume skipped due to non-finite values in checkpoint (scale token).
Root Cause (best guess): Scale head output exploded; non-finite parameters.
Fix: Clamp and nan_to_num on scale output; add checkpoint repair step before resume.
Validation: Resume no longer skipped; check "Resume checkpoint" line in launch log.
Next: Keep monitoring for bad loss; lower scale loss weight if repeat.

### 2026-02-03 07:15 +0800
Context: Batch size expectations vs actual GPU memory.
Symptom: User expected batch 40-48 but actual per-GPU batch fixed at 32.
Root Cause: Script caps MAX_IMGS_PER_GPU to 32 for 8-view DINOv2-large.
Fix: Document cap in scripts; adjust if safe.
Validation: .hydra/config.yaml shows max_num_of_imgs_per_gpu=32.
Next: If needed, raise cap with a short OOM test.

---

## RUN REGISTRY (active / important)

- Convention: all `scale_loss` values below refer to `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` (loss-space, lower is better toward 0).


- 20260223_nm4_scaleonly_w02_lr5e6_e1_fix (current promoted geometry startpoint; deploy protocol baseline)
  - ckpt: `experiments/local_runs/20260223_nm4_scaleonly_w02_lr5e6_e1_fix/pinhole_pose_depth_scale/checkpoint-best.pth`
  - fixed Test500 metrics (deploy protocol: `model_task=calibrated_sfm` / coop-first / keep_camera_poses=0):
    - coop: `scale_to_gt_mult_err_mean=1.2693` (<=1.3 ✅), `scale_to_gt_ratio_mean=0.9493`, `cross_agent_pose_trans_mean=10.5028m`, `depth_rel_mean=0.3468`
  - note: current gating baseline used by `scripts/run_geom_ablation_test500.sh` quick_report comparisons.
  - evidence:
    - `eval_runs/geom_recheck_test500_20260223_nm4_scaleonly_w02_calibrated/summary_test.json`

- 20260220_l10a_w0p12_vr1p_noconf_from_l9a2 (previous promoted geometry startpoint; historical)
  - ckpt: `experiments/local_runs/20260220_l10a_w0p12_vr1p_noconf_from_l9a2/pinhole_pose_depth_scale/checkpoint-best.pth`
  - fixed Test500 metrics:
    - single: `scale_to_gt_mult_err_mean=1.1151`, `scale_to_gt_ratio_mean=0.9971`, `pose_abs_mean=0.0622`, `depth_rel_mean=0.3161`
    - coop: `scale_to_gt_mult_err_mean=1.6359`, `scale_to_gt_ratio_mean=0.7123`, `pose_abs_mean=5.8352`, `depth_rel_mean=0.3247`
  - note: best-known passing run so far; single is already close to final target band, coop still above final target (`<=1.3`).
  - evidence:
    - `eval_runs/geom_ablation_test500_20260220_l10a_w0p12_vr1p_noconf_from_l9a2/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l10a_w0p12_vr1p_noconf_from_l9a2/summary_test.json`

- 20260220_l10b/l10c/l11 variants (demoted)
  - runs:
    - `20260220_l10b_w0p15_vr1p_noconf_pose075_from_l9a2`
    - `20260220_l10c_w0p12_vr1p_noconf_pose075_from_l9a2`
    - `20260220_l11a_w0p12_lr5e6_from_l10a`
    - `20260220_l11b_w0p13_from_l10a`
  - note:
    - `l10b`/`l11a`/`l11b`: all-frame failure (missing single/coop aggregates).
    - `l10c`: scale improves but single pose guardrail fails.
  - evidence:
    - `eval_runs/geom_ablation_test500_20260220_l10b_w0p15_vr1p_noconf_pose075_from_l9a2/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l10c_w0p12_vr1p_noconf_pose075_from_l9a2/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l11a_w0p12_lr5e6_from_l10a/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l11b_w0p13_from_l10a/quick_report.txt`

- 20260220_l9a2_w0p1_vr1p_noconf_from_sfix (previous promoted checkpoint)
  - ckpt: `experiments/local_runs/20260220_l9a2_w0p1_vr1p_noconf_from_sfix/pinhole_pose_depth_scale/checkpoint-best.pth`
  - fixed Test500 metrics:
    - single: `scale_to_gt_mult_err_mean=1.2557`, `scale_to_gt_ratio_mean=0.8121`, `pose_abs_mean=0.0587`, `depth_rel_mean=0.3214`
    - coop: `scale_to_gt_mult_err_mean=1.9498`, `scale_to_gt_ratio_mean=0.5843`, `pose_abs_mean=5.6596`, `depth_rel_mean=0.3280`
  - note: first hard-gate PASS run in this branch (`overall_pass=True`), but coop scale is still above final target (`<=1.3`).
  - evidence:
    - `eval_runs/geom_ablation_test500_20260220_l9a2_w0p1_vr1p_noconf_from_sfix/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l9a2_w0p1_vr1p_noconf_from_sfix/summary_test.json`

- 20260220_l9b2_w0p15_vr1p_noconf_from_sfix (strong-scale but pose-regressed)
  - ckpt: `experiments/local_runs/20260220_l9b2_w0p15_vr1p_noconf_from_sfix/pinhole_pose_depth_scale/checkpoint-best.pth`
  - fixed Test500 metrics:
    - single: `scale_to_gt_mult_err_mean=1.1221`, `scale_to_gt_ratio_mean=0.9588`, `pose_abs_mean=0.0670`, `depth_rel_mean=0.3145`
    - coop: `scale_to_gt_mult_err_mean=1.7011`, `scale_to_gt_ratio_mean=0.6804`, `pose_abs_mean=5.6475`, `depth_rel_mean=0.3245`
  - note: scale improves more than `l9a2`, but single pose guardrail fails (`0.0670 > 0.0628`), so not promotable.
  - evidence:
    - `eval_runs/geom_ablation_test500_20260220_l9b2_w0p15_vr1p_noconf_from_sfix/quick_report.txt`
    - `eval_runs/geom_ablation_test500_20260220_l9b2_w0p15_vr1p_noconf_from_sfix/summary_test.json`

- opv2v_cyl_pose_quick (pose_r60_v1)
  - ckpt: experiments/opv2v_cyl_pose_quick/checkpoint-validate_2021_09_11_00_33_16_1016_pose_r60_v1.pth
  - metrics: pose_err_rel_trans_mean≈0.715m, pose_err_rel_rot_mean≈0.805°; chamfer_predpose≈0.565; bev_iou_predpose≈0.419
  - note: PredPose≈GTPose; cylindrical route reaches sub-meter alignment.
  - evidence: eval_runs/demo_alignment/metrics_pose_r60_v1_recheck.json, docs/cyl/results_overview.md

- opv2v_cyl_geom_quick (geom_lidar_v1)
  - ckpt: experiments/opv2v_cyl_geom_quick/checkpoint-test_2021_08_24_20_49_54_207_geom_lidar_v1.pth
  - metrics: chamfer_gtpose≈0.44, bev_iou_gtpose≈0.426; pose_err_rel_mean remains large
  - note: Geometry is good under GTPose; remaining gap is pose alignment.
  - evidence: eval_runs/demo_alignment/metrics_geom_lidar_v1_r60_baseline.json, docs/cyl/results_overview.md

- opv2v_cyl_det_head_quick (det_v1 quick finetune)
  - ckpt: experiments/opv2v_cyl_det_head_quick/checkpoint-test_2021_08_24_20_49_54_207_det_v1.pth
  - metrics: det_ap_iou=1.000 (5-frame quick finetune)
  - note: Pipeline closure proof, not a generalization result.
  - evidence: eval_runs/cyl_det_eval/det_head_quick_v1_r60.json, docs/cyl/results_overview.md

- 20260130_pinhole_pose_depth_scale_superstrong_dist30_m32_cache
  - best ckpt: experiments/long_runs/20260130_pinhole_pose_depth_scale_superstrong_dist30_m32_cache/pinhole_pose_depth_scale/checkpoint-best.pth
  - val avg (pose_trans, pose_rot, depth_mae, scale_loss): ~0.45m, 0.86deg, 2.06m, 0.022
  - test avg (pose_trans, pose_rot, depth_mae, scale_loss): ~0.61m, 1.50deg, 1.73m, 0.017
  - evidence: experiments/long_runs/20260130_pinhole_pose_depth_scale_superstrong_dist30_m32_cache/pinhole_pose_depth_scale/log.txt

- 20260202_geom_stage2_pose_depth_scale_strong
  - best ckpt: experiments/long_runs/20260202_geom_stage2_pose_depth_scale_strong/pinhole_pose_depth_scale/checkpoint-best.pth
  - val avg (pose_trans, pose_rot, depth_mae, scale_loss): 0.3746m, 0.5757deg, 2.0331m, 0.0052
  - test avg (pose_trans, pose_rot, depth_mae, scale_loss): 0.5535m, 0.8774deg, 1.7847m, 0.0102
  - note: stable geometry checkpoint; used as init for later scale-only runs.
  - evidence: experiments/long_runs/20260202_geom_stage2_pose_depth_scale_strong/pinhole_pose_depth_scale/summary/summary.json

- 20260207_scale_only_direct_fix
  - best ckpt: experiments/local_runs/20260207_scale_only_direct_fix/pinhole_pose_depth_scale/checkpoint-best.pth
  - val avg (pose_trans, pose_rot, depth_mae, scale_loss): 10.0948m, 0.6631deg, 12.5350m, 0.0776
  - test avg (pose_trans, pose_rot, depth_mae, scale_loss): 11.2750m, 1.5051deg, 12.3182m, 0.1567
  - note: scale-only direct warmup; geometry metrics poor vs stage2.
  - evidence: experiments/local_runs/20260207_scale_only_direct_fix/pinhole_pose_depth_scale/log.txt

- 20260203_pinhole_pose_depth_scale_recover_v6 (active)
  - output dir (local): /tmp/mapanything_runs/20260203_pinhole_pose_depth_scale_recover_v6/pinhole_pose_depth_scale
  - output dir (nas): experiments/long_runs/20260203_pinhole_pose_depth_scale_recover_v6/pinhole_pose_depth_scale
  - init ckpt: experiments/long_runs/20260130_pinhole_pose_depth_scale_superstrong_dist30_m32_cache/pinhole_pose_depth_scale/checkpoint-best.pth
  - note: recovery run with reduced pose/scale weights; eval_freq=1; v5 failed due to NameError.
  - evidence: experiments/long_runs/20260203_pinhole_pose_depth_scale_recover_v6/pinhole_pose_depth_scale/log.txt

---

## PITFALLS & FIXES (quick reference)
- Documentation too coarse (paths without purpose/results): fix by linking eval metrics + visualizations and stating project significance.

- Conf loss NaNs on real data: use noconf wrapper or guard conf term before it returns NaN.
- Non-finite checkpoint params (scale token): detect before resume; repair or replace with known-good weights.
- Full-test NaN lines show scale_err_mean=0: treat as invalid; ignore for gating/summary.
- Over-strong pose loss or squared relative pose: can destabilize geometry; reduce weights or set squared=False.
- Small training set after max_agent_distance filter: expect higher variance; prefer conservative LR.
- Frequent NAS sync: can stall training; use local write + periodic sync (e.g., 120s).
- Badloss debug dump explosion: unbounded badloss_* artifacts -> disk fill; fix by cleanup + add dump cap.
- Hardcoded CUDA_VISIBLE_DEVICES on non-8-GPU machines: invalid device ordinal -> fix by GPU-count auto-detect or explicitly passing a valid GPU_LIST.
- Ephemeral instance reclaimed when entry command exits: detached rerun queues get killed -> fix by keeping entry alive until queue completion (hold hook) or running all jobs in one process tree.
- Repeated `frames_json` asset validation on already-verified full contracts can dominate startup and stall phase routing; fix by using `MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION=1` only for contracts already checked by the Phase 2 verifier.
- Reusing the same run_tag across repeated executions: eval artifacts get silently overwritten -> fix by adding attempt suffix (e.g. `__rerun2`) or snapshotting artifacts per attempt.

### 2026-03-06 18:12 +0800
Scope: Phase 2 closeout + first honest full2170 deploy-like det check on pinhole `promoted_v2`.
Key wins:
- Closed Phase 2 contract hardening: canonical contract verifier, det tier wrappers, and summary protocol validator are now in place.
- Produced the first protocol-valid full2170 deploy-like det evidence package via:
  - `map-anything/eval_runs/det_full2170_predpose_promoted_v2_fixalign_e10_20260306/summary_test.json`
  - validator PASS against `frames_test2170_nearest_seed42_full.json`.
Key risks:
- Honest full2170 deploy-like det still collapses badly for current pinhole mainline:
  - single `det_ap_iou=0.0054369495`
  - coop `det_ap_iou=0.0002192512`
- This confirms geometry quality alone is not bridging to collaborative detection under the locked protocol.
Next experiments:
- Run honest full2170 det for `sfix_direct`, `mid30_60_detach_pose12`, and `vggt_e30_best` under the same wrapper.
- Start full-data deploy-style det-head training using `opv2v_coop_det_ft_2a8v_pair_nearest_deploy_full`.

### 2026-03-06 18:12 +0800
Context: `promoted_v2` full2170 deploy-like det run completed under the Phase 2 wrapper.
Symptom:
- full2170 coop AP remained near zero (`0.0002192512`) despite protocol-valid metadata and full-contract coverage.
Root Cause (best guess):
- The current deploy det head was trained only on the short deploy dataset, and the geometry->det transfer for the current pinhole family is still fundamentally weak on full OPV2V.
Fix:
- Treat the result as a valid failure signal; move focus to (1) challenger full2170 reruns and (2) full-data deploy-style det-head training.
Validation:
- `python map-anything/scripts/validate_batch_eval_summary_protocol.py map-anything/eval_runs/det_full2170_predpose_promoted_v2_fixalign_e10_20260306/summary_test.json --expected-contract map-anything/eval_runs/frames_test2170_nearest_seed42_full.json --expected-model-task calibrated_sfm --expected-keep-camera-poses 0 --expected-keep-main-agent-poses 1 --require-det --require-rerun-file map-anything/eval_runs/det_full2170_predpose_promoted_v2_fixalign_e10_20260306/rerun_command.sh`
Next:
- Launch the next honest retry immediately and start the new full deploy training loop.

- full2170 honest deploy-like pinhole collapse: Test50 deploy-like ranking was not enough -> fix by treating full2170 protocol-valid runs as the only route-ranking truth.

### 2026-03-07 03:18 +0800
Context: Phase 3 honest retry loop hit a real data-loader bottleneck on full-train deploy contracts, so routing was switched to immediate full2170 challenger reruns with the strongest existing det head.
Symptom:
- repeated attempts to start `opv2v_coop_det_ft_2a8v_pair_nearest_deploy_full` spent startup time in `OPV2VCoopDataset` contract loading before meaningful route feedback
- the dataset path uses `+ 20_000 @ OPV2VCoopDataset(...)`, making the full deploy det-head branch multi-hour per epoch at current throughput
Root Cause:
- `frames_json` contract loading was still performing strict per-asset validation on every frame/agent/camera even when the contract had already been verified by Phase 2 tooling; this became the dominant startup tax for large locked contracts
Fix:
- patched `map-anything/mapanything/datasets/opv2v.py` so strict contract asset validation remains the default, but can be explicitly skipped with `MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION=1` for already-verified contracts
- smoke-checked the exact full-train contract under that env and confirmed loader init finishes quickly
- redirected GPU time from the slow full-train branch to an honest full2170 challenger queue using the strongest existing deploy det head:
  - det head: `map-anything/experiments/local_runs/20260227_det_headonly_deploy_predpose_aligngt0_fixalign_from_promotedv2_e10_v1/checkpoint-final.pth`
  - queue root: `map-anything/experiments/local_runs/20260306_honest_retry_chain_fixalign_e10_existing_head`
Validation:
- contract smoke: `[INFO] OPV2V frames_json contract hit: ... frames_train6374_nearest_seed42_full.json (frames=6374, split_filtered=0, asset_validation=skipped).`
- loader smoke: `loader_len 6374 elapsed_sec 0.175`
- active honest queue entered real evaluation on full2170:
  - `map-anything/eval_runs/det_full2170_predpose_sfix_direct_with_trained_fullhead_20260306_honest_retry_chain_fixalign_e10_existing_head`
  - queue progress reached `seen=100/4340`
Next:
- let `sfix_direct` full2170 honest eval finish and validate protocol
- continue automatically to `mid30_60_detach_pose12` and `vggt_e30_best`
- compare all three against the existing `promoted_v2` honest failure and route the next training budget only after those full2170 numbers land



### 2026-03-07 03:51 +0800
Context: Phase 3 honest retry loop on 1x4090 after the first protocol-valid `promoted_v2` full2170 failure.
Symptom:
- Full-data det-head-only retraining initially looked dead at `Building Train Data loader for dataset:` and one earlier launch had already failed because a relative `PRETRAIN_CKPT` path was resolved from the wrong working directory.
Root Cause:
- Two separate issues were mixed together:
  1. launcher accepted relative checkpoint paths like `map-anything/...`, which broke once the script `cd`'d into repo root;
  2. full deploy train/test loader construction on OPV2V nearest full contracts is extremely slow on this host, so `num_workers=0` can still be healthy even after a long apparent stall.
Fix:
- Canonicalized `PRETRAIN_CKPT` to an absolute path inside `map-anything/bash_scripts/train/run_pinhole_det_head_only_local_single.sh`.
- Hardened `map-anything/scripts/queue_full2170_honest_retries_after_training.sh` to fail loudly when no trained checkpoint exists and to suffix output directories with the training run basename.
- Stopped blind relaunching and instead kept the active honest challenger queue alive while arming a sequential follow-on supervisor for the next training/eval wave.
Validation:
- Existing-head honest challenger queue is actively progressing at `map-anything/experiments/local_runs/20260306_honest_retry_chain_fixalign_e10_existing_head/queued_retries.log`.
- Active out dir for the first challenger: `map-anything/eval_runs/det_full2170_predpose_sfix_direct_with_trained_fullhead_20260306_honest_retry_chain_fixalign_e10_existing_head`.
- Follow-on supervisor is alive at `map-anything/experiments/local_runs/20260306_followon_after_existinghead_queue_r8/supervisor.pid` with log `map-anything/experiments/local_runs/20260306_followon_after_existinghead_queue_r8/supervisor.log`.
Next:
- Let the existing-head `sfix_direct -> mid30_60_detach_pose12 -> vggt_e30_best` honest full2170 queue finish.
- Immediately after that, execute the queued one-epoch full-data det-head continuation from the short deploy det-head checkpoint and run its own honest full2170 + challenger chain.

### 2026-03-07 03:58 +0800
Context: full-data deploy det-head retrain looked hung at `Building Train Data loader for dataset:` on the locked nearest full-train contract.
Symptom:
- `run_pinhole_det_head_only_local_single.sh` stalled for a long time before `train dataloader built`, even with `dataset.num_workers=0`.
Root Cause:
- The bottleneck was not `DataLoader` worker fork. `OPV2VCoopDataset` was doing strict asset validation for every YAML/image/depth path in `frames_train6374_nearest_seed42_full.json`, causing a massive repeated `stat()` burst during `eval(dataset)`.
Fix:
- For the contract-verified full deploy det-head wrapper, export `MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION=1` by default in `map-anything/bash_scripts/train/run_pinhole_det_head_only_local_single.sh`.
- Added guarded debug tracing (`MAPANYTHING_DEBUG_DATALOADER=1`) around train dataloader/sampler construction to pinpoint future stalls cheaply.
Validation:
- `strace -f -p <stuck_pid> -e trace=%file` on `/tmp/ma_train_r7` showed repeated `stat()` over OPV2V train asset paths.
- Re-run with skip enabled reached `train dataloader built`, `init_model start`, `Start training for 1 epochs`, and `Epoch: [0] [    0/20000]` in `/tmp/ma_train_r8/pinhole_det_head_only/launch.log`.
- Official run `map-anything/experiments/local_runs/20260306_deploy_full_dethead_e3_promotedv2_single_r10_skipasset/pinhole_det_head_only/launch.log` now also reaches `Epoch: [0] [    0/20000]`.
Next:
- Let `20260306_deploy_full_dethead_e3_promotedv2_single_r10_skipasset` continue and then consume the queued honest full2170 challenger evals.

- full-train nearest contract apparent “dataloader hang”: often strict asset validation over `frames_json`, not a deadlock -> fix by using `MAPANYTHING_SKIP_CONTRACT_ASSET_VALIDATION=1` once the contract has already been verified out-of-band.
