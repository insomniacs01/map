# OPV2V：当前各方案效果与可视化索引（持续更新）

## 更新记录

| 日期 (UTC) | 内容 | 备注 |
| --- | --- | --- |
| 2026-01-20 | 在开头新增 TL;DR：路线总览、评测口径对比、评测维度/指标释义与现状解读 | 数值来自 `eval_runs/**/summary*.json` 与 `eval_runs/demo_alignment/metrics_*.json` |

## TL;DR

一页速览：路线现状 + 评测口径差异 + 指标释义 + 当前解读。详细数据与可视化路径见后文 A/B/C。

### 路线总览（当前代表结果）

| 线/方向 | 目的 | 代表指标（当前最佳） | 可视化 |
| --- | --- | --- | --- |
| A 圆柱协同重建/位姿对齐 | 多车圆柱 pano 协同重建 + 相对位姿对齐 | pose_err_rel_trans_mean≈0.715m、pose_err_rel_rot_mean≈0.805°、chamfer_predpose_mean≈0.565、bev_iou_predpose_mean≈0.419（`eval_runs/demo_alignment/metrics_pose_r60_v1_recheck.json`） | `eval_runs/demo_alignment/gt_validate_2021_09_11_00_33_16_1016_000275.html`；`eval_runs/demo_alignment/pred_bev_pose_r60_v1_validate_2021_09_11_00_33_16_1016_000275_fusedgt_v2.png` |
| A' 圆柱协同+检测头（quick/overfit） | 特定序列快速对齐 + 检测可视化验证 | pose_err_rel_trans_mean≈0.0556m、pose_err_rel_rot_mean≈0.1527°；det_ap_iou=1.000、det_precision_iou=0.909、det_recall_iou=1.000、det_mean_iou≈0.899（`eval_runs/pose_metrics/opv2v_cyl_geom_v1_207_r60.json` + `eval_runs/cyl_det_eval/det_head_quick_v1_r60.json`） | `eval_runs/cyl_det_viz/2021_08_24_20_49_54_000386_det_v1.html` |
| A'' 圆柱协同+检测头（test50 coop finetune） | 与 B1/B2 同一 50 帧，coop 评测 | det_ap_iou=0.4254、det_precision_iou=0.1311、det_recall_iou=0.5585、det_mean_iou≈0.743（score_thresh=0.05；`eval_runs/cyl_det_eval/det_head_test50_coop_v3_hardneg_t05_imgonly.json`） | 可视化采用 score_thresh=0.2：`eval_runs/cyl_det_viz/2021_08_24_20_49_54_000386_det_test50_coop_v3_t20.html` |
| A''' 圆柱协同+检测头（test50 single） | 与 B1/B2 同一 50 帧，single 评测 | det_ap_iou=0.0017、det_precision_iou=0.0095、det_recall_iou=0.0256、det_mean_iou≈0.599（`eval_runs/cyl_det_eval/det_head_test50_single_v2.json`） | `eval_runs/cyl_det_viz/2021_08_24_20_49_54_000386_det_test50_single_v2.html` |
| B1 传统 OPV2V + BEV 检测头（single） | 单车 BEV 检测 | det_ap_iou=0.199、det_precision_iou=0.194、det_recall_iou=0.228、det_mean_iou=0.697（`eval_runs/det_e2e_v5_eval_t005/summary_test.json`） | `eval_runs/det_e2e_v5_eval_t005/html/index.html` |
| B2 传统 OPV2V + BEV 检测头（coop） | 协同 BEV 检测 | det_ap_iou=0.138、det_precision_iou=0.189、det_recall_iou=0.213、det_mean_iou=0.665、pose_abs_mean≈17.745m/pose_rot_mean≈11.723°（`eval_runs/det_e2e_v5_coop_eval_t005/summary_test.json`） | `eval_runs/det_e2e_v5_coop_eval_t005/html/index.html` |
| C 传统 OPV2V 协同重建 baseline（stage2） | 针孔协同重建基线 | pose_abs_mean≈10.631m/pose_rot_mean≈6.270°、chamfer_pred_to_gt_mean≈2.809、bev_iou_raw_mean≈0.188（`eval_runs/stage2_baseline_eval/summary_test.json`） | 暂无 HTML/PNG；仅 `eval_runs/stage2_baseline_eval/stage2/coop_metrics.csv` |

### 评测口径对比（为什么不能直接横向比）

| 线/方向 | 评测脚本 | split | 采样/帧数 | agent/view 选择 | GT/视野裁剪 | 口径能反映什么 |
| --- | --- | --- | --- | --- | --- | --- |
| A 圆柱协同 | `scripts/eval_opv2v_cyl_metrics.py` | validate | 固定序列 + 5 帧均值 | num_views=4、max_agent_distance=60、agent_selection_policy=nearest | GT LiDAR=fused；BEV crop radius_max=60、z∈[-3,3] | 强制与可视化同口径，区分 pose 与几何（PredPose vs GTPose） |
| A'' 圆柱检测 | `scripts/eval_opv2v_cyl_det_metrics.py` | test | `summary_test.json` 里 50 帧 | num_views=2/1、agent_selection_policy=nearest | det cfg 与 B1/B2 对齐 | 与 B1/B2 同帧的圆柱 det 评估 |
| B1/B2 检测 | `scripts/batch_eval.py` | test | 随机抽样 50 帧 | single/coop 分开统计 | IoU 阈值下的 BEV 检测评估；点云/深度指标按脚本默认 | 反映 BEV 检测性能与整体几何质量，易受解码/阈值影响 |
| C stage2 baseline | `scripts/batch_eval.py`（stage2/coop） | test | 随机抽样 20 帧 | coop | 同上 | 传统针孔协同重建基线，用于对照 |

### 评测维度 & 指标释义（字段含义）

| 维度 | 指标字段 | 含义 | 适用线路 |
| --- | --- | --- | --- |
| Pose | pose_err_rel_trans_mean / pose_err_rel_rot_mean | 相对位姿误差：以 view0 为参考，比较 inv(T0)@Ti 的平移/旋转误差（m/°） | A |
| Pose | pose_abs_mean / pose_rot_mean | 参考视角坐标系下的平移/旋转绝对误差均值（m/°） | B/C |
| Geometry | chamfer_predpose_mean / chamfer_gtpose_mean | 预测点云与 GT LiDAR 的 Chamfer；PredPose=用预测 pose 放置，GTPose=用 GT pose 放置 | A |
| Geometry | chamfer_pred_to_gt_mean / chamfer_gt_to_pred_mean | 预测点云与 GT LiDAR 的双向 Chamfer | B/C |
| Geometry | bev_iou_predpose_mean / bev_iou_gtpose_mean | BEV 栅格占据 IoU（PredPose/GTPose） | A |
| Geometry | bev_iou_raw_mean / bev_iou_filtered_mean | BEV 占据 IoU（原始/过滤） | B/C |
| Detection | det_ap_iou / det_precision_iou / det_recall_iou / det_mean_iou | 单一 IoU 阈值下 AP/精度/召回/匹配平均 IoU | B |
| Depth | depth_rmse_mean / depth_mae_mean / depth_rel_mean | 深度误差（RMSE/MAE/相对误差） | B/C |
| Scale (OPV2V 主口径) | scale_to_gt_err_mean / scale_to_gt_log_err_mean / scale_to_gt_ratio_mean | ratio-to-GT 尺度误差（`s/g`）；目标：err→0，ratio→1 | B/C |
| Scale (legacy，仅兼容) | scale_err_mean / scale_ratio_mean | legacy `|s-1|`（隐含 GT=1 假设）；不用于排名/验收 | B/C |
| Density | pts_in_box_*_median_mean | GT 框内点数中位数（用于 sanity check） | A |

### 现状解读（基于当前结果）

- A 线 pose 已到亚米/亚度级，对齐基本成立；剩余误差主要来自几何细节与点云稠密度。
- A' 线是特定序列的 quick finetune（5 帧），指标极高但不可与 B/C 抽样评测直接对比；适合验证“流程/坐标系/框对齐”是否闭环。
- A'' 在 test50 finetune 后 det AP≈0.43（仅说明“该 50 帧可拟合”），A''' 仍接近 0；泛化能力仍不足。
- B 线 single 明显优于 coop；coop 的 pose_abs_mean≈17.7m/11.7° 表明协同 pose 未对齐是检测 AP 的主要瓶颈。
- C 线作为针孔协同重建基线，Chamfer≈2.81、BEV IoU≈0.188，适合作为后续优化对照。

本文把仓库里已经跑出来的“可复现结果”按方案归类，并给出：
- 评测脚本/评测口径
- 关键指标（能反映“是否真的对齐/是否真的检测到”）
- 对应可视化产物路径（PNG/HTML）

> 注意：不同方案的评测口径不同（例如圆柱协同 vs 传统 OPV2V），数字不能直接横向比较；请按各自章节里的“口径说明”解读。

---

## A. 圆柱协同重建/位姿对齐（OPV2VCoopCylindricalDataset）

### A.0 评测口径（固定）

脚本：`scripts/eval_opv2v_cyl_metrics.py`

关键点：统一“采样规则”和“可视化视野”，避免出现“图对但数爆/图爆但数对”的错觉。

当前 demo 口径（用于对比不同 ckpt）：
- split: `validate`
- sequence: `2021_09_11_00_33_16`
- main_agent: `1016`
- frames: `000269,000271,000273,000275,000277`（5 帧均值）
- num_views: `4`
- max_agent_distance: `60`（只评估 60m 内的协同车）
- agent_selection_policy: `nearest`
- gt_lidar_mode: `fused`
- BEV crop: `radius_max=60, z_min=-3, z_max=3`
- mask_thresh: `0.5`

可视化（同口径）：`scripts/viz_opv2v_pred_bev_png.py`（PredPose vs GTPose 两列）

### A.0' Quick finetune 口径（仅用于流程验证，非泛化评测）

用于验证“pose/几何/检测框对齐是否闭环”，不与 A/B/C 的抽样评测横向比。

- split: `test`
- sequence: `2021_08_24_20_49_54`
- main_agent: `207`
- frames: `000380,000382,000384,000386,000388`（5 帧均值）
- num_views: `2`
- max_agent_distance: `60`
- agent_selection_policy: `nearest`
- det_score_thresh: `0.05`

可视化（含 Pred det boxes + GT boxes）：`eval_runs/cyl_det_viz/2021_08_24_20_49_54_000386_det_v1.html`

### A.0'' Test50 口径（与 B1/B2 同一 50 帧，可横向对比）

用于衡量圆柱 det head 的泛化检测能力，与针孔 B1/B2 使用同一帧列表。
当前 A'' 结果来自对这 50 帧的 det-head-only finetune（400 steps，hard-neg 版），用于验证“可拟合”，不代表泛化性能。

- split: `test`
- frames_json: `eval_runs/det_e2e_v5_eval_t005/summary_test.json`
- num_views: `2`（coop）或 `1`（single）
- agent_selection_policy: `nearest`
- det_head_cfg: `bev_centernet_wide_v7_gate_noconf`
- det_score_thresh: `0.05`
- model/task: `images_only`

输出：
- coop 指标（score_thresh=0.05）：`eval_runs/cyl_det_eval/det_head_test50_coop_v3_hardneg_t05_imgonly.json`
- coop 指标（score_thresh=0.2）：`eval_runs/cyl_det_eval/det_head_test50_coop_v3_hardneg_t20_imgonly.json`
- coop 可视化（score_thresh=0.2）：`eval_runs/cyl_det_viz/2021_08_24_20_49_54_000386_det_test50_coop_v3_t20.html`
- single 指标（旧）：`eval_runs/cyl_det_eval/det_head_test50_single_v2.json`
- single 可视化（旧）：`eval_runs/cyl_det_viz/2021_08_24_20_49_54_000386_det_test50_single_v2.html`

### A.1 GT 基准（必须先看：框和点云确实对应）

- GT 3D HTML：`eval_runs/demo_alignment/gt_validate_2021_09_11_00_33_16_1016_000275.html`

### A.2 关键方案对比（同口径）

#### (1) det_e2e ckptbest（圆柱上用于对照的“坏例子”）

- ckpt: `experiments/opv2v_coop_det_e2e/checkpoint-best.pth`
- metrics: `eval_runs/demo_alignment/metrics_det_e2e_ckptbest_r60.json`
  - pose_err_rel_mean≈`36.50m / 63.96°`
  - chamfer(predpose)≈`1.65`，bev_iou(predpose)≈`0.237`
- vis: `eval_runs/demo_alignment/pred_bev_det_e2e_ckptbest_validate_2021_09_11_00_33_16_1016_000275_fusedgt.png`

判读要点：PredPose 与 GTPose 都不理想，属于“pose+几何都没进正确 basin”。

#### (2) geom_lidar_v1（LiDAR 投影监督几何：几何好，但 pose 很差）

- ckpt: `experiments/opv2v_cyl_geom_quick/checkpoint-validate_2021_09_09_19_27_35_2314_geom_lidar_v1.pth`
- metrics: `eval_runs/demo_alignment/metrics_geom_lidar_v1_r60_baseline.json`
  - pose_err_rel_mean≈`112.54m / 43.53°`（PredPose 会飞）
  - chamfer(gtpose)≈`0.44`，bev_iou(gtpose)≈`0.426`（GTPose 下几何接近 OK）
- vis: `eval_runs/demo_alignment/pred_bev_geom_lidar_v1_validate_2021_09_11_00_33_16_1016_000275_fusedgt.png`

判读要点：**GTPose 好、PredPose 差 → 主要是 pose 问题**。

#### (3) pose_r60_v1（只修 pose：当前“最对齐”的那版）

- ckpt: `experiments/opv2v_cyl_pose_quick/checkpoint-validate_2021_09_11_00_33_16_1016_pose_r60_v1.pth`
- metrics: `eval_runs/demo_alignment/metrics_pose_r60_v1_recheck.json`
  - pose_err_rel_mean≈`0.715m / 0.805°`（已对齐）
  - chamfer(predpose)≈`0.565`，bev_iou(predpose)≈`0.419`
  - chamfer(gtpose)≈`0.438`，bev_iou(gtpose)≈`0.426`（PredPose≈GTPose）
- vis: `eval_runs/demo_alignment/pred_bev_pose_r60_v1_validate_2021_09_11_00_33_16_1016_000275_fusedgt_v2.png`

判读要点：PredPose≈GTPose，说明 pose 已基本正确；剩余误差主要来自几何细节/点云稠密度。

#### (6) pose+geom+det quick（本次序列 2021_08_24_20_49_54）

- ckpt: `experiments/opv2v_cyl_det_head_quick/checkpoint-test_2021_08_24_20_49_54_207_det_v1.pth`
- metrics（同 A.0'）：  
  - pose_err_rel_trans_mean≈`0.0556m`、pose_err_rel_rot_mean≈`0.1527°`  
    (`eval_runs/pose_metrics/opv2v_cyl_geom_v1_207_r60.json`)  
  - det_ap_iou=`1.000`、det_precision_iou=`0.909`、det_recall_iou=`1.000`  
    (`eval_runs/cyl_det_eval/det_head_quick_v1_r60.json`)
- vis: `eval_runs/cyl_det_viz/2021_08_24_20_49_54_000386_det_v1.html`

判读要点：该结果是 **5 帧 quick finetune**，用于验证“框和 GT 对齐”与流程正确性，不代表泛化性能。

#### (4) ddp_smoke_r60_v8（DDP 训练的失败例）

- ckpt: `experiments/opv2v_cyl_coop_ddp_smoke_r60_v8/checkpoint-final.pth`
- metrics: `eval_runs/demo_alignment/metrics_ddp_smoke_r60_v8.json`
  - pose_err_rel_mean≈`40.30m / 118.50°`
  - chamfer(predpose)≈`10.15`，bev_iou(predpose)≈`0.015`
- vis: `eval_runs/demo_alignment/pred_bev_ddp_smoke_r60_v8_validate_2021_09_11_00_33_16_1016_000275_fusedgt.png`

#### (5) pose_only_ddp_r60_demo（DDP pose-only 的失败例）

- ckpt: `experiments/opv2v_cyl_pose_only_ddp_r60_demo/checkpoint-final.pth`
- metrics: `eval_runs/demo_alignment/metrics_pose_only_ddp_r60_demo.json`
  - pose_err_rel_mean≈`39.43m / 41.33°`
  - chamfer(predpose)≈`1.736`，bev_iou(predpose)≈`0.220`
- vis: `eval_runs/demo_alignment/pred_bev_pose_only_ddp_r60_demo_validate_2021_09_11_00_33_16_1016_000275_fusedgt.png`

---

## B. 传统 OPV2V（针孔）+ BEV 检测头（MapAnythingDet）

### B.0 评测口径

脚本：`scripts/batch_eval.py`

常见设置（这些 run 里基本一致）：
- split: `test`
- 抽样 `frames=50`（非全量）
- det_ap_iou：单一 IoU 阈值（通常 iou_thresh=0.5）的 AP（不是 COCO-style mAP）
- 同时会输出 pose/depth/scale 的均值（但它们和圆柱协同的 pose_err_rel_mean 不是一回事）

可视化产物：
- Plotly 交互 HTML：`eval_runs/<run>/html/<model>/<mode>/*_{best,median,worst}.html`
- 同目录 `index.html` 可快速跳转
- 评测中会保存代表帧的预测点云/预测框：`eval_runs/<run>/<model>/<mode>_representatives/*.(pcd|json)`

### B.1 单车检测（single）最佳/代表结果

目前在本机 eval_runs 里，single 模式 AP 最高的是（v5：更强 hard-neg + vehicle loss 不回传 scale）：

- run: `eval_runs/det_e2e_v5_eval_t005/summary_test.json`
  - det_ap_iou=`0.1992`，precision=`0.1942`，recall=`0.2280`，mean_iou=`0.6975`
  - vis index: `eval_runs/det_e2e_v5_eval_t005/html/index.html`
  - slim（更少框，便于肉眼对齐）: `eval_runs/det_e2e_v5_eval_t005/html_slim/index.html`
  - best/worst/median: `eval_runs/det_e2e_v5_eval_t005/html/e2e_v5/single/`

上一版对照（v3，decode tuned）：

- run: `eval_runs/tune_e2e_v3_detdecode_t001_den03/summary_test.json`
  - det_ap_iou=`0.1884`，precision=`0.0430`，recall=`0.2622`，mean_iou=`0.6732`
  - vis index: `eval_runs/tune_e2e_v3_detdecode_t001_den03/html/index.html`
  - best/worst/median: `eval_runs/tune_e2e_v3_detdecode_t001_den03/html/e2e_v3/single/`

另一个更“稳阈值”的对照：

- run: `eval_runs/det_e2e_posegeom_det_v3_strongdet_eval_t005/summary_test.json`
  - det_ap_iou=`0.1764`，precision=`0.1766`，recall=`0.2134`，mean_iou=`0.6922`
  - vis index: `eval_runs/det_e2e_posegeom_det_v3_strongdet_eval_t005/html/index.html`
  - best/worst/median: `eval_runs/det_e2e_posegeom_det_v3_strongdet_eval_t005/html/e2e_v3/single/`

> 备注：上述两者的模型/几何指标几乎一致，主要差异来自“解码/后处理参数”（score_thresh、support filtering、NMS 等）对 AP 的影响。

### B.2 协同检测（coop）现状

coop 模式目前明显弱于 single，且 pose_abs_mean 很大（说明协同 pose 没对齐，直接拖垮 det）。

当前 coop 模式 AP 最高的是：

- run: `eval_runs/det_e2e_v5_coop_eval_t005/summary_test.json`
  - det_ap_iou=`0.1380`，precision=`0.1892`，recall=`0.2134`，mean_iou=`0.6652`
  - vis index: `eval_runs/det_e2e_v5_coop_eval_t005/html/index.html`
  - slim: `eval_runs/det_e2e_v5_coop_eval_t005/html_slim/index.html`
  - pose_abs_mean≈`17.745m`，pose_rot_mean≈`11.723°`

上一版对照（v3）：

- run: `eval_runs/tune_e2e_v3_coop_den03/summary_test.json`
  - det_ap_iou=`0.1271`，precision=`0.0372`，recall=`0.2268`，mean_iou=`0.6573`
  - pose_abs_mean≈`17.867m`，pose_rot_mean≈`11.957°`

---

## C. 传统 OPV2V 协同（针孔）重建/融合基线（stage2）

- run: `eval_runs/stage2_baseline_eval/summary_test.json`
  - coop: frames=20
  - pose_abs_mean≈`10.63m`，pose_rot_mean≈`6.27°`
  - chamfer_pred_to_gt_mean≈`2.81`，bev_iou_raw_mean≈`0.188`
- 产物：`eval_runs/stage2_baseline_eval/stage2/coop_metrics.csv`

---

## 快速打开可视化（无 GUI）

HTML 建议用简易静态服务器：
```bash
cd map-anything
python -m http.server 8000
```
然后在浏览器访问相应 `eval_runs/.../index.html`（或把端口 ssh 转发到本地）。
