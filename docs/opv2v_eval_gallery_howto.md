# OPV2V Eval 可视化画廊：怎么看 + 怎么重跑（Metric Viz / Det Compose）

本 repo 里常见的两类离线画廊：

- `eval_runs/metric_viz_test500_*/index.html`：**几何/pose/scale/深度** 的“可读 BEV 对比画廊”（多模型并排）。
- `eval_runs/det_compose_*/bev_png/index.html`：**检测头 compose eval** 的代表帧 BEV PNG 画廊（single/coop 分组）。

下文解释这两类图的颜色/标注含义，并给出一键重跑命令。

---

## 1) `metric_viz_test500_*_readable*`：Readable Metric Viz（多模型并排）

### 1.1 结构（HTML）
- **行（Rows）= 帧**：从 selector 模型的 per-frame 指标里选出的代表帧（best / median / worst）。
- **列（Cols）= 模型**：同一帧在不同模型下的 BEV 叠加结果。
- **Frame 单元格**：显示 `metric:tag (value)`，以及 best/median/worst 的 badge。
- **Tile（每张图）**：左下角有一块等宽字体注释框（也是 HTML tooltip/caption）。

### 1.2 Tile 里画的是什么（PNG）
- **GT LiDAR**：浅灰点（主车 ego frame；`--gt_lidar_mode=fused` 时是 main+coop 融合到主车系）。
- **GT boxes**：黑色框（车辆真值框）。
- **Pred points（重点）**：默认推荐用 `--pred_color_by match` 或 `agent_match`：
  - `match`：绿色 = 该 BEV 网格 cell 与 GT 有重叠；红色 = 没重叠（更容易看出“哪里错了”）。
  - `agent_match`（仅 `--pred_pcd_strategy infer` 可用）：按 **agent id** 上色（匹配点），并用红色标出 mismatch；更容易看出 **coop 位姿漂移导致的“双影/错位”**。
  - `z`：按高度 z（turbo）上色（保留结构感，但不如 match 直观）。

### 1.3 读注释框（on-tile notes）
常见字段（不同 run 可能缺一些）：
- `sel=metric/tag`：该帧被选中的原因（例如 `sel=cross_agent_pose_trans_m/worst`）。
- `pose_abs/pose_rot`：多视角 pose 平均误差（越小越好；coop 时也包含跨 agent 的影响）。
- `scale_abs/scale_mult/ratio_gt`：scale 相关（`scale_mult≈1` / `ratio_gt≈1` 越好）。
- `cross_t/cross_r`：跨 agent 相对 pose 的误差（coop 关键指标）。
- `depth_rel`：深度相对误差（越小越好）。
- `chamfer_f / bev_iou_f`：点云对齐程度（越好越小/越大；取决于指标定义）。
- `baseline`：主车与协同车的 GT 距离（用于判断“远车是否在视野/统计范围内”）。
- `bev_match=XX%`：Pred 与 GT 在 BEV 网格上的重叠比例（`--match_res` 控制网格分辨率；这是**快速直观信号**，不是严格指标）。

### 1.4 快速判读建议
- **看红色（mismatch）分布**：红色越集中/越成片，通常意味着 pose/scale 有系统性漂移。
- **coop misalignment**：典型表现是路沿/建筑轮廓出现“平移双影”；`agent_match` 会更明显。
- **scale 问题**：整体形状相似但“膨胀/收缩”；`scale_mult/ratio_gt` 会明显偏离 1。

---

## 2) `det_compose_*`：Det BEV PNG Gallery（single/coop 分组）

入口：`eval_runs/det_compose_*/bev_png/index.html`

每张图是 `scripts/viz_batch_eval_bev_png.py` 画出来的 2D BEV 叠加：
- **GT LiDAR**：浅灰点。
- **Pred points**：
  - 默认 `--pred_point_color_by auto`：如果能找到 GT 点云，则用 `match`（绿/红）上色；
  - 否则退化为单色点（plain）。
- **GT boxes**：黑色框。
- **Pred boxes**（当启用 `--color_pred_by_match`）：
  - 绿色：TP
  - 橙色：FP
  - 红色虚线 GT 框：FN
- 左上角文本框：`TP/FP/FN` + `IoU>=thr`，若启用点云 match 也会附带 `bev_match=XX%`。

---

## 3) 怎么重跑（推荐命令）

### 3.1 重跑 Readable Metric Viz（优先：复用已有 pred_pcd）
适用：你只改了画图风格（颜色/标注），不想重新推理。

```bash
cd map-anything
python scripts/metric_viz_readable_bev_gallery.py \
  --models nm4_base=eval_runs/geom_fairlock_test500_20260226_nm4_base \
          promoted_v2=eval_runs/geom_fairlock_test500_20260226_promoted_v2 \
          vggt_coop_e30_best=eval_runs/geom_fairlock_test500_20260226_vggt_coop_e30_best \
  --mode coop \
  --metrics scale_to_gt_mult_err cross_agent_pose_trans_m cross_agent_pose_rot_deg depth_rel \
  --selector_model nm4_base \
  --out_dir eval_runs/metric_viz_test500_20260226_readable_all_models \
  --pred_pcd_strategy reuse \
  --reuse_pcd_root eval_runs/metric_viz_test500_20260226_readable_all_models/pred_pcd \
  --pred_color_by match \
  --match_res 0.5 \
  --write_plots
```

### 3.2 重跑 Readable Metric Viz（需要：按 agent 上色）
适用：你想让 coop misalignment “一眼就能看出是哪个 agent 的点在漂”。

这需要 `--pred_pcd_strategy infer`（因为需要 per-view / per-agent 的点来源信息）：

```bash
cd map-anything
python scripts/metric_viz_readable_bev_gallery.py \
  --models nm4_base=eval_runs/geom_fairlock_test500_20260226_nm4_base \
          promoted_v2=eval_runs/geom_fairlock_test500_20260226_promoted_v2 \
          vggt_coop_e30_best=eval_runs/geom_fairlock_test500_20260226_vggt_coop_e30_best \
  --mode coop \
  --metrics scale_to_gt_mult_err cross_agent_pose_trans_m cross_agent_pose_rot_deg depth_rel \
  --out_dir eval_runs/metric_viz_test500_20260226_readable_all_models \
  --pred_pcd_strategy infer \
  --pred_color_by agent_match \
  --match_res 0.5
```

### 3.3 重跑 det_compose 的 BEV PNG 画廊
```bash
cd map-anything
python scripts/make_det_bev_png_gallery.py \
  --eval_root eval_runs/det_compose_predpose_aligngt0_test50_20260227_fixalign
```

### 3.4 单张图快速导出（调试用）
```bash
cd map-anything
python scripts/viz_batch_eval_bev_png.py \
  --pred_pcd <SOME>.pcd \
  --yaml data/opv2v/test/<SEQ>/<MAIN_AGENT>/<FRAME>.yaml \
  --out_png /tmp/bev.png \
  --pred_point_color_by match \
  --match_res 0.5 \
  --color_pred_by_match
```

