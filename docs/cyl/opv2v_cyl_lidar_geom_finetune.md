# OPV2V 圆柱协同：LiDAR 投影监督的几何微调（收敛打法）

> 目标：当 MapAnything 在 OPV2V 上出现“预测点云扇形/长射线/尺度爆炸/框内没点”等现象时，用一套可复现的 **GT 基准 → 定位 root cause → 快速微调 → 反复可视化闭环** 流程，把几何（ray dirs / depth / scale / mask）收敛到“预测点云≈GT 点云”，并可扩展到多车（>2 agents）协同重建。

## 0. 两个必须先统一的事实

1. **GT LiDAR + GT boxes 是唯一“肉眼可判”的坐标基准**  
   先把真值点云与真值框叠起来，确认“框里确实有点”“rings 形态正常”。否则后续所有判断都不可信。
2. **坐标系坑：OPV2V=CARLA，MapAnything=OpenCV**  
   OPV2V `.pcd`：CARLA/UE（X forward, Y right, Z up）  
   MapAnything 内部几何：OpenCV（X right, Y down, Z forward）  
   变换矩阵定义在 `data_processing/opv2v_pose_utils.py`（`CARLA_TO_CAMERA_CV`）。

## 0.1 指标和可视化“对不上”的常见原因：统计范围没对齐

一个非常隐蔽但高频的坑：**你在 BEV 可视化里只看 `radius_max=60m` 的视野**，但 `pose_err_rel_mean` 之类的指标可能在统计时**把更远的协同车（100–300m）也算进去了**。

结果就是：
- 你肉眼看着 “60m 内点云/框已经对齐了”，感觉 pose OK；
- 但指标 `pose_err_rel_mean` 依然很大（因为被视野外的远车拉爆）。

解决原则：**训练采样范围、可视化范围、指标统计范围必须一致**。

本 repo 已在 OPV2V 圆柱协同数据集里加入了距离过滤：
- `max_agent_distance`：只保留主车坐标系 XY 距离 <= R 的协同车（建议与 `radius_max` 取同值）
- `agent_selection_policy=nearest`：当可用 agent > `num_views` 时优先选最近的协同车（更稳定）

## 0.2 推荐的“可视化=指标”一键评估脚本

脚本：`scripts/eval_opv2v_cyl_metrics.py`

它会强制使用同一套：
- agent 选择（`num_views/max_agent_distance/agent_selection_policy`）
- 点云裁剪（`z_min/z_max/radius_max`）

并输出：
- `pose_err_rel_mean`（相对位姿误差，单位 m/°，参考 view0）
- Chamfer / BEV IoU（分别在 PredPose 与 GTPose 两种放置方式下统计）

示例（validate / 近距离多车，建议用于 sanity check）：
```bash
PYTHONPATH=$(pwd) python scripts/eval_opv2v_cyl_metrics.py \
  --checkpoint <CKPT> \
  --split validate --sequence 2021_09_11_00_33_16 --main_agent 1016 \
  --frames 000269,000271,000273,000275,000277 --num_views 4 \
  --max_agent_distance 60 --agent_selection_policy nearest --gt_lidar_mode fused \
  --radius_max 60 --z_min -3 --z_max 3 \
  --out_json eval_runs/demo_alignment/metrics_<TAG>.json
```

这一段推荐的“基准对比”（同设置，GT=fused，radius=60m）：
- `geom_lidar_v1`（未做近距离 pose 微调）：`pose_err_rel_mean≈112.5m/43.5°`，`Chamfer(predpose)≈0.855`，`BEV IoU(predpose)≈0.263`
- `pose_r60_v1`（只做 pose quick finetune）：`pose_err_rel_mean≈0.715m/0.805°`，`Chamfer(predpose)≈0.564`，`BEV IoU(predpose)≈0.420`

## 1. “先看什么”的最短闭环

### 1.1 GT 基准可视化（必须先过）

- HTML（交互 3D/BEV）：  
  `PYTHONPATH=$(pwd) python scripts/viz_opv2v_gt_pcd_boxes_html.py --split validate --sequence <SEQ> --agent <MAIN_AGENT> --frame <FRAME>`
- PNG（快速 BEV）：  
  `python scripts/viz_batch_eval_bev_png.py --pred_pcd data/opv2v/validate/<SEQ>/<MAIN_AGENT>/<FRAME>.pcd --yaml data/opv2v/validate/<SEQ>/<MAIN_AGENT>/<FRAME>.yaml --out_png eval_runs/demo_alignment/gt_bev_<SEQ>_<MAIN_AGENT>_<FRAME>.png`

判据（至少满足其一）：  
- rings 形态明显；  
- 车辆框内有足够点（不是稀稀拉拉几粒点）。

### 1.2 预测可视化：分离 Pose 与 Geometry 的影响

核心思路：同一份预测，用两种方式放置点云：

- **PredPose**：直接用 `pred['pts3d']`（模型自己预测的 pose 放置）  
- **GTPose**：用 GT pose 把 `pred['pts3d_cam']` 放到主车系（“只看几何，不看 pose”）

命令：
`PYTHONPATH=$(pwd) python scripts/viz_opv2v_pred_vs_gt_boxes_html.py --checkpoint <CKPT> --split validate --sequence <SEQ> --frame <FRAME> --main_agent <MAIN_AGENT> --num_views 2 --mask_thresh 0.5 --out_html eval_runs/demo_alignment/pred_vs_gt_<TAG>.html`

判据（决定下一步训练策略）：

- **如果 GTPose 也很差**：不是 pose 漂移，是 **几何（ray dirs/depth/scale）崩了**，必须做几何监督。  
- **如果 GTPose 好、PredPose 差**：主要是 **pose**，先做 pose quick finetune。

## 2. Quick finetune 两阶段打法（推荐）

### 2.1 Pose quick finetune（只修 pose）

脚本：`scripts/finetune_opv2v_cyl_pose_quick.py`  
只训练 `pose_head/pose_adaptor`，让 `pose_err_rel` 先降到可接受。

示例：
```bash
PYTHONPATH=$(pwd) python scripts/finetune_opv2v_cyl_pose_quick.py \
  --init_checkpoint experiments/opv2v_coop_det_e2e/checkpoint-best.pth \
  --out_checkpoint experiments/opv2v_cyl_pose_quick/checkpoint-<SEQ>_<MAIN_AGENT>_pose_v1.pth \
  --split validate --sequence <SEQ> --main_agent <MAIN_AGENT> \
  --frames <F1>,<F2>,<F3>,<F4>,<F5> --num_views 2 \
  --steps 400 --lr 1e-4
```

### 2.2 Geometry finetune（LiDAR 投影监督：修 ray dirs/depth/scale/mask）

脚本：`scripts/finetune_opv2v_cyl_geom_quick.py`  
关键参数：`--pts_supervision lidar`（用 GT `.pcd` 投影监督 `pred['pts3d_cam']`）+ `--w_mask`（用 LiDAR occupancy 监督 mask logits）。

示例：
```bash
PYTHONPATH=$(pwd) python scripts/finetune_opv2v_cyl_geom_quick.py \
  --init_checkpoint experiments/opv2v_cyl_pose_quick/checkpoint-<SEQ>_<MAIN_AGENT>_pose_v1.pth \
  --out_checkpoint experiments/opv2v_cyl_geom_quick/checkpoint-<SEQ>_<MAIN_AGENT>_geom_lidar_v1.pth \
  --split validate --sequence <SEQ> --main_agent <MAIN_AGENT> \
  --frames <F1>,<F2>,<F3>,<F4>,<F5>,<F6>,<F7>,<F8>,<F9> --num_views 2 \
  --steps 1000 --lr 5e-5 \
  --w_pts_cam 1.0 --w_pose 0.2 --w_mask 0.2 \
  --pts_supervision lidar
```

训练中/训练后必做：重新跑 1.2 的 HTML 对比，确认：
- GTPose 下的点云形态从“楔子/射线”变成“rings/表面点”；
- GT boxes 内重新出现稠密车辆点；
- PredPose 与 GTPose 两列逐渐趋同（说明 pose 也在一起收敛）。

## 3. “LiDAR 投影监督”到底在监督什么（机制拆解）

目标：把 GT LiDAR 点云变成和模型输出同形状的 supervision（每个像素一个 3D 点）。

1. **CARLA→OpenCV 坐标变换**：把 `.pcd` 点从（X前Y右Z上）转到（X右Y下Z前）。  
2. **投影到 cylindrical panorama**：对每个点计算 azimuth/elevation，落到像素 `(u,v)`，并对同一像素做最近点 z-buffer，得到稀疏 depthmap `D`。  
3. **depthmap→GT 点图**：数据集预先提供 `ray_directions_cam[h,w,3]`，直接 `pts3d_cam_gt = D * ray_dirs`。  
4. **监督模型输出**：对 `valid_mask=(D>0)` 的像素，回归 `SmoothL1(pred['pts3d_cam'], pts3d_cam_gt)`。  
   - 因为 `pred['pts3d_cam']` 由 `ray_dirs * depth_along_ray * metric_scaling_factor` 组合而来，这个损失会同时拉回 **方向、深度、尺度**。  
5. **可选 mask 监督**：把 `(D>0)` 当 occupancy，监督 `non_ambiguous_mask_logits`（BCEWithLogits + pos_weight），减少“虚空点/长射线”。

实现入口：
- `scripts/finetune_opv2v_cyl_geom_quick.py`：`_lidar_pcd_to_cyl_depthmap()` / `_pts3d_cam_loss(supervision='lidar')` / `_mask_logits_loss_from_lidar()`

## 4. 训练期“必须盯”的可视化与指标

### 4.1 一张图就能看懂（BEV PNG）

脚本：`scripts/viz_opv2v_pred_bev_png.py`  
左右两张：PredPose vs GTPose（用于分离 pose/geometry）。

示例（主车 GT LiDAR）：
```bash
PYTHONPATH=$(pwd) python scripts/viz_opv2v_pred_bev_png.py \
  --checkpoint <CKPT> --split validate --sequence <SEQ> --frame <FRAME> --main_agent <MAIN_AGENT> \
  --num_views 2 --mask_thresh 0.5 --gt_lidar_mode main \
  --out_png eval_runs/demo_alignment/pred_bev_<TAG>.png
```

示例（多车 fused GT LiDAR：把每个 agent 的 `.pcd` 变换到主车系叠加）：
```bash
PYTHONPATH=$(pwd) python scripts/viz_opv2v_pred_bev_png.py \
  --checkpoint <CKPT> --split validate --sequence <SEQ> --frame <FRAME> --main_agent <MAIN_AGENT> \
  --num_views 4 --mask_thresh 0.5 --gt_lidar_mode fused \
  --out_png eval_runs/demo_alignment/pred_bev_<TAG>_fusedgt.png
```

### 4.2 指标（用于避免“看着对其实不稳”）

推荐同时看：
- Chamfer(mean)：预测点云与 GT 点云的最近邻双向均值（越小越好）
- BEV IoU：占据栅格的 IoU（越大越好）
- pts-in-box median：GT boxes 内点数的中位数（越大越好）
- pose_err_rel：多 agent 相对位姿误差（t/r）

> 经验：如果 “GTPose 下 rings 不对/全是射线”，Chamfer/IoU 往往也会很差；反之 rings 对了，指标会跟着回来。

## 5. 扩展到多车（>2 agents）协同重建收敛

同一套方法论，只需要把 `num_views` 提升到 4/6/8，并选取“同一帧有足够 agent”的样本即可：

1. 先用 1.1/1.2 在 `num_views=4` 下确认问题类型（pose or geometry）。  
2. 跑 2.1（可选）让 `pose_err_rel` 先下来。  
3. 跑 2.2（`--pts_supervision lidar`）让每个 agent 的 `pred['pts3d_cam']` 对齐各自的 GT LiDAR。  
4. 用 4.1 的 `--gt_lidar_mode fused` 叠加多车 GT 点云，确认多车点云在主车系里整体“重合/一致”。

建议从验证集里“agent 数很多”的序列开始（例如 `2021_09_09_19_27_35` 常见 7 agents），优先挑车辆密集帧做 quick finetune，收敛信号最明显。

### 5.1 可复现示例（validate / 4 agents）

以 `validate/2021_09_09_19_27_35` 为例，该序列中多数帧有 7 agents。选择主车 `2314`，固定抽 4 agents 训练：

**(A) Pose quick finetune**
```bash
PYTHONPATH=$(pwd) python scripts/finetune_opv2v_cyl_pose_quick.py \
  --init_checkpoint experiments/opv2v_coop_det_e2e/checkpoint-best.pth \
  --out_checkpoint experiments/opv2v_cyl_pose_quick/checkpoint-validate_2021_09_09_19_27_35_2314_pose_v1.pth \
  --split validate --sequence 2021_09_09_19_27_35 --main_agent 2314 \
  --frames 000091,000093,000095,000097,000099 --num_views 4 --steps 400 --lr 1e-4
```

**(B) Geometry finetune（LiDAR supervision）**
```bash
PYTHONPATH=$(pwd) python scripts/finetune_opv2v_cyl_geom_quick.py \
  --init_checkpoint experiments/opv2v_cyl_pose_quick/checkpoint-validate_2021_09_09_19_27_35_2314_pose_v1.pth \
  --out_checkpoint experiments/opv2v_cyl_geom_quick/checkpoint-validate_2021_09_09_19_27_35_2314_geom_lidar_v1.pth \
  --split validate --sequence 2021_09_09_19_27_35 --main_agent 2314 \
  --frames 000083,000085,000087,000089,000091,000093,000095,000097,000099 --num_views 4 \
  --steps 1000 --lr 5e-5 --w_pts_cam 1.0 --w_pose 0.2 --w_mask 0.2 --pts_supervision lidar
```

**(C) 多车可视化：用 fused GT LiDAR 作基准**
```bash
PYTHONPATH=$(pwd) python scripts/viz_opv2v_pred_bev_png.py \
  --checkpoint experiments/opv2v_cyl_geom_quick/checkpoint-validate_2021_09_09_19_27_35_2314_geom_lidar_v1.pth \
  --split validate --sequence 2021_09_09_19_27_35 --frame 000093 --main_agent 2314 \
  --num_views 4 --mask_thresh 0.5 --gt_lidar_mode fused \
  --out_png eval_runs/demo_alignment/pred_bev_geom_lidar_v1_validate_2021_09_09_19_27_35_2314_000093_fusedgt.png
```

产物示例：
- `eval_runs/demo_alignment/pred_bev_geom_lidar_v1_validate_2021_09_09_19_27_35_2314_000093_fusedgt.png`
- `eval_runs/demo_alignment/pred_vs_gt_geom_lidar_v1_validate_2021_09_09_19_27_35_2314_000093.html`

**(D) 结果趋势（同设置 9 帧均值，GT=fused）**
- `det_e2e`：Chamfer≈2.200m，BEV IoU≈0.194，pose_err_rel_mean≈181.6m / 48.1°
- `pose_v1`：Chamfer≈2.928m，BEV IoU≈0.164，pose_err_rel_mean≈51.4m / 20.0°
- `geom_lidar_v1`：Chamfer≈0.408m，BEV IoU≈0.353，pose_err_rel_mean≈10.6m / 2.5°

> 注：多车场景的 “fused GT” 会明显比“主车 LiDAR”更稠密、更难匹配，因此 Chamfer 不会像单车那样轻松到 0.3m 左右；但只要 rings/表面形态对齐、IoU 变高、相对位姿误差显著下降，就说明协同重建已经进入可训练区域。
