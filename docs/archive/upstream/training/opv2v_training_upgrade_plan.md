# OPV2V 训练改进方案（Mask-Aware + Scale 蒸馏 + 多车一致性）

> 目的：针对当前 Stage2 微调在高空区域缺乏监督、尺度漂移、协同视角不稳定的问题，设计三项可落地的训练改动。以下每步都列出了所需代码文件、验证方式与可能的风险。

## 1. Mask-Aware Depth Loss
- **目标**：深度 GT 只覆盖 ~2–3% 像素，未标注区域经常出现在天空/高空，直接参与 L1/L2 会导致模型输出“漂浮层”。Mask-aware loss 仅在 `mask>0` 的像素上计算主损失，同时对 `mask=0` 区域加入高度/先验正则。
- **实现建议**：
  1. 在 `mapanything/train/losses/depth_loss.py`（或 `overall_loss` 调用深度分支处）读取 `sample["depth_mask"]`，改成：
     ```python
     valid = (gt_depth > 0).float()
     depth_loss = (valid * (pred - gt).abs()).sum() / valid.sum().clamp(min=1)
     ```
  2. 对 `valid==0` 区域追加正则，例如 `lambda_height * relu(pred_z - z_max)`，其中 `z_max≈4 m`，避免预测漫无目的地延伸到高空。
  3. 将权重写入 Hydra 配置（`configs/loss/depth.yaml`），方便 ablation。
- **验证**：对 Stage2 再训几千 step，观察 depth RMSE/MAE 是否随 mask-aware loss 收敛。配合 `scripts/filter_pointcloud.py` 的 Chamfer/BEV 指标，确认高空“漂浮层”减少。

## 2. Scale 蒸馏（预训练 → 微调）
- **目标**：预训练模型在局部 pose/scale 上更稳定，但绝对尺度未对齐；Stage2 则在协同输入下漂移更大。将预训练模型作为 Teacher，对 Stage2 pose/scale head 施加约束，保留其本地几何特性。
- **实现建议**：
  1. 在训练脚本（`mapanything/train/training.py`）增加 Teacher 模型 loader：`teacher = MapAnything.load_from_checkpoint(pretrain_ckpt).eval()`，梯度不回传。
  2. 在前向过程中额外保存 `teacher_preds = teacher.infer(inputs, memory_efficient=True)`，提取其中的 `camera_poses`、`metric_scaling_factor`。
  3. 编写新的 loss（`loss/scale_distill_loss.py`），对 Stage2 输出与 Teacher 输出计算 L2/L1，支持 `detach()` + 温度系数。
  4. 在 Hydra config 里增加开关 `loss.scale_distill_weight`，可随实验调节。
- **验证**：监控训练日志中的 scale_err、pose_abs；评估阶段对 Stage2 单车/协同都对比 Teacher 蒸馏前后的 Chamfer/BEV。理想情况是：协同模式保持现有优势，单车模式不再退化到 3–4 m。

## 3. 多车一致性约束
- **目标**：Stage2 期望同一时间戳多车辆输入时保持一致 pose，但当前 loss 主要针对主车 -> 其他视角，缺乏“车辆之间的互相对齐”。可在 feature/pose 层引入一致性 regularization。
- **实现建议**：
  1. 在 `mapanything/models/modules` 中实现 `PairwisePoseConsistencyLoss`：遍历 batch 内所有 agent，计算 `pose_i * pose_j^-1` 与 YAML 真值之间的差距，或至少保证 `pose_i`、`pose_j` 的相对距离/角度满足先验（例如 < 某阈值）。
  2. 若数据中存在车辆 3D bbox，可把 `vehicle_bboxes` 变换到主车坐标系，监督“当前推理出的点云与 bbox 之间的偏差”。
  3. 配置中引入 `loss.coop_consistency_weight`，默认仅在 Stage2 训练生效。
- **验证**：在多车推理时观察 `stage2/coop_metrics.csv` 的 worst-case 帧（例如 `2021_08_18_19_48_05/000340`），确认 pose_abs 明显下降，scale_err 收敛，同时对 Chamfer/BEV 指标进行 before/after 对比。

## 推荐迭代顺序
1. **Mask-aware depth loss**（风险低，易实现）→ 快速跑一段 Stage2 训练，验证“漂浮层”明显减少。
2. **Scale 蒸馏**（需额外显存，但能增强单车输入表现）。
3. **多车一致性**（改动最大，可在前两项稳定后逐步加入）。

每次改动都建议配合 `scripts/batch_eval.py --pc_metrics ...` 记录 Chamfer/BEV，以及 `filter_pointcloud.py` 的多帧过滤结果，确保输出点云满足车辆检测/交通参与者检测的需求。
