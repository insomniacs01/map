# OPV2V 单车评估（641）修复报告

> ⚠️ **归档说明（2025-11-28）**：本报告记录的是早期 `scripts/batch_eval_single_fixed.py` 的 6 帧 CPU 抽样结果，其流程与 `docs/opv2v_batch_eval_analysis_image_only.md` 里确立的 image-only 基准存在冲突，因而仅保留作脚本修复历史。关键差别：
> - **输入模态**：脚本仍沿用旧版数据管线，推理阶段可以访问 YAML 外参 + `depth_z`，因此给出了“pretrain≈4 cm”的过高结论；新版基线在预处理后调用 `strip_external_calibration_inputs` 并剥离深度，仅允许 RGB + intrinsics。
> - **指标定义**：这里比较的是 `camera_i` 相对于 `camera_0` 的相对位姿；标准报告使用绝对 `Pose Abs`/`Pose Rot`/`Scale Err`，对齐部署所需的车辆坐标系。
> - **样本与设备**：本档案只抽测 agent 641 在同一序列的 6 帧、CPU 推理；新版评测锁定 test split 中 10 个多车帧并通过 GPU 并行运行。
>
> 如需真实 image-only 误差或多模型对比，请查阅 `opv2v_batch_eval_analysis_image_only.md` 与 `opv2v_eval_report_image_only.md`。

## 1. 背景与数据
- **数据根目录**：`/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V`（RGB/YAML/PCD）与 `/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/opv2v_depth`（深度）。
- **运行脚本**：`scripts/batch_eval_single_fixed.py`（修复后的单车评估脚本，计算纯“camera1 相对于 camera0”的相对位姿误差）。
- **评估设定**：只使用 agent **641**、test split、随机抽样 6 帧（序列 `2021_08_21_09_28_12` 中的 `000093/195/257/289/321/359`），开启高度 `[−2, 4.2] m`、半径 `<120 m` 的点云过滤。
- **命令**：

```bash
~/miniconda3/envs/mapanything_reconstructed/bin/python \
  scripts/batch_eval_single_fixed.py \
  --split test --sample_size 6 --seed 42 --agent 641 \
  --pc_metrics --pc_filter_z_min -2 --pc_filter_z_max 4.2 --pc_filter_radius 120 \
  --output_root eval_runs/single_agent_fixed --device cpu
```

## 2. 修正版脚本要点
1. **纯相对姿态误差**：`_compute_pose_metrics` 不再涉及任何世界坐标转换，只比较 `cam_i` 相对于 `cam_0` 的 SE(3)（`batch_eval_single_fixed.py:325-350`）。这样直接对齐了“camera1 vs camera0”与 YAML 给定的真实相对位姿，排除了 `camera0` 到 LiDAR 的固定偏移。
2. **点云仍做 CV→UE 旋转**：检测指标需要与 LiDAR `.pcd` 对比，因此依旧保持 `R_CV_TO_UE` 矫正（`batch_eval_single_fixed.py:366-415`）。
3. **其它逻辑**：只保留单车流程、agent 过滤与 CSV/JSON 输出，便于随时复现。

## 3. 新的单车评估结果
`eval_runs/single_agent_fixed/summary_test.json`（6 帧平均）如下：

| 模型 | Pose Abs (m) | Pose Rot (°) | Depth RMSE (m) | Scale Err | Chamferᵣ (m) | Chamferᶠ (m) | BEV IoUᵣ | BEV IoUᶠ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Pretrain** | **0.041** | **0.55** | **11.66** | 4.93 | **1.43 / 1.00** | **0.95 / 1.00** | **0.284** | **0.306** |
| **Stage1（单车微调）** | 0.393 | 3.41 | 11.51 | **3.25** | 2.41 / 1.23 | 2.07 / 1.23 | 0.298 | 0.298 |
| **Stage2（协同微调）** | 4.50 | 4.72 | 14.71 | 11.56 | 3.24 / 1.65 | 1.61 / 1.65 | 0.189 | 0.180 |

> 备注：Chamferᵣ/ᶠ 为“未过滤/过滤高度后的预测→GT / GT→预测”距离；BEV IoUᶠ 使用 `z∈[−2,4.2] m`、`r<120 m` 裁剪。

### 3.1 观察
- 纯相对误差下，pretrain 的姿态误差 ≈4 cm / 0.55°，Stage1 虽下降到 0.39 m / 3.4°（优于 Stage2），但仍明显劣于预训练。这说明 Stage1 确实学到更好的尺度（`scale_err=3.25`），但相机环自身仍有 30~40 cm 的漂移。
- Stage2 由于训练时依赖协同约束，单车推理时相对位姿最不稳定（4.5 m / 4.7°），验证了“协同模型不能直接用于单车输入”。
- 点云指标与先前结论一致：预训练在 Chamfer/BEV 上仍最优，Stage1 通过过滤可接近但仍有 0.2~0.4 m 的整体偏移痕迹。

## 4. 使用建议
1. 若要扩展到其他 agent，修改 `--agent` 并保持相同的 YAML/深度/PCD 路径结构即可。
2. 继续沿用 `eval_runs/single_agent_fixed/{model}/single_metrics.csv` 与 `summary_test.json` 中的数据，确保任何讨论都基于相同的“相对姿态”定义。
3. 若需更多帧或 GPU 加速，可在具备显存的节点上运行同一脚本（改用 `--device cuda`）。

如需更深入的帧级可视化，可使用 `scripts/color_compare.py --coop_root ... --coop_agents 641` 复查预测点云。所有指标均基于本报告所述的相对姿态定义。
