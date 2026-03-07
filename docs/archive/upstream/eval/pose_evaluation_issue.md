# 彩色对比脚本 pose 评估问题说明

## 背景
- 运行 `scripts/color_compare.py` 时，如果通过 `--config_path` 传入 YAML，脚本会从配置中读取真值相机位姿和内参。
- 这些数据除了用于评估误差，还会一起进入 `model.infer`。因此在“有配置/无配置”两种模式下，模型收到的输入不同。
- 用户预期 `image_only` 任务下模型应始终只依赖图像，即使外部提供了真值位姿，也只用于误差计算。

## 原因
1. `load_views_from_config` 会把 `intrinsics` 和 `camera_poses` 放入 view 字典（`scripts/color_compare.py:178-260`）。
2. `preprocess_inputs` 和 `preprocess_input_views_for_inference` 会保留这些键并将其转成 `camera_pose_quats/camera_pose_trans`（`mapanything/utils/image.py`, `mapanything/utils/inference.py`）。
3. `MapAnything` 在前向中检测到这些 pose 数据后会走 `_encode_and_fuse_cam_quats_and_trans` 分支，把外部 pose 特征也融合进来（`mapanything/models/mapanything/model.py:1012+`）。
4. 当不提供配置文件时，view 中不存在这些键，模型就只能完全依赖图像。两种情况输入模态不同，预测位姿自然也不同。

## 解决方案
- 在 `color_compare.py` 中新增 `strip_external_calibration_inputs`，把 `intrinsics/camera_poses/camera_pose_quats/camera_pose_trans/ray_directions` 等所有标定相关键从 view 中删除。
- 载入 YAML 视图后立即调用该函数，只保留图像数据传入 `model.infer`，但仍保留 `camera_info_list` 供误差计算、点云坐标变换和日志使用。
- 这样无论是否提供配置文件，模型的输入都只包含图像，预测位姿保持一致，真值 pose 只在推理后用于评估。

## 当前状态
- 修改后的脚本会在日志中提示“Stripped external intrinsics/pose inputs…”以明确该行为。
- `color_compare_pose_log.txt` 仍会记录每次命令、预测 pose 以及与真值的误差，方便后续比对。
