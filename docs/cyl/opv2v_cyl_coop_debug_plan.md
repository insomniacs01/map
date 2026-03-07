# OPV2V 多车训练调试记录

## 1. 目标

- 对齐 `docs/multi_car_training_plan.md` 中的方案 1（圆柱拼接 + 车心 Pose）。
- 解决 20251201_021942 这次训练暴露出的所有流程问题，确保重新训练前每个环节都可复现、可验证。

## 2. 已知问题（含日志证据）

1. **环境不一致导致数据集类缺失**  
   - `train.log:4897` 报 `NameError: OPV2VCoopCylindricalDataset`，原因是从 `/home/qqxluca/map-anything3` 目录恢复，读取了旧版 `mapanything`。
2. **Epoch 步数重置错误**  
   - 初始训练 `Epoch: [0] [5000/5000]`（`train.log:3260`）；重启后 `Epoch: [1] [4000/4000]`（`train.log:7693`）。学习率 warmup/余弦调度与 loss 平均因此失真。
3. **验证指标始终很高**  
   - 所有 `Test Epoch`（如 `train.log:3275`, `8103`, `10243`）的总 loss 仍在 39–45，`FactoredGeometryScaleRegr3DPlusNormalGMLoss_pose_trans_avg` 约 7–8 米。
4. **NonAmbiguousMaskLoss 几乎为 0**  
   - 验证时 `NonAmbiguousMaskLoss_mask_avg ≈ 0.008`（`train.log:10249`），可能是 GT mask 极端不平衡或模型输出全 0，导致协同监督失效。
5. **推理流程与训练不一致**  
   - 推理脚本基于最新 checkpoint 渲染的结果“完全不对”，与验证指标高而训练 loss 低的现象一致，说明模型没有泛化。

## 3. 排查流程（必须按序完成）

### 3.1 环境与入口

1. 在 `mapanything_ft` 根目录执行训练 / 恢复：  
   ```bash
   cd /home/qqxluca/vggt_series_4_coop/mapanything_ft
   ```
2. 确认 `PYTHONPATH` 优先包含 `mapanything_ft`，避免加载旧 `map-anything`。
3. 恢复命令固定传入（根据 GPU 数选择 dataset）：
   - 4 卡：`machine=local3090 dataset=opv2v_cyl_coop_ft model=mapanything loss=overall_loss train_params=opv2v_cyl_coop`
   - 5 卡：`machine=local3090 dataset=opv2v_cyl_coop_ft_5gpu model=mapanything loss=overall_loss train_params=opv2v_cyl_coop`
   - `dataset.max_sets_per_epoch=5000 dataset.val.max_sets_per_epoch=1000`（如需，可在 dataset 配置增加显式字段）。

### 3.2 数据与数据加载

1. **圆柱拼接一致性**  
   - 检查 `mapanything_ft/mapanything/datasets/opv2v_cyl.py`，确认 `train/val/test` 都实例化 `OPV2VCoopCylindricalDataset`。
   - 在 `scripts/opv2v_panorama_preview.py` 中随机抽样训练、验证样本，各自确认 4n→n 聚合正常。
2. **Steps / batch 长度**  
   - 在 dataloader 初始化时打印 `len(loader)`；或在 Hydra 配置中增加 `max_sets_per_epoch` 保证数值一致。
3. **Non-ambiguous mask 统计**  
   - 运行一次训练 dataloader，打印 `gt["non_ambiguous_mask"].mean()`；验证 dataloader 同样采样，确保正样本比例不是极端 0。
   - 如果均值 < 0.05，需要检查 `fused["valid_mask"]` 生成逻辑（`opv2v_cyl.py:394/623`），确保拼接后没有把所有像素裁掉。

### 3.3 训练阶段校验

1. **学习率调度**  
   - 使用 `--print-lr` 或手动插桩在每个 epoch 起始打印 `optimizer.param_groups[0]["lr"]`，核对 warmup/余弦曲线。
2. **掩膜损失**  
   - 已在 `NonAmbiguousMaskLoss` 中增加 `pred_mask_mean` 与 `gt_mask_mean` 的日志指标，训练时需重点关注是否在 [0.25,0.5] 区间，否则调整 BCE 权重。
3. **Checkpoints**  
   - 每次保存（`checkpoint-last.pth`）后记录 `epoch`,`global_step`，防止 resume 重复/跳步。

### 3.4 验证与推理

1. **验证集日志**  
   - `Test Epoch` 输出中加入 `valid_mask_ratio`（GT 与预测），并绘制到 TensorBoard，观察是否逐渐上升。
2. **对齐推理脚本**  
   - 使用验证集中的一段 batch 直接喂给推理/可视化脚本，比较输出与训练时 `preds`，确认 pipeline 一致。
3. **定性评估**  
   - 在 `eval_runs` 目录保存渲染结果时同步记录所用 checkpoint、Hydra 配置及 git commit，便于追溯。

### 3.5 分布式一致性

- `DynamicBatchedMultiFeatureRandomSampler` 的 `len(loader)` 与 `world_size` 成反比，若恢复训练时 GPU 数量变化（例如 4 → 5），每个 epoch 的 step 会变少（5000→4000），Warmup / LR 调度会重新按新长度计算。  
- 建议固定 `torchrun --nproc_per_node` 与首轮训练一致，或在新实验中记录 `len(loader_train)` 并据此调整 `train_params.warmup_epochs` / `epochs`。若硬件无法保持一致，可考虑新增 `train_params.steps_per_epoch`，强制 LR 调度按固定步数推进。

## 4. 解决方案与执行顺序

1. **修复数据与 mask**  
   - 若 mask 均值过低，重新检查 `fused["valid_mask"]` 的生成（视线裁剪、拼接区域）。必要时先在日志中把 mask 直接可视化。
   - 调整 `NonAmbiguousMaskLoss` 的权重或在 BCE 中加入 `pos_weight`，防止网络靠全 0 获得低 loss。
2. **统一 loader 长度**  
   - 在 `configs/dataset/opv2v_cyl_coop_ft.yaml` 中新增 `max_sets_per_epoch` 字段，并在 `mapanything/train/training.py` 读取该值或直接设置 `len(dataset)`。
3. **重新训练**  
   - 以上检查通过后，删除旧的 20251201_021942 目录或另起新 experiment，记录 git commit hash、Hydra 配置、step 数。
4. **验证与推理**  
   - 训练过程中每个 `eval_freq` 保存 `Test Epoch` 指标；若仍不下降，再考虑增大训练数据或正则项。

## 5. 复现 Checklist

1. `git status` 干净；记录当前 commit。
2. 运行 `python scripts/train.py ...` 并确认：
   - 日志中 `len(loader_train)=5000/epoch`。
   - `Test Epoch`  Loss 在前几轮开始下降。
   - `NonAmbiguousMaskLoss_mask_avg` 与 `sigmoid(pred)` 均值同步打印，且 > 0.05。
3. 推理脚本在同一 commit + checkpoint 下生成的定性结果与验证指标一致。

## 6. 圆柱方案校验记录

- **脚本验证**：已执行  
  ```bash
  cd /home/qqxluca/vggt_series_4_coop/mapanything_ft
  PYTHONPATH=$(pwd) /home/qqxluca/miniconda3/envs/mapanything_ft/bin/python \
      scripts/validate_opv2v_cyl_setup.py --splits train,val --num-samples 8 --verbose
  ```
  输出显示：
  - Train split：视角数恒为 2（batch 中固定抽两辆车）、`camera_model={'cyl'}`、panorama 分辨率为 (1008×252)，`non_ambiguous_mask` 与 `valid_mask` 均值分别位于 `[0.328, 0.329]` 与 `[0.312, 0.314]`。  
  - Val split：视角恒为 4，mask/valid 均值 ≈ `[0.331, 0.316]`。  
  说明圆柱拼接后非歧义像素覆盖率正常，验证 `4n → n` 聚合可行。
- **视觉抽查**：已运行  
  ```bash
  PYTHONPATH=$(pwd) /home/qqxluca/miniconda3/envs/mapanything_ft/bin/python \\
      scripts/opv2v_panorama_preview.py --root ${machine.opv2v_images_root} \\
      --depth_root ${machine.opv2v_depth_root} --split train --index 0 \\
      --output_dir eval_runs/opv2v_cyl_precheck/train_idx0
  PYTHONPATH=$(pwd) /home/qqxluca/miniconda3/envs/mapanything_ft/bin/python \\
      scripts/opv2v_panorama_preview.py --root ${machine.opv2v_images_root} \\
      --depth_root ${machine.opv2v_depth_root} --split validate --index 0 \\
      --output_dir eval_runs/opv2v_cyl_precheck/val_idx0
  ```
  生成的 `panorama.png` 展示 360° 圆柱图、遮挡 mask/深度 `.npy` 保存在 `eval_runs/opv2v_cyl_precheck/` 供后续复查。  
- **Pose 一致性**：脚本输出中 `label=sequence/agent` 每条独立对应 1 张 panorama，`camera_pose_trans` 的范数分布在 0~60m，符合“车心坐标”预期。

## 7. 交互式可视化（HTML）

- 为了在无 GUI 的 SSH 环境下仍能检查圆柱样本，添加了 `scripts/visualize_opv2v_cyl_html.py`：  
  ```bash
  cd /home/qqxluca/vggt_series_4_coop/mapanything_ft
  PYTHONPATH=$(pwd) /home/qqxluca/miniconda3/envs/mapanything_ft/bin/python \\
      scripts/visualize_opv2v_cyl_html.py --split train --index 0 \\
      --output-html eval_runs/opv2v_cyl_precheck/train_idx0/panorama.html
  PYTHONPATH=$(pwd) /home/qqxluca/miniconda3/envs/mapanything_ft/bin/python \\
      scripts/visualize_opv2v_cyl_html.py --split val --index 0 \\
      --output-html eval_runs/opv2v_cyl_precheck/val_idx0/panorama.html
  ```
- 每个 HTML 会包含三列（RGB / 深度 / non-ambiguous mask），支持 Plotly 的交互缩放。若需本地浏览，可在仓库根目录运行 `python -m http.server 8000`，然后在浏览器打开  
  `http://<服务器IP>:8000/mapanything_ft/eval_runs/opv2v_cyl_precheck/train_idx0/panorama.html`（或 `val_idx0`）。  
- 该可视化与训练实际数据完全一致，可随时调整 `--index` 抽样更多车辆，或切换 `--split test` 对比测试集分布。

## 8. 模型输出点云调试（2025-12-07 更新）

1. **脚本**：`scripts/render_pointcloud_html.py`（新增功能：Plotly 交互视角、per-view PNG、JSON 摘要、相机视角控制）。  
2. **命令模板**  
   ```bash
   cd /home/qqxluca/vggt_series_4_coop/mapanything_ft
   PYTHONPATH=$(pwd) conda run -n mapanything_ft python scripts/render_pointcloud_html.py \
       --config scripts/local_infer_config.json \
       --checkpoint /home/qqxluca/map-anything3/experiments/mapanything/training/opv2v_cyl_coop/20251201_021942/checkpoint-best.pth \
       --output-dir eval_runs/opv2v_cyl_precheck \
       --view-output-dir eval_runs/opv2v_cyl_precheck/val_idx0_views \
       --debug-summary-json eval_runs/opv2v_cyl_precheck/val_idx0_summary.json \
       --val-index 0 \
       --point-pose-source pred
   ```
   - 将 `--point-pose-source` 换成 `gt` 可查看 GT pose 下的点云，便于定位问题来自姿态还是深度。
3. **产物**  
   - `val_idx0_pointcloud.html`：彩色点云、稀疏/稠密深度、mask 视图均可放大；窗口尺寸已调大，浏览器端可直接全屏。  
   - `val_idx0_summary.json` / `_gtpose.json`：统计 pose/scale/depth RMSE、边界值等，训练趋势分析依赖该文件。  
   - `val_idx0_views/*.png`：每辆车（包含 agent_id）导出彩色点云、深度、mask 截图，便于自查；无需人工再截图。  
   - `val_idx0_views/*.npy`：缓存关键 tensor，可在 Notebook 中复现渲染过程。
4. **使用建议**  
   - 每个新的 checkpoint（尤其是验证 loss < 5 的节点）立即运行一次渲染，并把 HTML/JSON 路径记录在本文件“圆柱方案校验记录”段落下方。  
   - 若 `summary.json` 中 `pred_mask_mean < 0.05` 或 `pose_rmse > 10`，先对比 `gtpose` 结果，再结合训练日志排查 loss 设置。  
   - 需要把服务器端口暴露给本地浏览时，可执行 `python -m http.server 8000` 并通过 `ssh -L` 转发端口。

---

> 备注：此文档面向方案 1 的第一轮调试。所有修改需同步更新 `docs/multi_car_training_plan.md` 中的实现状态，并在 PR/日志中引用本文件。
