# OPV2V 多车协同修复记录（2026-03-11）

## 结论

多车点云“完全不能看”并不只是“车离得太远”。这次排查确认至少有四个核心问题叠在一起：

1. `OPV2VCoopDataset` 原先把多车多相机视角打平后随机采样，破坏了“同一辆车的 4 个相机是固定 rig”这个最关键的结构先验。
2. 模型没有任何显式的 `agent/camera` 身份信息，注意力层会把跨车视角误当成同一刚体去解释。
3. 协同训练配置没有启用 pairwise relative pose loss，跨车几何关系没有被直接监督。
4. 运行层面还有两个隐形阻塞：单卡显存不足（其实是 GPU 被别人任务占满）以及根盘写满导致 checkpoint 保存失败。

所以，单纯给图片“打标签”这个思路只对了一半：确实需要让模型知道哪些图来自同车、哪些来自不同车，但更重要的是要把这种结构信息写进 **数据采样方式 + 模型输入身份嵌入 + 训练损失**，而不是只靠分类式标签本身。

## 已实现修复

### 1. 结构化协同采样
- 新增 `structured_sampling / agents_per_sample / views_per_agent / agent_selection_policy / max_agent_distance / require_complete_rig / shuffle_views / emit_identity_metadata`。
- 现在一个 2 车样本会稳定返回：
  - agent 0: camera 0,1,2,3
  - agent 1: camera 0,1,2,3
- 这一步直接把“同车 4 相机距离近、外参固定”这个先验交给了模型。

### 2. agent/camera identity embedding
- 在 `MapAnything` 里新增 `agent_identity_embedding` 和 `camera_identity_embedding`。
- 每个视角在 encoder feature 上注入身份偏置，再进入 info-sharing transformer。
- 这不是让模型“认识某一台具体车辆编号”，而是让它知道当前视角在样本中的角色：
  - 来自哪辆协同车
  - 是该车的哪个相机

### 3. 协同损失显式打开相对位姿监督
- 新增 `configs/loss/opv2v_coop_pairwise_loss.yaml`。
- 打开 `compute_pairwise_relative_pose_loss=True` 和 `compute_world_frame_points_loss=True`。
- 这一步让多车之间不再只靠稀疏共视区域“猜”，而是直接被 loss 约束到统一世界几何上。

### 4. 资源问题修复
- 通过 `sudo` 释放了被其他训练任务占满的 GPU，确认原始 448 分辨率协同配置在 **空闲 24GB 3090** 上可以启动并跑过首个 iteration，不再 OOM。
- 发现根盘 `/` 只剩 0 空间，checkpoint 保存失败并留下 `.tmp` 文件；后续实验改写到大盘：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments`

## 已验证结果

### smoke（224x126, encoder frozen, checkpointing on）
- 运行时间：约 51 秒 / 1 epoch
- 最大显存：`8596 MiB`
- 成功完成训练、验证，并保存：
  - `checkpoint-last.pth`
  - `checkpoint-1.pth`
  - `checkpoint-best.pth`
- 输出目录：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_v1_smoke_run2`

### 原始目标配置（448x252, 8 views, encoder trainable）
- 在 GPU 空闲后已成功跑过第一个训练 iteration。
- 首个 iteration 统计：
  - `max mem: 14181 MiB`
  - 不再发生 OOM
- 输出目录：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_v1_debug_fullgpu`

## 推荐训练顺序

### Stage A：先把结构和身份学稳
- `dataset=opv2v_coop_ft_structured_stagea`
- `model=mapanything_opv2v_coop_lowmem`
- `train_params=opv2v_coop_stagea`
- 特点：224 分辨率、8 视角、冻结 encoder、训练 info-sharing/head/identity embedding。
- 目标：先把跨车相对位姿和同车 rig 结构学稳定。

### Stage B：再做 448 分辨率细化
- `dataset=opv2v_coop_ft_structured_stageb`
- `model=mapanything_opv2v_coop`
- `train_params=opv2v_coop_stageb`
- 特点：448 分辨率、小数据量精修、encoder 低学习率解冻。
- 目标：把深度边界、尺度和点云细节再拉回来。

## 对你原始想法的判断

### “给同车图片打同一标签”
方向是对的，但实现上不应该只停留在标签本身。

更有效的做法是三件事一起做：
- 数据采样时保证同车 4 相机成组出现
- 模型输入里显式注入 `agent/camera` 身份嵌入
- loss 里显式约束多车相对位姿

### “多车太远，共同特征太少”
这也对，但它更像 **课程学习 / 数据分布** 问题，而不是第一性根因。

如果后面发现远距离样本仍不稳定，建议下一步再做：
- 先加 `max_agent_distance`，只训近邻车对
- 等模型收敛后逐步放宽距离阈值

## 当前最重要的事实

现在最关键的问题已经从“模型完全学不动 / 点云彻底乱掉”转成了“如何把稳定跑通的协同训练继续做成更高质量的正式版”。

换句话说：
- **结构化采样 + 身份嵌入 + pairwise loss** 已经把方向纠正过来了；
- 接下来主要是按照 Stage A / Stage B 跑完，并做点云可视化验收。

## 补充发现：为什么 GPU 没跑满

之前显卡利用率不高，主要不是 dataloader 卡死，也不是模型没有真正训练，而是当前 long 配置在采样器里被折算成了 **每卡每步只有 1 个样本**：

- `dataset.num_views = 8`
- `train_params.max_num_of_imgs_per_gpu = 8`
- `easy_dataset.py` 里实际 batch size 计算为 `max_num_of_imgs_per_gpu // num_views`
- 所以当时真实 batch size = `8 // 8 = 1 sample / GPU`

这会带来两个直接后果：

1. 单步显存只用到约 `8.6GB`，GPU 计算波峰很短，所以 `nvidia-smi` 看起来像“没跑满”。
2. 虽然训练本身是正常推进的，但单卡吞吐明显偏低，不利于做更长、更有说服力的实验。

已经完成的探针结果表明：

- `160` 是当前 `224x126 / 8 views / lowmem / encoder frozen` 设置下的稳定高吞吐上限
- 该档位约使用 `19.7GB` 显存
- `176` 及以上开始 OOM

因此，接下来的正式长跑应优先使用 `max_num_of_imgs_per_gpu=160`，先把单卡真正吃满，再决定是否扩展到更多 GPU。

## 补充发现：为什么会在验证阶段 OOM

把单卡训练吞吐提到 `max_num_of_imgs_per_gpu=160` 之后，训练本身是稳定的，但验证阶段会在第一个 batch 后 OOM。排查后确认：

- 训练 dataloader 真实 batch size = `160 // 8 = 20` 个多视角样本
- 代码里原本把验证 batch 固定写成 `2 * (train_max_imgs // num_views)`
- 因此验证 batch size 被放大到了 `40` 个多视角样本
- 在 `8 views / 224x126 / lowmem` 设置下，这个验证 batch 会把显存打到约 `22.3GB`，随后再申请约 `4.3GB` 时直接 OOM

所以这不是“训练配置本身仍然不稳”，而是 **验证 batch 的放大规则不再适用于高吞吐协同训练**。

对应修复是：

- 新增 `max_num_of_test_imgs_per_gpu`
- 默认让验证回落到和训练同级别的显存预算，而不是强行 `2x`
- 在进入验证前调用 `torch.cuda.empty_cache()`，减少缓存碎片的影响

修复后，两条 `bs160` 长跑都已经成功：

- 跑完整个验证集 `26` 个 batch
- 显存峰值降到约 `13.4GB`
- 验证结束后继续进入 `Epoch 1` 训练

这说明当前真正可持续的方案是：**训练吃满单卡，验证单独控 batch，不再共享“翻倍”策略。**

---

# 2026-03-11 晚间新增诊断

## 1. 固定 20 帧正式评测结论

- 原始评测目录：`/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/eval_bs160_epoch0_test20_seed42`
- 修正尺度口径后的评测目录：`/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/eval_bs160_epoch0_test20_seed42_scalefix`
- `base`：`pose_abs=20.31`，`pose_rot=5.09`，`depth_rmse=16.51`，`depth_mae=9.35`，`bev_iou_filtered=0.2677`
- `near40`：`pose_abs=20.06`，`pose_rot=4.83`，`depth_rmse=16.31`，`depth_mae=9.26`，`bev_iou_filtered=0.2655`
- 已确认此前引用的 `scale_err = |s-1|` 是 legacy 口径，不适用于 OPV2V；正确口径应为基于 GT 的 `scale_to_gt_*`

修正后的结论：

1. `near40` 的确在 pose/depth 上略优。
2. 几何指标没有同步改善：`chamfer_filtered_pred_to_gt 1.569 -> 1.588`，`bev_iou_filtered 0.2677 -> 0.2655`。
3. 此前“`near40` 尺度更差”的判断是错的；按正确尺度口径，`near40` 反而更好：
   - `scale_to_gt_err: 0.8016 -> 0.7147`（`20/20` 帧更优）
   - `scale_to_gt_log_err: 1.7260 -> 1.3611`（`20/20` 帧更优）
   - `scale_to_gt_eq_rel_err: 5.5281 -> 3.5328`（`20/20` 帧更优）
4. 因此当前真正的 trade-off 是：`near40` 提升了 pose/depth 和尺度，但没有带来更好的跨车几何融合。

## 2. 距离截断的真实代价

训练集原始可用多车帧（同帧原始 `>=2` agent）：`6374`

- `<=30m`：`3573`（`56.1%`）
- `<=40m`：`4690`（`73.6%`）
- `<=50m`：`5339`（`83.8%`）
- `<=60m`：`5758`（`90.3%`）
- `<=80m`：`6160`（`96.6%`）

同时，`>=3 agents` 的帧占比从原始 `61.5%` 降到 `40m` 截断后的 `25.5%`。

这说明 `40m` 虽然没有把数据砍到很少，但显著削弱了多车协同的“多主体丰富度”，也减少了长基线尺度样本。

## 3. 为什么“同车打标签”不是当前最有效修复

当前代码里 `label` 字段主要用于 view naming / 外部模型的 image path bookkeeping，不会直接变成“同标签更近、异标签更远”的损失。

所以：

- 单纯把同车图像打相同标签，不会自动让模型学到 rig 内几何更近。
- 真正有效的是：
  - 显式 `agent/camera identity embedding`（已完成）
  - 同车 rig 刚体约束 / baseline 约束（待加强）
  - overlap-aware / curriculum 采样（下一版优先做）

## 4. 新增代码级修复

- [x] 修正 `scripts/batch_eval.py` 的 OPV2V 尺度定义：从 legacy `|s-1|` 改为基于 GT 的 `scale_to_gt_*`
- [x] 在 `OPV2VCoopDataset` 初始化阶段增加距离/视角约束后的场景预过滤
- [x] 目的：避免 `Need 2 agents after filtering but got 1` 的无效样本在训练中反复重试
- [x] 新增 `configs/dataset/opv2v_coop_ft_structured_stagea_mix40.yaml`
- [x] 策略：`4096 near40 + 4096 full-range` 混合 curriculum

## 5. 下一版验证计划

- [x] 已启动 `mix40` 正式训练：`opv2v_coop_stagea_long_bs160_mix40_run1`（PID `3104211`）
- [x] 在相同 epoch 快照上复用固定 `20` 帧评测（已完成 `base / near40` 的 corrected scale rerun）
- [ ] 如果混合 curriculum 仍有明显 `scale_to_gt_*` 偏差，再补“跨车 baseline / rig-consistency”损失
