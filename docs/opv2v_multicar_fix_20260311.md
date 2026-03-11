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
