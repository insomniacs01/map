## MapAnything 协同感知 Vehicle-ID Embedding 方案记录

### 背景 & 现状
- 当前已经完成单车 OPV2V 微调，数据 loader（`mapanything/datasets/opv2v.py`）会把每辆车的相机姿态转换到各自的 ego frame，再交给 MapAnything 训练/推理。
- 新目标是做多车协同（同一时间戳下多辆车共享视角、位姿、稀疏深度），以解决遮挡和远距离盲区问题。
- 原始模型默认把所有视角当成“无序图集”，没有车辆 ID 概念，导致多车样本被误当成同一刚体，严重干扰跨车几何关系。

### 核心痛点
1. **身份缺失**：模型无法区分 `641/650/659` 等车辆，注意力层会尝试在同一刚体内解释所有视角，破坏真实的跨车位姿关系。
2. **刚体约束浪费**：每辆车自身 4 个相机有固定外参，但模型因为缺少 ID 会把跨车视角也拉进同一刚体，使得本可利用的刚体先验被稀释。
3. **世界系不统一**：原 `OPV2VDataset` 把每辆车的视角都转换到自己 ego frame。多车协同需要统一到 CARLA 世界系，否则无法同时训练多个车辆的视角。

### 主要难点
- **数据重排**：需要在 dataloader 中按 sequence/timestamp 对多辆车求交集，保持统一世界系 cam2world，并附带 agent id、深度/稀疏 mask。
- **模型改动尽量小**：不能大规模重训，只允许在已有 MapAnything 上增加少量可训练参数（Prompt/Adapter），并使用 checkpoint 微调。
- **推理显存限制**：多车多视角会把 `num_views` 拉高，需要在数据层或推理策略上控制视角数量、分辨率，不能简单倍增开销。

### 解决方案概述
1. **数据集调整**
   - 新建协同版 dataset：同一时间戳下聚合多车 `frame_id`，保持 world frame cam2world，不再转 ego。
   - `view` dict 增加 `agent_id` (`int`) 和可选 `non_ambiguous_mask`（稀疏深度 > 0）。
   - `label/instance` 保留原有信息用于日志；`dataset_name` 按需设置（如 `OPV2V-Coop`）。

2. **Vehicle-ID Embedding（Prompt + Adapter）**
   - 在模型 `__init__` 中新增 `nn.Embedding(num_agents, enc_dim)`，默认只针对出现过的车辆 ID。
   - 在 `_encode_n_views` 之后或构造 `MultiViewTransformerInput` 前，把 `agent_emb` 加到每个视角的 patch/global token；也可将其作为额外的 input token。
   - 训练时只解冻 `agent_embed`（以及可选的一层小 MLP/Adapter）。主干 encoder / info-sharing / heads 继续冻结，满足“不重新训练”的约束。

3. **微调策略**
   - 继续使用单车 stage1 checkpoint 初始化。
   - Batch 中混合多车样本并随机打乱视角顺序，防止模型把“ID 数字”与具体方向绑定。
   - Loss：沿用 MapAnything pose+depth loss；必要时增加同车外参一致性约束或跨车点云对齐正则。

4. **推理/部署**
   - 输入图像 + 对应 agent_id，经过模型得到每张图的 `camera_pose`（统一 world frame）、深度/点云。
   - 将各车辆点云变换到同一世界系即可做协同建图或 3D 检测。
   - 如果单车推理仍需兼容，默认 `agent_id=0` 即可保持旧行为。

### 预期收益
- 模型知道“哪张图属于哪辆车”，注意力能在同车视角内保留刚体约束，在跨车视角之间只在共视区域建立联系。
- 降低跨车位姿估计的歧义，融合点云时不再出现严重错位。
- 由于只训练小规模 embedding/adapter，可在单卡上快速收敛，满足算力约束。

### 待验证/疑点
1. **num_agents 选择**：OPV2V 场景中车辆 ID 数目较多，embedding 是否需要按“车辆编号”还是“协同角色（ego/协作）”区分？需要实验对比。
2. **完全无共视**：若某些车之间完全没有重叠视角，embedding + pose loss 是否足够收敛？需在数据上确保有共视或引入额外先验（来自 IMU/GNSS）。
3. **Adapter 位置**：仅给 encoder 输出加偏置 vs. 把 embedding 作为额外 token，哪种方式更稳定？可能需要两种方案的 ablation。
4. **推理视角上限**：多车样本的视角数增多，注意力成本会显著提升，是否需要在 loader 里做视角筛选（例如仅取每车前/左/右相机）？需要在显存和精度之间平衡。
5. **稀疏深度质量**：OPV2V 深度是稀疏的，是否需要额外的 mask 或 confidence 来避免噪声影响跨车监督？需要进一步评估。

### 下一步计划
1. 完成协同版 dataset（统一世界系 + agent_id + mask）并做可视化验收。
2. 在 `MapAnything` 模型中实现 Vehicle-ID Embedding 注入点，确保不破坏原有推理。
3. 准备小规模协同样本集，冻结主干，仅训练 embedding/adapter，观察 pose/深度质量对比。
4. 针对上面的疑点逐项实验，记录效果与资源开销，为后续部署提供依据。

