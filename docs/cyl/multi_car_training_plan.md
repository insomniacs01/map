# 多车协同训练技术方案

面向刚接触相机模型的同学，本文梳理 MapAnything 当前的多视角数据流，分析为何 **4n 视角 → n 车辆** 能缓解收敛，给出两条实现路径，并尽量沿用现有代码，降低入侵性改动。

---

## 更新记录

| 日期 (UTC) | 内容 | 备注 |
| --- | --- | --- |
| 2025-12-07 | 1) 新增 `configs/loss/opv2v_pose_loss.yaml`，在 `FactoredGeometryScaleRegr3DPlusNormalGMLoss` 中开启 `compute_pairwise_relative_pose_loss=True`，并把 `pose_quats_loss_weight`、`pose_trans_loss_weight` 提升到 3；2) `scripts/render_pointcloud_html.py` 支持 per-view PNG、Plotly 交互视角与 JSON 摘要，方便定位姿态漂移；3) `mapanything/models/mapanything/model.py` 加载 checkpoint 时强制 `map_location="cpu"` 以避免抢占 GPU；4) 已尝试 10×3090 训练（`loss=opv2v_pose_loss`），但服务器上 `/home/gongyan/anaconda3/envs/UniMODE/bin/python` 占满 10 卡（约 14 GB/卡），导致 checkpoint 加载或第一轮 backward CUBLAS OOM，等待资源释放后需继续。 | 详见 `docs/cyl/opv2v_cyl_coop_debug_plan.md` “训练阶段校验 / 复现 Checklist”。 |
| 2025-12-05 | 完成 `scripts/visualize_opv2v_cyl_html.py` 与 `scripts/opv2v_panorama_preview.py`，可在无 GUI 场景下检查圆柱样本；`scripts/validate_opv2v_cyl_setup.py` 输出 mask/视角统计，确认 4n→n 聚合质量。 | HTML 示例位于 `eval_runs/opv2v_cyl_precheck/val_idx0/`。 |
| 2025-12-01 | 搭建 `OPV2VCoopCylindricalDataset`，修复 `mapanything/utils/geometry.py` 的 `camera_model=="cyl"` 分支；整理 `docs/README.md` 并拆分 `docs/cyl/*`。 | 对应 checkpoint `20251201_021942`。 |

---

## 1. 现状与痛点

### 1.1 MapAnything 的 2D → 3D 几何流水线（新人入门）

为了理解后续为什么需要圆柱相机，这里先梳理 MapAnything 默认的几何数据流：

1. **Dataset 负责输出“视角字典”**  
   - 每个 `view` 包含 `img`、`camera_intrinsics`、`camera_pose`、`depthmap` 等字段。`BaseDataset._getitem_fn` 会把它们打包成 tensor，并生成 `camera_pose_quats/trans`、`non_ambiguous_mask` 等附加信息（`mapanything_ft/mapanything/datasets/base/base_dataset.py:454-631`）。  
   - 在这一步还会调用几何工具，把像素/深度转换成 `pts3d`、`ray_directions_cam`、`depth_along_ray`，并为所有视角添上 `camera_model`（默认 `"pinhole"`）。
2. **几何工具根据相机类型分发**  
   - `mapanything_ft/mapanything/utils/geometry.py:1467-1496` 中的 `get_pointmaps_and_rays_info` 会根据 `camera_model` 决定走针孔还是圆柱路径：  
     - 针孔 (`"pinhole"`/`"perspective"`) → `get_absolute_pointmaps_and_rays_info`；  
     - 圆柱 (`"cyl"`/`"cylindrical"`) → `get_cylindrical_pointmaps_and_rays_info`（需要额外的 `virtual_camera` 参数描述 360° 视野）。  
   - 因此只要在 dataset 里把 `camera_model` 改成 `"cyl"` 并提供 `virtual_camera`，其余代码无需改动即可获得正确的射线/点云。
3. **模型如何消费这些几何量**  
   - `MapAnything._encode_n_views` 会把 `num_views` 张图片堆成 `[B*num_views, C, H, W]` 送进 DinoV2，再在列表维度拆回来（`mapanything_ft/mapanything/models/mapanything/model.py:618-640`）。  
   - `_encode_and_fuse_cam_quats_and_trans`、`_encode_and_fuse_ray_directions`、`_encode_and_fuse_depths` 等函数会把视角的 pose/射线/深度编码成 token 并与视觉特征融合（`mapanything_ft/mapanything/models/mapanything/model.py:642-840`）。  
   - 如果某个视角没有这些字段（例如随机关闭射线输入），对应的 mask 会被置 False，模型不会强行使用无效几何。

理解这套 pipeline 后，再来看圆柱方案：我们所做的就是在 dataset 层把 4 台针孔相机融合成 1 张圆柱全景图，并保证 `camera_model`/`pose`/`mask` 等字段保持 MapAnything 期望的形态，使模型能够“无缝理解”来自圆柱相机的 3D 信息。

### 1.2 统一几何表示的背景（参考论文 Sec.3）

Keetha et al.《MapAnything: Universal Feed-Forward Metric 3D Reconstruction》将整个模型建立在“Factored multi-view geometry representation”之上：每个视角都要输出深度图、局部射线（ray directions）、摄像机位姿和一个 metric scale，以便在 feed-forward 预测中保持度量一致性。论文强调三点（Figure 1 & Section 3）：

1. **输入柔性**：模型可以接收任意数量的图像，以及可选的内参、外参、深度或已有重建结果。只要 view 字典中填好这些字段，Transformer 就会把它们编码为 token。  
2. **输出统一**：无论任务是 SfM、MVS 还是深度补全，网络最终都会预测一组 factored 表示（深度、射线、pose、scale）。这让不同任务的 supervision 能共享同一通道，只需在 loss 中挑自己关心的部分即可。  
3. **Feed-forward 推理**：由于整个 pipeline 不依赖 BA 或重投影优化，推理速度远快于传统 SfM/MVS，可在一次前向中解决“给定任意几张图的一切几何需求”。

多车场景正是利用了 MapAnything 的这套统一几何接口：我们只需在 dataset 里提供符合规范的圆柱视角（含 pose/射线/遮挡 mask），模型就能像处理其他任务一样进行 feed-forward 推理。

- **数据流假设**  
  `mapanything/datasets/base/base_dataset.py` 的 `_getitem_fn` 会把每个视角视为独立样本：它要求每个 `view` 字典携带 `img`、`camera_intrinsics`、`camera_pose`、`depthmap`，然后调用 `get_absolute_pointmaps_and_rays_info`（`mapanything/utils/geometry.py:1250+`）在针孔模型假设下生成 `ray_directions_cam`。  
  `MapAnything._encode_n_views` (`mapanything/models/mapanything/model.py:622+`) 会将 `num_views` 张图片拼成 `[B*num_views, C, H, W]` 输入 DinoV2，并在 `_encode_and_fuse_optional_geometric_inputs` 中遍历每个视角的射线、深度、姿态。

- **Pose 归一化的副作用**  
  Encoder 将所有视角的 `camera_pose_trans` 堆成 `[B, V, 3]`，通过 `normalize_pose_translations`（`utils/geometry.py:1558+`）按 **batch 平均模长** 归一化，再编码为 Pose token (`model.py:1070-1129`)。  
  在 V2X 场景，单车内部 4 个环视基线只有 ~2 m，而车心之间动辄 20-50 m。平均模长被“大车间距离”统治后，环视 pose 的尺度几乎归零，Pose token 贡献被放大/缩小不均，Transformer 权重难以收敛。

- **4n → n 的好处**  
  将输入阶段按车聚合后，`num_views = n`，每个 Pose token 表示一辆车的中心。`normalize_pose_translations` 此时只处理车辆间的位移，尺度天然一致；同车内的 4 张图被压成“单节点”后，几何编码器看到的都是“以车心为参考”的射线/特征，梯度会稳定很多。

---

## 2. 方案一：几何预处理（Image-Level Fusion）

借助圆柱/球面投影，把单车的 4 张针孔图像几何拼接成一张“全景图”和一个车心 Pose。

> **为什么要优先验证圆柱方案？**  
> 论文 Sec.4 将 MapAnything 的输入划分为“图像 + 可选几何 token”。圆柱预处理本质上是把车内 4 张针孔图压成 1 张等效图像，同时把真实位姿/射线封装进 view 字典，从而让模型继续以“每个 view 一个 token”的方式工作。相比直接修改 Transformer，这种做法最符合论文提倡的“在 dataset 层标准化输入”的理念，也能最快验证 feed-forward 模型在新领域（多车 V2X）中的泛化能力。

### 2.1 虚拟相机与 Remap 表

1. **模型选择**  
   - IPM / 将所有视角 warp 到单一前视针孔会在 ±120° 以外出现严重拉伸与视差空洞。  
   - 虚拟圆柱相机（安装在车顶中心，轴向沿前进方向）能提供 360° 水平视野，避免单点汇聚。
2. **构建映射**  
   - 设定全景尺寸 `W_cyl × H_cyl`、垂直 FOV、圆柱半径。  
   - 枚举每个圆柱像素 `(u, v)`，计算其方位角 `azimuth = 2π*(u/W_cyl-0.5)` 和仰角 `elevation`，得到单位射线 `d_cyl`。  
   - 将 `d_cyl` 依次旋转到 4 个环视相机坐标系（用 YAML 外参），判断是否落在各自视锥内，再用对应内参投影到 `(x, y)`。  
   - 生成 `map[(u,v)] -> (cam_idx, x, y)` 的 Remap 表；重叠区域可按视线与光心夹角最小或“距离圆柱角度最近”策略挑选，保持拼接稳定。
3. **离线缓存**  
   - 对固定安装的车辆，Remap 表与深度插值权重都可提前存成 `.npz`，推理时调用 `cv2.remap` 或 `torch.nn.functional.grid_sample` 即可。

### 2.2 数据流改造要点

- **Dataset 新增 agent 聚合层**  
  - 通过继承 `BaseDataset` 新增 `CoopCylDataset`：`_get_views` 以“车辆”为单位返回 `{agent_id, camera_group: [view0..3]}`。  
  - 在 `_getitem_fn` 中：  
    1. 拿到四张图像与外参，按 Remap 表 warp 成 `cylinder_img`；  
    2. 构造新的 view：`{"img": cylinder_img, "camera_pose": agent_pose, "camera_intrinsics": K_cyl, "camera_model": "cyl"}`；  
    3. 若有深度/LiDAR，可用相同映射生成 `depthmap_cyl`；没有就填全零并把 `depth_prob` 设为 0。  
    4. 将 `num_views` 设置为 “batch 中车辆数量”，让原有 collate 逻辑依旧成立。
- **参考实现**  
  - `mapanything_ft/mapanything/datasets/opv2v_cyl.py` 提供了 `OPV2VCylindricalDataset`：内部加载 4 个环视相机、按虚拟圆柱射线做加权融合，并给 view 附带 `camera_model="cyl"` 和 `virtual_camera` 参数，便于后续几何模块识别。`scripts/opv2v_panorama_preview.py` 可视化数据聚合结果，帮助在集群运行训练前先确认拼接质量。
- **射线与几何信息**  
  - `get_absolute_pointmaps_and_rays_info` 默认针孔，需要实现 `get_cylindrical_pointmaps_and_rays_info`（放在 `mapanything/utils/geometry.py`），根据 `camera_model` 分支调用。  
  - `geometry.get_pointmaps_and_rays_info` 已在行 1467+ 根据 `camera_model` dispatch：传 `"cyl"` 即走 `get_cylindrical_pointmaps_and_rays_info`（需携带 `virtual_camera`），否则回退到针孔路径。只要 dataset view 中写明 `camera_model="cyl"`，整条几何链路就会自动切换。
  - 初期可暂时不传 `ray_directions`（`geometric_input_config["ray_dirs_prob"]=0`），把全景图当纹理；验证有效后再补射线与深度。
- **非侵入改动建议**  
  1. 不直接修改 `BaseDataset`，在新 dataset 中重写 `_getitem_fn` 并调用父类帮助函数（如 `_crop_resize_if_necessary`）。  
  2. 利用 view 字典支持自定义字段（例如 `camera_model`），在 `get_absolute_pointmaps_and_rays_info` 外包一层工厂函数：  
     ```python
     def build_pointmaps(view):
         if view.get("camera_model") == "cyl":
             return get_cylindrical_pointmaps_and_rays_info(**view)
         return get_absolute_pointmaps_and_rays_info(**view)
     ```
  3. 通过配置把新 dataset 注册到 dataloader，避免动核心模型代码。

### 2.3 圆柱方案的校验流程

在任何训练/推理前，需要完成以下 checklist，确保“4n → n” 聚合和数据质量可信：

1. **视觉检查**  
   - 运行 `python scripts/opv2v_panorama_preview.py --root ... --depth_root ...` 抽样若干场景，核对圆柱图是否 360° 连续、拼接无明显断裂。  
   - 同时观察 `fused["valid_mask"]`，确认空洞区域合理（大面积空洞需要调节 `view_selection_margin` 或输入深度）。
2. **自动统计**  
   - 执行 `python scripts/validate_opv2v_cyl_setup.py --num-samples 8`。脚本会按 Hydra 配置实例化 train/val 数据集并输出：  
     - 每个样本的视角数（即 batch 中车辆数）；  
     - panorama 分辨率、`camera_model` 集合；  
     - `non_ambiguous_mask` 与 `valid_mask` 的均值区间。  
   - 正常情况下 mask 均值应 ≥0.05 且 camera_model 仅为 `"cyl"`，若不满足则回到数据生成环节排查。
3. **Pose 与标签一致性**  
   - 在 `validate_opv2v_cyl_setup.py` 输出的 label 中确认 `sequence/agent` 唯一映射至一张圆柱图；取两辆车的 `camera_pose_trans` 做差应接近车心距离（米级），而非单个相机的 1~2 m 基线。  
4. **射线/深度通路**  
   - 在 `base_dataset.py` 中为 `view["camera_model"]=="cyl"` 的样本打印 `ray_directions_cam.shape` 等，确认 `get_pointmaps_and_rays_info` 走了 cylinder 分支。  
   - optional：导出 `view["pts3d"]` 用 MeshLab 验证点云的环状结构。

通过以上流程，可确认方案 1 的数据与几何配置正确，避免把偏差带入训练阶段。自动脚本输出的正常范围可参考本次统计：

- Train：`num_views range = (2,2)`、`camera_models={'cyl'}`、`non_ambiguous_mask mean ∈ [0.328,0.329]`、`valid_mask mean ∈ [0.312,0.314]`、分辨率 (1008×252)；
- Val：`num_views range = (4,4)`、mask/valid 均值约 `[0.331,0.315]`。

若实际输出明显偏离上述区间，应优先回查 dataset 配置或数据资产，再考虑进入训练。

### 2.3 空洞/接缝处理

- **来源**：相邻相机之间存在盲区（A 柱遮挡、车辆自遮挡），映射后会产生“无像素”区域；曝光差异会在接缝处留下亮度跳变。
- **对策**：  
  - 记录 Remap 命中的 `valid_mask` 并在 view 中携带，确保 `ray_directions`、`depth` 编码阶段跳过空洞。  
  - 对重叠区做简单羽化或基于距离的加权平均，缓解接缝。  
  - 在训练配置中加入 augmentation（如色彩抖动）减轻亮度不一致造成的伪边缘。

### 2.4 圆柱可视化

在没有本地 GUI 的情况下，可用 Plotly HTML 检查圆柱样本：

```bash
cd /home/qqxluca/vggt_series_4_coop/mapanything_ft
PYTHONPATH=$(pwd) /home/qqxluca/miniconda3/envs/mapanything_ft/bin/python \
    scripts/visualize_opv2v_cyl_html.py --split train --index 0 \
    --output-html eval_runs/opv2v_cyl_precheck/train_idx0/panorama.html
```

每个 HTML 包含 RGB / 深度 / non-ambiguous mask 三列，可用 `python -m http.server 8000` 暴露给本地浏览器查看（`http://服务器IP:8000/mapanything_ft/eval_runs/.../panorama.html`）。建议在调参前抽查 train/val 样本，确保遮挡/深度合理。

---

## 2.4 分布式训练与 epoch 步数

`opv2v_cyl_coop_ft` 的 `train_dataset` 使用 `40_000 @ Dataset` 扩展样本数，并配合 dynamic sampler（`max_num_of_imgs_per_gpu=4`）在 `num_views=2` 时一次最多消费 2 张 panorama。这样每 40k / 2 = 20k 个 batch，再除以 `world_size` 就是单卡 step 数。示例：

- 4 卡：20,000 / 4 = 5,000 step/epoch（本次失败日志中 02:19 训练即为此配置）。
- 5 卡：20,000 / 5 = 4,000 step/epoch（13:18 的 resume 因多出一张卡导致）。

为维持 5,000 step，可以把 ResizedDataset 的目标长度改为 50,000（`+ 50_000 @ ...`），对应配置 `configs/dataset/opv2v_cyl_coop_ft_5gpu.yaml`。启动命令改用：

```
python scripts/train.py dataset=opv2v_cyl_coop_ft_5gpu machine=local3090 ...
```

若未来 GPU 数量再变，只需按公式 `target_size = steps_per_epoch * world_size * max_batch_size` 调整 `ResizedDataset` 即可。务必在每次启动时检查 `train.log` 前几十行的 `Train dataset length`，确认步数符合预期。

此外，`torchrun --nproc_per_node` 必须和调度评估一致，否则 warmup/余弦曲线都会被压缩。

---

## 3. 方案二：特征级融合（Feature-Level Fusion）

保持数据层仍然是 `4n` 图像，但在 encoder 早期将同车四视角融合成单个特征，再作为 1 个 view 输入多视图 Transformer。

### 3.1 Tensor 形状与流程

```
输入: imgs shape = [B_agents, 4, 3, H, W]
Step1: 重排为 [B_agents*4, 3, H, W] -> 过 DinoV2 patch_embed
Step2: 将同车的 4 份 patch 特征 reshape 成 [B_agents, 4, C_embed, H', W']
Step3: 进行 early fusion -> [B_agents, 1, C_embed, H', W']（拼通道 / Attention / Conv3d）
Step4: 送入 MultiView Transformer，此时 num_views = n_agents
```

### 3.2 Encoder 改动选项

1. **通道拼接**  
   - 直接把 4×RGB 拼成 12 通道，再改造 DinoV2 的 `patch_embed`（第一层 Conv）接受 12 通道。  
   - 初始化策略：复制原 3 通道卷积权重并取平均，或在加载 checkpoint 后仅对新通道做小幅噪声初始化。  
   - 优点是简单；缺点是对预训练权重破坏大，且首层算力涨 4 倍。
2. **共享编码 + Learnable 融合**  
   - 先保持 3 通道输入，分别通过共享的 ViT patch_embed，得到 `[B_agents, 4, C_embed, H', W']`。  
   - 在这一层上插一个轻量的 cross-view attention/ConvGRU，将 4 个视角融合成 1 个特征图。  
   - 这样主体 ViT 可以继续沿用预训练，只在融合模块上做小改动。
3. **视角维 3D 卷积**  
   - 把视角当作“深度轴”，可在 patch 之前加一个 `Conv3d(kernel_size=(4,3,3))` 模块，输出 1 个特征，再送进原 ViT。  
   - 需要注意对齐权重与 BatchNorm/LayerNorm 统计。

### 3.3 Pose/射线注入

- 原 Pipeline 在 `_encode_and_fuse_cam_quats_and_trans` 里按视角分别编码 Pose。如果我们把图像融合成单个 view，仍需决定在哪里注入 per-camera 外参：
  1. **Pose Token 聚合**：把 4 份 `camera_pose_quats/trans` 分别编码，再通过同样的融合模块压成一个“agent pose token”，与 fused 图像一同喂给 Transformer。  
  2. **Late Pose Injection**：在图像走完 ViT 后，单独做一次 cross-attention，将 per-camera Pose token 作 query，输出 agent token。这能保留更多几何差异，但实现复杂度更高。
- 如果完全丢弃 per-camera Pose，网络将在融合阶段失去“哪块特征来自车身哪一侧”的信息，易导致左右/前后歧义，因此至少要保留一个方位编码（例如附加 `[cos(azimuth), sin(azimuth)]` 通道或 Learnable view embedding）。

### 3.4 风险与入侵度

- **预训练破坏**：修改 `patch_embed` 或在 ViT 起始插入新模块会让 `state_dict` 对不上，需要自定义加载和重新训练前几层。  
- **算力开销**：通道/视角叠加会线性放大 FLOPs；在低算力环境里不利于快速验证。  
- **工程复杂度**：需要在 dataloader 里保证视角严格按 `[front,left,rear,right]` 顺序堆叠，还要处理分布式情况下的 reshape/同步。  
- 综合评估，此方案更适合在方案一验证成功后，再作为进一步提升的方向。

---

## 4. 相机参数兼容性评估

### 4.1 方案一：从针孔到圆柱的内外参映射

- **存在形式**：原始四张图依旧保持 `view["camera_intrinsics"]=K_i` 与 `view["camera_pose"]=T_{world<-cam_i}`，以便 Remap 计算；新增的全景 view 则只保留 `agent_pose`（车心）以及描述圆柱的参数：`view["virtual_camera"] = dict(model="cyl", fx=W/(2π), fy=W/(2π), cx=W/2, cy=H/2, fov_vertical=...)`。  
  - 在 `BaseDataset` 层实现一个 `build_view_dict(agent_views)`：首先读取 4 个针孔 view，执行 warp，最后输出一个新的 view，其 `img` 为全景图、`camera_pose` 为 agent pose、`camera_intrinsics` 替换为虚拟参数。  
  - `ray_directions_cam` 不再依赖针孔参数，而是由新的 `get_cylindrical_pointmaps_and_rays_info(img.shape, virtual_camera)` 生成；若我们暂不提供圆柱射线，可直接设置 `ray_dirs_prob=0`，这样 MapAnything 仍会把全景图当普通纹理。
- **输出兼容**：`postprocess_model_outputs_for_inference` 默认会读取 `ray_directions` 并调用 `recover_pinhole_intrinsics_from_ray_directions`（`utils/inference.py:327-365`，`utils/geometry.py:304+`）强行恢复针孔 `fx, fy, cx, cy`。当 view 标记为圆柱时，需要在该流程前检查 `camera_model`，跳过针孔恢复或改用 `recover_cylindrical_parameters`，否则会导出错误的 3×3 内参并影响可视化/评估。
- **输入端保持 4 份原始针孔内参/外参**：Remap 表计算阶段仍需要精确的 `K_i` 与 `T_{car->cam_i}`。这些参数可直接从 OPV2V YAML 读取，不需要改变 MapAnything 现有的字段结构。
- **虚拟圆柱“内参”表示**：虽然圆柱相机不再用 3×3 针孔矩阵，我们仍可构造一个等效参数组（例如存储 `fx=fy=W_cyl/(2π)`, `cx=W_cyl/2`, `cy=H_cyl/2`, 外加一个 `camera_model="cyl"` 标志）。  
  - 与 `get_absolute_pointmaps_and_rays_info` 兼容的路径：将射线生成逻辑抽象成工厂函数；对 `camera_model="pinhole"` 使用原逻辑，对 `"cyl"` 使用新函数返回 `ray_directions_cam`。  
  - 2D→3D 映射完全由 Remap 表控制，因此我们可以先在单元测试里验证：将立方体点云投影到 4 个针孔，再通过 Remap/反投影恢复，误差 < ε（例如 <1px）。只要测试通过，即表示新的圆柱参数没有破坏 `MapAnything` 依赖的字段。
- **兼容破坏评估**：  
  1. 在切换到圆柱前，保存原始针孔路径的特征统计（例如 `ray_directions_cam` 的分布、`depth_along_ray` 平均值）。  
  2. 替换为圆柱路径后对同 batch 做一次前向，确认新增字段没有触发断言（`base_dataset.py:573-585`）且数值不会出现 NaN。  
  3. 通过脚本检查 `camera_pose_quats/trans` 是否仍与 agent 级 pose 相符；如果 Remap 只修改图像而 Pose 没改，那么 `_encode_and_fuse_cam_quats_and_trans` 的输出仍与旧流程一致，实现完全前向兼容。

### 4.2 方案二：保留针孔内参 + 仅提供车内相对外参

- **存在形式**：每张图仍携带原始 `K_i` 与 `T_{world<-cam_i}` 真值，用于训练阶段 supervision 或调试；另外新增 `agent_pose_prior`（可为空）与 `rel_pose_to_agent`。  
  - `agent_pose_prior`：车辆中心在世界坐标的 SE(3)，推理时可来自 GPS/Odometry，也可能带噪声；若未提供就设置标志位并在模型内跳过使用。  
  - `rel_pose_to_agent`：4 个固定环视与车心之间的 4×SE(3)，噪声小且可视为常量，可在 Encoder 融合模块中生成方向 embedding。  
- **可选输入设计**：  
  1. 在模型配置加入 `use_agent_pose_prior`、`use_rel_pose_prior` 两个开关；Forward 时根据布尔值决定是否把这些 token 注入 `_encode_and_fuse_cam_quats_and_trans` 或新的跨视角融合层。  
  2. 对噪声较大的 `agent_pose_prior`，可以在编码前附加一个 learnable 偏移估计器（例如一个 MLP 学习修正量），即便输入误差大也能逐步自适应；当输入缺失时，直接跳过该分支，保证行为与无先验时一致。
- **先验输入格式**：  
  1. **可选 Pose Token**：定义 `pose_priors = { "agent_pose": T_agent, "local_rel": [T_{cam_i<-agent}] }`。编码时若只提供 `agent_pose`，就退化为方案一的行为；若 `local_rel` 也提供，则在 early fusion 模块中生成 view embedding（例如把 `T_{cam_i<-agent}` 转换成方向向量/方位角），实现“能用就用”的前向兼容。  
  2. **内参缓存**：将 4 个 `K_i` 存为 `view["pinhole_intrinsics"][i]`，在 feature-level 融合模块需要重建射线或方位编码时再调用；若模块未启用，MapAnything 依然可以忽略这个键。
- **输出调整**：  
  - 即便不知道车间全局姿态，最终预测仍需回到世界系。我们可以延续现有流程：Transformer 输出 `n` 个 agent 级场景表示 + `n` 个 pose/scale token，后处理阶段再根据可用先验（如果 `agent_pose_prior` 可用就直接套用，否则仍靠模型预测）恢复到世界坐标。  
  - 通过引入软开关，输出接口保持一致：当没有任何先验时，系统回退到纯数据驱动；当提供 noisy 先验时，模型利用 learnable 校准项融合它们。
- **破坏评估方法**：  
  1. 在没有跨车位姿的情况下，确保 `per_sample_cam_input_mask` 只针对 “提供了 agent pose” 的视角置 True，避免 `_compute_pose_quats_and_trans_for_across_views_in_ref_view` 使用无意义的跨车相对位姿。  
  2. 使用单车数据跑一次 `forward`，比较与原模型输出差异（例如 L2 差值、Pose 误差）；只有当额外先验开启时才允许输出变化。

---

## 5. 非侵入式改造建议清单

1. **Dataset 层**  
   - 通过新增类实现圆柱拼接或视角分组，不直接修改 `BaseDataset`。  
   - 若需要给 view 添加额外字段（如 `camera_model`、`agent_id`），利用 Python 字典即可，`BaseDataset` 只检查少数关键键。`OPV2VCylindricalDataset` 已示范如何在继承类里聚合视角并复用父类工具链。
2. **几何工具**  
   - 在 `mapanything/utils/geometry.py` 中新增函数而不是改动原函数，避免影响现有针孔路径。  
   - 使用工厂函数或小型注册表，根据 `camera_model` 选择合适的射线生成逻辑。
3. **配置驱动**  
   - 把 `num_views`、`geometric_input_config` 等改动写入新的 Hydra 配置或 YAML，训练脚本通过配置切换 agent 视图，不破坏默认单车流程。
4. **渐进式开关**  
   - 每次引入新模态（圆柱射线、深度、Pose token）都配一组可控的 `*_prob`，方便做 ablation。

---

## 6. 现有流程的注意点

| 环节 | 需要关注的细节 | 影响 |
| --- | --- | --- |
| `_getitem_fn` | 期望 `view["depthmap"]` 形状与 `img` 匹配，并在结束前 `[..., None]` 扩维；忘记会触发断言 (`base_dataset.py:573-585`). | 数据加载报错 |
| `get_absolute_pointmaps_and_rays_info` | 假设针孔、无 skew。若传入圆柱图像必须绕开或自实现。 | 射线错误、NaN |
| `_encode_and_fuse_optional_geometric_inputs` | 会按随机 `prob` 决定哪些视角提供射线/深度/pose；若我们只提供 agent-level Pose，需要确保 mask 与 view 数一致。 | 几何 token 对不上 |
| `normalize_pose_translations` | 均值缩放依赖 view 数量；将 `num_views` 改为 n 后要确认所有 config（特别是 Transformer 层的 `views` 数）同步更新。 | Pose token 缩放异常 |
| `postprocess_model_outputs_for_inference` | 默认把 `ray_directions` 拟合成针孔 3×3 内参；如果 view 是圆柱/球面，需要跳过或改用自定义恢复函数。 | 输出内参错误、后处理崩溃 |

---

## 7. 建议的实施顺序

1. **圆柱拼接 PoC**  
   - 搭建 Remap 表，生成单车全景 + 车心 Pose；`geometric_input_config` 先关闭射线/深度。  
   - 验证 `4n → n` 后是否收敛更快，记录 loss/pose 误差。
2. **补充几何信息**  
   - 在圆柱路径中实现射线/深度 warp，重新开启 `ray_dirs_prob`、`depth_prob`。  
   - 若仍有收敛瓶颈，再考虑在 Transformer 输入端增加 agent-level view embedding。
3. **探索 Feature-Level Fusion**  
   - 先做“共享编码 + 轻量融合”版本，只在 ViT 前增加一个跨视角 attention 层，以最小改动验证可行性。  
   - 若效果显著，再评估更深度的 encoder 改写。

---

结论：对于低算力、希望尽快验证多车 Pose 稳定性的场景，优先推进 **方案一（圆柱全景）**，通过 dataset 层的几何预处理把 `4n` 视角压成 `n` 车辆，再逐步补充射线/深度信息。方案二在工程与算力成本更高，适合后续迭代，以提升端到端融合能力。

---

## 8. 当前微调策略（2025-12-07 更新）

1. **损失与日志**  
   - `configs/loss/opv2v_pose_loss.yaml` 启用 pairwise pose loss，`pose_quats_loss_weight=pose_trans_loss_weight=3`，其余几何项保持 1，并保留 `0.3 * NonAmbiguousMaskLoss`。  
   - `NonAmbiguousMaskLoss` 额外记录 `pred_mask_mean`、`gt_mask_mean`，训练时需关注该指标（目标 ≥0.2），否则重新检查圆柱 mask 或调整 BCE `pos_weight`。
2. **训练命令**  
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   PYTHONPATH=$(pwd) conda run -n mapanything_ft python -m torch.distributed.run --standalone --nproc_per_node=10 \
       scripts/train.py machine=local3090 model=mapanything \
       dataset=opv2v_cyl_coop_ft_5gpu dataset.num_workers=12 dataset.principal_point_centered=true \
       train_params=opv2v_cyl_coop loss=opv2v_pose_loss \
       model.model_config.pretrained_checkpoint_path=/home/qqxluca/map-anything3/experiments/mapanything/training/opv2v_cyl_coop/20251201_021942/checkpoint-best.pth \
       model.encoder.gradient_checkpointing=true \
       model.info_sharing.module_args.gradient_checkpointing=true \
       model.pred_head.gradient_checkpointing=true \
       hydra.run.dir=/home/qqxluca/map-anything3/experiments/mapanything/training/opv2v_cyl_posefix/${now:%Y%m%d_%H%M%S} \
       hydra.job.name=opv2v_cyl_posefix
   ```
   - `max_num_of_imgs_per_gpu` 需保持 4；若降到 2，验证 batch 将变 0（`training.py:90`）。如需更小 batch，请先释放显存或减少 `num_views`。
3. **资源状态**  
   - `nvidia-smi` 显示 `/home/gongyan/anaconda3/envs/UniMODE/bin/python` 正占用 10×3090（约 14 GB/卡），导致 `torch.load` 阶段或第一轮 backward 报 `CUDA error: CUBLAS_STATUS_ALLOC_FAILED`。需要等待该任务结束或沟通共享策略。
4. **监控 Checklist**  
   - `len(loader_train)` 应为 5,000（`opv2v_cyl_coop_ft_5gpu`）；若变 4,000 说明 dataset 配置未切换。  
   - `FactoredGeometryScale..._pose_trans_avg`（验证）需逐步降至 <5；若常驻 7~8，请结合点云摘要分析 Pose 问题。  
   - `checkpoint-last.pth` 的 `epoch`、`global_step` 应保持单调递增，防止 resume 后学习率曲线错乱。

## 9. 点云可视化与截图规范

1. **推理脚本**
   ```bash
   PYTHONPATH=$(pwd) conda run -n mapanything_ft python scripts/render_pointcloud_html.py \
       --config scripts/local_infer_config.json \
       --checkpoint /home/qqxluca/map-anything3/experiments/mapanything/training/opv2v_cyl_coop/20251201_021942/checkpoint-best.pth \
       --output-dir eval_runs/opv2v_cyl_precheck \
       --view-output-dir eval_runs/opv2v_cyl_precheck/val_idx0_views \
       --debug-summary-json eval_runs/opv2v_cyl_precheck/val_idx0_summary.json \
       --val-index 0 \
       --point-pose-source pred
   ```
2. **产物说明**  
   - `val_idx0_pointcloud.html`：Plotly 点云窗口，默认可全屏与缩放。  
   - `val_idx0_summary.json`：记录预测 pose/scale/depth 的平均值与 RMSE；`val_idx0_summary_gtpose.json` 为 GT pose 对照。  
   - `val_idx0_views/*.png`：每辆车输出彩色点云、深度、mask 截图，命名包含 `agent_id`；仅供自查，无需手动截图分享。  
   - `val_idx0_views/*.npy`：缓存稀疏/稠密深度、pose Tensor，方便 Notebook 二次分析。
3. **调试流程**  
   - 先渲染 `--point-pose-source pred`，若点云形变严重，再渲染 `--point-pose-source gt` 对比定位问题。  
   - `val_idx0_summary.json` 中 `pred_mask_mean < 0.05` 时，优先回查训练日志和圆柱 mask；`pose_rmse` 大则重点关注 `opv2v_pose_loss` 设定。  
   - 每个新 checkpoint 建议即刻生成一套 HTML/PNG/JSON，并把路径记录到 `docs/cyl/opv2v_cyl_coop_debug_plan.md`，便于追溯。
4. **访问方式**  
   - 服务器端可用 `python -m http.server 8000` 暴露 `mapanything_ft` 目录，然后在浏览器打开 `http://<server-ip>:8000/mapanything_ft/eval_runs/opv2v_cyl_precheck/val_idx0_pointcloud.html`；刷新即可看到最新结果。  
   - 若需共享给他人，可把 HTML 链接 + 对应 checkpoint 写入 README 或 PR 描述，避免重复描述问题。
