# OPV2V 多车协同修复执行计划（2026-03-11）

> 目标：把多车协同训练从“点云完全不可用”推进到“训练稳定、点云对齐正常、可以持续迭代”的状态。

## 总目标

1. 找出多车协同失败的真实根因，而不是只做表面修补。
2. 先让训练链路 **稳定跑通**，再逐步提高精度。
3. 通过并行实验比较：
   - 基线结构化协同训练
   - 近邻车辆 curriculum 训练
   - 高分辨率精修训练
4. 最终得到一版可持续复现的多车协同训练方案，并把每个有效版本提交到 GitHub。

---

## 第一阶段：根因排查与最小修复

- [x] 阅读并梳理 `mapanything/datasets/opv2v.py`、`configs/loss/*`、`configs/dataset/*`、`MapAnything` 模型代码
- [x] 确认多车失败不是单一“距离太远”，而是以下问题叠加：
  - [x] 数据采样把多车多相机视角打平，破坏同车 4 相机 rig 结构
  - [x] 模型缺少 `agent/camera` 身份信息
  - [x] 协同损失没有显式启用 pairwise relative pose loss
  - [x] 训练时存在显存与磁盘写满两个运行层阻塞
- [x] 把结论记录到文档中，避免后续重复排查

---

## 第二阶段：代码级修复

### 2.1 数据层修复
- [x] 为 `OPV2VCoopDataset` 增加结构化采样能力
- [x] 支持以下采样参数：
  - [x] `structured_sampling`
  - [x] `agents_per_sample`
  - [x] `views_per_agent`
  - [x] `agent_selection_policy`
  - [x] `max_agent_distance`
  - [x] `require_complete_rig`
  - [x] `shuffle_views`
  - [x] `emit_identity_metadata`
- [x] 确保 2 车样本按“4 相机 + 4 相机”返回，而不是无序视角集合
- [x] 修复 `BaseDataset` 对动态/结构化视角数量的断言兼容性

### 2.2 模型层修复
- [x] 在 `MapAnything` 中增加 `agent_identity_embedding`
- [x] 在 `MapAnything` 中增加 `camera_identity_embedding`
- [x] 在 encoder feature 上注入身份偏置，再送入 info-sharing 模块
- [x] 兼容旧 checkpoint 加载，允许 identity embedding 权重缺失

### 2.3 损失与配置修复
- [x] 新增协同训练损失配置 `configs/loss/opv2v_coop_pairwise_loss.yaml`
- [x] 开启 `compute_pairwise_relative_pose_loss=True`
- [x] 开启 `compute_world_frame_points_loss=True`
- [x] 新增结构化协同数据配置：
  - [x] `configs/dataset/opv2v_coop_ft_structured.yaml`
  - [x] `configs/dataset/opv2v_coop_ft_structured_debug.yaml`
- [x] 新增协同模型配置：
  - [x] `configs/model/mapanything_opv2v_coop.yaml`
  - [x] `configs/model/mapanything_opv2v_coop_lowmem.yaml`
- [x] 新增低显存 smoke / stageA / stageB 训练配置：
  - [x] `configs/train_params/opv2v_coop_v1_smoke.yaml`
  - [x] `configs/train_params/opv2v_coop_stagea.yaml`
  - [x] `configs/train_params/opv2v_coop_stageb.yaml`
- [x] 修复原有 `configs/dataset/opv2v_coop_ft.yaml` 的验证 split 问题

### 2.4 训练基础设施修复
- [x] 修复 `inference.py` 对 batch 中 identity 字段的兼容性
- [x] 为 `save_on_master` 增加原子保存逻辑，避免残缺 checkpoint 覆盖正式文件
- [x] 增加可调分布式初始化 timeout，避免长时间挂死

---

## 第三阶段：运行环境排障

- [x] 检查远程服务器各卡显存占用情况
- [x] 发现之前 OOM 的直接原因是其他任务占满 GPU，而不是当前代码本身必然无法训练
- [x] 清空整机 GPU，拿到可用 24GB 3090
- [x] 发现根盘 `/` 写满，导致 checkpoint 保存失败并留下 `.tmp`
- [x] 将后续实验输出改写到大盘目录：
  - [x] `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments`

---

## 第四阶段：已完成的验证实验

### 4.1 Smoke 验证（低分辨率、低显存）
- [x] 成功运行 `224x126 / 8 views / encoder frozen / checkpointing on`
- [x] 成功完成训练 + 验证 + 保存 checkpoint
- [x] 生成以下文件：
  - [x] `checkpoint-last.pth`
  - [x] `checkpoint-1.pth`
  - [x] `checkpoint-best.pth`
- [x] 确认最大显存约 `8.6GB`
- [x] 结果目录：
  - [x] `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_v1_smoke_run2`

### 4.2 原始目标配置验证（448 分辨率）
- [x] 成功运行 `448x252 / 8 views / encoder trainable`
- [x] 确认不再 OOM
- [x] 成功完成 1 个 epoch 的训练、验证、保存 checkpoint
- [x] 结果目录：
  - [x] `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_v1_debug_fullgpu`

### 4.3 GitHub 版本落地
- [x] 已提交当前修复版本到 git
- [x] 已推送到 GitHub 分支：`opv2v-coop-multicar`
- [x] 提交号：`367cae2`
- [x] 仓库：`insomniacs01/map`

### 4.4 单卡高吞吐校准（2026-03-11 下午）
- [x] 已确认此前 GPU 没跑满的直接原因不是代码空转，而是 `max_num_of_imgs_per_gpu=8` 且 `num_views=8`，实际只形成 `1 sample / GPU step`
- [x] 已完成 `16 / 24 / 32 / 64 / 96 / 128 / 160` 档位探针
- [x] 已确认 `176 / 184 / 192 / 224` 会 OOM
- [x] 当前安全稳定上限为 `max_num_of_imgs_per_gpu=160`
- [x] `160` 档位稳定运行约 30+ step，显存约 `19.7GB`，单步时间约 `2.8~2.9s`
- [x] 使用 `160` 重启正式长跑并重新观察收敛

### 4.5 验证阶段 OOM 修复（2026-03-11 傍晚）
- [x] 已确认崩溃点不在训练阶段，而是在 epoch 结束后的验证阶段
- [x] 已定位根因：`training.py` 默认把验证 batch 设置成训练 batch 的 `2x`，在 `8 views` 协同设置下显存直接顶到约 `22.3GB` 后 OOM
- [x] 已新增 `max_num_of_test_imgs_per_gpu`，允许训练和验证分别控制显存预算
- [x] 已将 `opv2v_coop_stagea_long` 的验证上限改为 `160 images / GPU`，与训练一致
- [x] 已验证基线 `bs160` 恢复后可以完整跑过 `26` 个验证 batch，并继续进入 `Epoch 1` 训练
- [x] 已验证 `near40 bs160` 恢复后也可以完整跑过 `26` 个验证 batch，并继续进入 `Epoch 1` 训练

---

## 第五阶段：当前正在执行的训练任务

### 5.1 Stage A：结构化协同基线训练
- [x] 已启动
- [x] 已进入训练 iteration
- [ ] 跑完整个 3 epoch
- [ ] 保存并检查 `checkpoint-best.pth`
- [ ] 记录验证 loss 曲线和收敛速度
- 目录：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_stagea_run1`

### 5.2 Stage A-近邻版：加入车距 curriculum（`max_agent_distance=40`）
- [x] 已启动
- [x] 已进入训练 iteration
- [ ] 跑完整个 3 epoch
- [ ] 对比它和 Stage A 基线的验证表现
- [ ] 判断“先训近邻车再扩展距离”是否有效
- 目录：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_stagea_near40_run1`

### 5.3 Stage B：高分辨率精修训练
- [x] 已启动
- [x] 已进入训练 iteration
- [ ] 跑完整个 3 epoch
- [ ] 确认高分辨率训练不会重新引入显存问题
- [ ] 比较精修前后的深度边界、位姿和尺度表现
- 目录：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_stageb_resume1`

### 5.4 Stage A-long：单卡高吞吐基线（`bs160`）
- [x] 已启动
- [x] 已进入训练 iteration
- [ ] 跑完整个 8 epoch
- [ ] 记录中期验证 loss 与最终 `checkpoint-best.pth`
- [ ] 与低吞吐版本比较收敛速度和点云稳定性
- 目录：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_stagea_long_bs160_run1`

### 5.5 Stage A-long-近邻版：高吞吐 `near40` 对照（`bs160`）
- [x] 已启动
- [x] 已进入训练 iteration
- [ ] 跑完整个 8 epoch
- [ ] 对比它和 `bs160` 基线的验证表现
- [ ] 判断“先吃满单卡再做 near40 对照”是否带来更稳定点云
- 目录：
  - `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_stagea_long_bs160_near40_run1`

---

## 第六阶段：下一步必须完成的工作

### 6.1 指标比较
- [ ] 对比以下实验的验证集指标：
  - [ ] Stage A 基线
  - [ ] Stage A 近邻版
  - [ ] Stage B 精修版
- [ ] 重点看：
  - [ ] pose translation
  - [ ] pose quaternion
  - [ ] scale
  - [ ] depth / pts3d / cam_pts3d

### 6.2 点云可视化验收
- [ ] 从每条实验中选取相同场景做点云可视化
- [ ] 检查是否仍存在：
  - [ ] 跨车点云错位
  - [ ] 大尺度飘移
  - [ ] 同车 4 相机内部撕裂
  - [ ] 不同车辆之间完全无法融合
- [ ] 明确哪条训练路线的点云最接近“可用版本”

### 6.3 训练策略决策
- [ ] 若 Stage A 基线更稳：继续延长 Stage A，再转 Stage B
- [ ] 若 Stage A 近邻版更稳：采用 curriculum 路线，逐步放宽 `max_agent_distance`
- [ ] 若 Stage B 直接最好：以 Stage B 为主线继续精修
- [ ] 最终确定“正式主线配置”

### 6.4 GitHub 版本管理
- [ ] 每得到一版确认更优的 checkpoint 和配置，就提交一次 GitHub
- [ ] 保证每次提交都能说明：
  - [ ] 用了哪套 config
  - [ ] 训练了多久
  - [ ] 指标是否改善
  - [ ] 点云是否更稳定

---

## 第七阶段：可能的后续增强（尚未执行）

### 7.1 更强的 curriculum
- [ ] 先训近邻 20m
- [ ] 再训 40m
- [ ] 再放开所有距离

### 7.2 更强的结构约束
- [ ] 增加同车 4 相机 rig 一致性约束
- [ ] 进一步强化跨车相对位姿监督

### 7.3 更可靠的评估工具
- [ ] 检查 `scripts/batch_eval.py` 是否可直接复用
- [ ] 如果现有脚本不稳定，补一个最小可用的多车点云导出/可视化脚本

---

## 当前判断

目前已经可以明确认为：

- [x] 多车协同训练的主干方向已经修正正确
- [x] “结构化采样 + 身份嵌入 + pairwise loss” 不是猜想，而是已经跑通的有效修复
- [x] 原始 448 分辨率协同训练在空闲 24GB 3090 上是可以稳定跑的
- [ ] 还没有最终证明哪条训练路线的点云最好
- [ ] 还没有拿到最终“完全没问题”的正式版本 checkpoint

因此，当前状态不是“问题还没动”，而是已经从 **不能训练 / 训练即坏** 进入到 **多条可运行训练路线并行比较** 的阶段。
