# 需求与问题综述（OPV2V / VGGT / MapAnything）

> 入口：请优先阅读 `docs/master_plan.md`（唯一持续维护的执行计划）和 `docs/goal_based_test_plan.md`（验收标准）。

## 1. 目标与验收标准

### 1.1 核心目标
- 在 OPV2V 协同数据上，把 **pose / depth / scale** 三类误差同时压低，并保持稳定。
- 训练过程要能持续运行、自动监督、不中断；任何阶段必须以指标达标为停止条件。

### 1.2 硬指标（阶段门槛）
- 以 **full test** 的指标为主；1k 测试为快速参考。
- **Scale 目标（OPV2V 正确口径）**：使用 `scale_to_gt_*`（ratio-to-GT），而不是 legacy `scale_err_mean=|s-1|`。
  - 直观口径（推荐记录）：`scale_to_gt_mult_err_mean = exp(scale_to_gt_log_err_mean)`，理想值 **1.0**。
  - 阶段目标：`scale_to_gt_mult_err_mean <= 1.3`（约 <=30% 典型比例误差）。
  - 解释：`scale_to_gt_log_err_mean` / `scale_to_gt_eq_rel_err_mean` 可从 eval summary 直接读取。
- **Pose / Depth**：不能出现明显倒退（约束：相对基线不劣化 >10%）。

### 1.3 过程约束
- 训练任务需要 **持续运行**，未达指标不停止。
- 需要 **定期汇报** 进度（当前实际实现为日志文件输出）。
- 资源需要 **尽量满负载使用**（本机 4x4090，远端 2x8 A800 作为可能的扩展资源）。

## 2. 当前资源与运行状态

### 2.1 资源分布
- 资源是动态的（本机 4×4090 / 单机 8×A800 / 火山 2×8×A800 都出现过）。
- 计划应以“固定合同评测 + 可复现实验产物”为核心，不依赖某一台机器的固定路径/状态（入口脚本已做多挂载点探测）。

### 2.2 已部署的自动化
- 过去存在自动顺序训练/重启脚本（S1→S2→S3 loop），但其 gate 基于 legacy `scale_err_mean`，会误导 OPV2V 的 scale 结论。
- 现阶段 **必须** 以 fixed-contract `batch_eval.py` 的 `scale_to_gt_*` 为主口径；旧 loop 仅保留作历史参考（不建议继续使用）。

## 3. 数据来源与结构

### 3.1 OPV2V 数据
- 图像根目录：`map-anything/data/opv2v_images`
- 深度根目录：`map-anything/data/opv2v_depth`（稀疏深度 `.npy`）
- 目录层级：`{split}/{sequence}/{agent}/{frame}_{cameraX}.png|.yaml` + 对应深度

### 3.2 VGGT 相关数据配置
- VGGT 不是数据集，而是几何模型；在 OPV2V 上做微调以恢复尺度。
- 轻量级微调配置：
  - 单车 4 视角：`configs/dataset/opv2v_ft_vggt_1k.yaml`
  - 协同 2 车 8 视角：`configs/dataset/opv2v_coop_ft_2a8v_vggt_500.yaml`
- 正式训练使用的是全量协同配置：`configs/dataset/opv2v_coop_ft_2a8v_full.yaml`

## 4. 指标与术语对照

### 4.1 Scale 相关指标
- **`scale_to_gt_err_mean`**：`mean(|s/g - 1|)`（ratio-to-GT），目标接近 0。
- **`scale_to_gt_log_err_mean`**：`mean(|log(s/g)|)`，强调比例差（对称对待 0.5x 与 2x）。
- **`scale_to_gt_eq_rel_err_mean`**：`exp(scale_to_gt_log_err_mean) - 1`（等价相对误差）。
- **`FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale`**：损失空间的 scale loss，不等价于尺度误差（不能直接和 `scale_err_mean` 对比）。
- **`scale_valid_ratio_avg`**：有效深度监督比例（稀疏深度导致通常只有 3–4%）。

### 4.2 其他核心指标
- `pose_trans_l2_m`：平移误差（m）。
- `pose_rot_deg`：旋转误差（deg）。
- `depth_z_mae_m / depth_z_rmse_m`：深度 Z 误差。

### 4.3 关键损失项
- `direct_scale_loss`：直接回归尺度比例（ratio）。
- `loss_in_log`：在 log 空间回归尺度（更符合比例误差）。
- `norm_mode`：是否对 pose/scale 做规范化（影响尺度的“是否固定在 metric 级别”）。

## 5. 当前观察到的问题

### 5.1 Scale 始终不收敛
- 过去的很多训练记录里出现的 `scale_err_mean` 是 legacy `|s-1|`，对 OPV2V 不应作为尺度质量的主判断依据。
- 正确评估应使用 `scale_to_gt_*`（需要 eval summary/contract backfill 支持）。
- 典型现象：pose/depth 有所收敛，但 scale 仍显著偏离。

### 5.2 监督过稀
- `scale_valid_ratio_avg` 只有 ~0.03–0.04（3–4% 有效深度），意味着 scale loss 几乎只有很少像素提供梯度。
- 这是 scale 难以收敛的主要原因：**监督稀疏 + 噪声比例高**。

### 5.3 指标失真 / NaN
- 若日志出现 `loss: nan` 或 `scale_err_mean=0`，表示该次 eval 失效。
- 这类结果必须剔除，不能用于判断训练效果。

### 5.4 资源与路径问题
- A800 节点路径不一致导致入口失败（训练命令在远端不可用路径）。
- 需要统一 `map-anything` 根目录路径或在入口命令中显式设置。

## 6. 已采取的措施

### 6.1 训练流程调整
- **Stage1 → Stage2 → Stage3** 顺序训练脚本已部署：
  - Stage1：scale-only warmup
  - Stage2：full loss（w0.2）
  - Stage3：full loss（w1.0）
- 训练中记录 `scale_valid_ratio_avg` 以确认监督稀疏程度。

### 6.2 Loss 配置扩展
- 新增 “无 conf loss” 的 loss config，避免 conf 带来的 NaN。
- 新增无 gating 的变体以排查 mask gating 的过度过滤。

### 6.3 工程稳定性修复
- 设置 `TMPDIR` 规避 AF_UNIX 路径过长导致的异常。
- 监控/重启脚本确保训练不中断。

## 7. 当前计划的关键假设与待验证点

### 7.1 深度稀疏是主要瓶颈（高优先级）
- 假设：scale 监督稀疏是核心原因。
- 验证方向：
  - 生成 **更稠密的深度**（`opv2v_depth_dense`）
  - 在相同训练设置下观察 `scale_to_gt_*` 是否显著下降（优先 `scale_to_gt_log_err_mean` / `scale_to_gt_mult_err_mean`）。

### 7.2 需要尺度校准或一致性约束
- 可能需要增加 **scale calibration**（全局比例因子）或 **跨车一致性 loss**。
- 目的：减轻稀疏监督导致的尺度漂移。

### 7.3 不能仅调 loss 权重
- 仅增大 scale_loss_weight 已被证明不足以解决问题。
- 需要结合数据监督密度与一致性约束。

## 8. 当前执行位置（概况）

- 已完成：固定合同（Test500）下的几何 ckpt 重评测 + 指标/可视化一致性审计 + baseline 选定。
- 当前 baseline：`experiments/local_runs/a800_sfix_direct/pinhole_pose_depth_scale/checkpoint-best.pth`
- 基线问题：single 尚可但 under-scale；coop 存在明显 tail 崩溃（见 tail 诊断报告）。

## 9. 下一阶段明确行动

### 9.1 数据与监督
- 先做“结构性实验”验证 scale 是否被几何损失耦合拉偏（detach scale from geom losses）。
- 若验证有效，再推进 densified depth / pseudo-depth 以提升监督密度，并在同合同下复核泛化。

### 9.2 损失形态与校准
- 评估 log-scale vs ratio-scale 的收敛效果。
- 引入全局 scale calibration，减少系统性偏移。
- 增加跨车 scale consistency loss（同一时刻多车尺度一致）。

### 9.3 资源调度
- 重新恢复 A800 两台 8 卡节点的训练入口命令与路径一致性。
- 将全量训练分配到 A800，4090 用于快速验证和 ablation。

## 10. 验收与停机条件

- 只有当 **full test** 满足：
  - `scale_to_gt_mult_err_mean <= 1.3`（或同等 `scale_to_gt_*` 阈值）
  - pose/depth 不退化
- 才允许停止训练。
- 若不达标，必须继续执行计划下一阶段，不可中止。
