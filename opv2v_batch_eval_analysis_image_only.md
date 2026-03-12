# OPV2V 抽样评估（Image-Only 输入修复版）

> 评估脚本：`scripts/batch_eval.py`。  
> - 环境：`mapanything_reconstructed`（Torch 2.6 / CUDA 12.4）。  
> - 运行日期：2025-11-26。  
> - 关键改动：在预处理完成后调用 `strip_external_calibration_inputs` 并手动移除 `depth_z`，确保推理阶段只接受图像输入；真值深度仅用于指标计算。  
> - 命令：同时启动三条进程，占用 GPU `1/4/5`，分别评估 pretrain / stage1 / stage2：
>
> ```bash
> CUDA_VISIBLE_DEVICES={1,4,5} python scripts/batch_eval.py \
>   --split test \
>   --frames_json /media/.../opv2v_batch_eval/summary_test.json \
>   --output_root /media/.../opv2v_batch_eval_colorfix \
>   --device cuda --save_representative \
>   --pc_metrics --pc_filter_z_min -1 --pc_filter_z_max 4.2 --pc_filter_radius 120 \
>   --model_filter {pretrain|stage1|stage2}
> ```
>
> 所有输出（CSV / JSON / PCD）均保存在 `/media/.../opv2v_batch_eval_colorfix`。

## 0. 回顾与动机

- 11 月前两版汇报（`docs/opv2v_batch_eval_analysis.md`、`docs/opv2v_eval_report.md`）针对 CPU/GPU run 给出了“单车 3cm、协同 20cm”级别的数值，但脚本在预处理阶段默认把 `depth_z` 与外参矩阵一并喂入模型。  
- 这些 run 与推理端（color_compare / 上板 demo）“只给 RGB + intrinsics” 的设定不一致，导致指标高估，尤其是预训练模型的协同模式。  
- 本次严格执行 image-only 流程，既是对历史结论的回溯，也是为了回答老师“为何实验室展示和汇报数字存在落差”的疑问：我们需要证明真正上线时的误差水平，并找出偏差来源。  
- 因此，以下分析默认只与“相同输入模式”的 run 做对比：CPU/CUDA 报告可作为参考背景，但若配置不同必须注明原因。

## 1. 抽样帧概览

固定 10 帧，皆来自 test split 且包含 ≥2 车辆，帧列表记录在 `eval_runs/opv2v_batch_eval_colorfix/summary_test.json`。

| 序列 | 帧 | 主车/协同 |
| --- | --- | --- |
| 2021_08_20_21_10_24 | 000113, 000223 | 1996 + 2005 |
| 2021_08_22_07_52_02 | 000195 | 3477 + 3486 |
| 2021_08_23_12_58_19 | 000266 | 357 + 366 |
| 2021_08_23_15_19_19 | 000156, 000182, 000332 | 8690 + 8699 |
| 2021_08_24_20_09_18 | 000280 | 3174 + 3183 |
| 2021_08_24_20_49_54 | 000200, 000262 | 207 + 216 |

> 与 11 月 CPU/GPU 报告不同，本次使用 `summary_test.json` 中固定的 10 帧子集：为保证可复现性，我们复用了 color_compare 里验证过的场景，并对每帧都保存代表点云。帧列表与旧抽样并不完全重合，因此只能比较“趋势”而非逐帧指标。


## 2. 聚合指标（image-only 输入）

> 与 11 月 CPU/GPU 报告的主要区别：本次所有模型都在推理前剥离了 `depth_z` 与额外外参，因此只能与“真正 image-only” 的场景比较；此前 CPU/GPU run 的 0.03 m / 0.2 m 等数字仅能说明“模型在拿到外参时的效果”，不能视为可部署表现。

### 2.1 单车模式

| 模型 | Pose Abs (m) | Pose Rot (°) | Depth RMSE | Depth MAE | Depth Rel | Scale Err |
| --- | --- | --- | --- | --- | --- | --- |
| **Pretrain** | 6.35 | 18.59 | 15.19 | 6.99 | 0.38 | 10.31 |
| **Stage1（单车微调）** | **0.058** | **0.62** | **14.10** | **5.47** | **0.23** | **3.32** |
| **Stage2（多车微调）** | 0.16 | 1.02 | 14.87 | 6.71 | 0.35 | 16.59 |

> - 真正的 image-only 推理下，预训练模型的位姿误差飙到 6–22 m（`pretrain/single_metrics.csv`），证明历史上“CPU 抽样优异”完全是外参泄漏造成的。  
> - Stage1 单车仍保持 5~6 cm 的位姿精度；Stage2 单车退化到 15 cm / 1°，且 metric scaling 失控（`scale_err≈16.6`）。  
> - 深度/Chamfer/BEV 在三个模型之间差异不大，说明这个 10 帧样本的主要差距来自 pose/scale。  
> - 对比 11 月 GPU run（Stage1 单车 0.31 m、Stage2 单车 3.46 m）：Stage1 的真实表现反而更好，而 Stage2 则在剥离外参之后继续恶化，证实“Stage2 无法仅凭影像推理”的隐患。

### 2.2 协同模式

| 模型 | Pose Abs (m) | Pose Rot (°) | Depth RMSE | Depth MAE | Depth Rel | Scale Err |
| --- | --- | --- | --- | --- | --- | --- |
| **Pretrain** | 6.28 | 4.59 | 15.04 | 7.22 | 0.41 | 11.47 |
| **Stage1（单车微调）** | 10.85 | 12.66 | **14.16** | **5.72** | **0.26** | **3.31** |
| **Stage2（多车微调）** | **2.09** | **1.50** | 15.07 | 6.94 | 0.38 | 19.01 |

> - Stage2 coop 虽能把 pose_abs 压到 2 m，但尺度偏差 (19.0) 远高于预训练，导致深度/Chamfer 仍不理想。  
> - Stage1 coop 完全崩溃：多帧 pose_abs >10 m，旋转误差达 10° 以上。  
> - 预训练 coop 表现也大幅恶化（6 m / 4.6°），符合 color_compare 里的观测。  
> - 与 11 月协同 run（预训练 0.21 m、Stage2 6.4 m）相比，所有模型的误差都上了一个数量级——说明之前的“协同优势”来自信息泄漏，而非模型真正消化了多车输入。Stage2 仍在所有模型里最接近可用，但 2 m / 19 scale_err 离量产标准相差甚远。

### 2.3 点云与 BEV 指标

| 模型-模式 | Chamferᵣ | Chamferᶠ | BEV IoUᵣ | BEV IoUᶠ |
| --- | --- | --- | --- | --- |
| Pretrain-单车 | 9.64 / 3.18 | 8.08 / 8.71 | 0.145 | 0.025 |
| Stage1-单车 | **5.81 / 2.87** | 7.25 / 7.91 | 0.097 | 0.033 |
| Stage2-单车 | 6.44 / 2.80 | 7.06 / 7.66 | 0.144 | 0.045 |
| Pretrain-协同 | 12.46 / 3.13 | 7.45 / 8.10 | 0.162 | 0.036 |
| Stage1-协同 | **5.51 / 2.79** | 6.94 / 7.43 | 0.112 | 0.050 |
| Stage2-协同 | 11.78 / 2.95 | **6.59 / 7.04** | **0.161** | **0.065** |

（斜杠前为预测→GT，斜杠后为 GT→预测；过滤条件：`z∈[-1,4.2] m`、`r<120 m`。）

> - 由于输入改成纯图像，Chamferᶠ 和 BEV IoUᶠ 均较 11 月报告恶化 10%~25%；尤其 Stage1 coop 的 Chamferᶠ 由 3.14 升至 7.43，证明协同漂移直接把点云抛离地面。  
> - Stage2 coop 在 BEV IoUᶠ 上重新夺回领先（0.065 vs 0.033），说明多 agent 仍能提供部分检测友好度，但 Chamferᵣ=11.78 也揭示其大尺度平移带来的惩罚。

## 3. 典型场景

- **Pretrain（单车）**：`2021_08_22_07_52_02 / 000195` 出现 21.66 m / 7.38° 的误差（`pretrain/single_metrics.csv:3`），即使只有主车视角也完全漂移。  
- **Stage1（单车）**：同一帧只剩 3.6 cm / 0.28°（`stage1/single_metrics.csv:3`），验证微调确实学到了纯图像姿态。  
- **Stage1（协同）**：`2021_08_22_07_52_02 / 000195` 的 coop 误差 23.55 m / 14.7°（`stage1/coop_metrics.csv:3`），说明加入协同视角会破坏其对主车的建模。  
- **Stage2（协同）**：`2021_08_20_21_10_24 / 000113` 达到 0.62 m / 1.31°（`stage2/coop_metrics.csv:1`），是少数成功帧；但 `2021_08_23_12_58_19 / 000266` 仍有 8.68 m（`...:4`），且尺度误差 16+，表现不稳定。

## 4. 深度与尺度误差解读

1. **深度**：三种模型的 depth RMSE 都集中在 14–16 m，协同模式并未改善。Stage1 在协同输入下仍保持最低 RMSE (14.16)，显示其深度头对额外视角并不敏感。  
2. **尺度**：Stage1 维持 `scale_err≈3.3`，而 Stage2 单车/协同分别飙到 16.6 / 19.0，预训练也在 10~11 范围。这意味着 Stage2 的 metric scaling 分支急需重新约束。  
3. **检测指标**：Stage1/Stage2 在协同模式下的 BEV IoUᶠ >0.05，明显优于预训练 (0.036)，说明它们能把点云集中到可检测区域；但 Chamferᵣ 同时升高，体现出 pose 偏差带来的整体平移。

## 5. 汇报结论与后续规划

1. **修正历史认知**：CPU/CUDA 抽样报告中的“厘米级协同精度”来自外参泄漏；在真实 image-only 模式下，预训练模型即刻退化到 6 m / 4°。这解释了老师在 demo 中看到的严重漂移，也说明需要重新定义“合格基线”。  
2. **Stage1 = 稳定单车基线**：它在剥离外参后仍保持 5~6 cm 的绝对位姿和 3.3 的 scale_err，与之前 color_compare 的观感一致；但协同模式会破坏其学习到的主车几何，目前不能作为多车方案。  
3. **Stage2 = 协同独苗但尺度失控**：只在多车输入下略优于 Stage1，但 2 m+ 的位姿和 19 的尺度误差意味着无法直接部署。我们需要把 Stage1 的尺度知识或外部传感器先验蒸馏到 Stage2，否则多车微调的收益抵不过新的风险。  
4. **下一步动作**：  
   - 扩样：继续沿用 image-only 脚本，把样本数提升到 ≥20 帧，并按模式单独统计方差，拿到稳定分布而非单次均值。  
   - 定点剖析：对 Stage2 coop 的极端帧（`2021_08_23_12_58_19/000266` 等）输出逐相机对齐图，找出尺度漂移从哪台摄像头开始。  
   - 训练对齐：在 Stage2 训练脚本里加入 `strip_external_calibration_inputs` 开关或直接禁用外参输入，保证训练/推理模态一致，避免再出现指标与 demo 脱节。  
   - 回归集：以 `summary_test.json` 为基础维护长期 sample list，后续每次改动都跑这套 image-only 报告，形成 regression baseline。
