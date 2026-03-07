# OPV2V 微调阶段评估汇报（面向导师）

> ⚠️ **归档说明（2025-11-28）**：本文件保存 2025-11 CPU/GPU 抽样的历史记录，但其流程与当前标准（`docs/opv2v_batch_eval_analysis_image_only.md`）不一致，导致“Stage1/Stage2 仍可厘米级定位”的结论与最新 image-only 评测相冲突。主要差异：
> - **推理输入**：当时的 `scripts/color_compare.py` 仍把 YAML 外参和 `depth_z` 一同喂入模型，模型在推理时等价于拿到真值姿态；新版批评（`batch_eval.py`＋`strip_external_calibration_inputs`）强制只看 RGB + 内参。
> - **样本与统计**：这里的定量结果只涵盖少量 demo 帧（多为单帧渲染），并未锁定 `summary_test.json` 的 10 帧子集；无法与 image-only 报告直接对比。
> - **指标定义**：本报告中的“姿态误差 0.22 m/2.26°”等引用相对视角的即时输出；最新文档以绝对 `Pose Abs`/`Scale Err` 为准，并同步给出点云/BEV 过滤设置。
>
> 若需可部署场景下的真实误差，请参阅 `opv2v_batch_eval_analysis_image_only.md` 或 `opv2v_eval_report_image_only.md`；仅当需要还原 11 月前的调研脉络时，再参考本档案。

> 本文面向课题老师，回顾目前在本机 (`/home/qqxluca/map-anything3`) 针对 OPV2V 数据集已经完成的单车与协同微调，以及相应的可视化评估结果、现存问题与下一步计划。

## 1. 项目背景与目标
- **主线任务**：在 MapAnything 框架内，将 OPV2V 的单车微调（Stage1）与协同多车微调（Stage2）串联起来，并建立一套“随训随验”的可视化评估流程。
- **动机**：
  1. 已有的 `map-anything3` 仓库中保存了多次训练产出，但缺少系统化的复盘；我需要确认这些 checkpoint 是否可用于后续实验。
  2. 服务器通过 SSH 远程连接，Open3D GUI 不一定能弹出，因此需要摸索离线/无界面的可视化方案。
  3. 现有脚本里混入了大量示例/假数据，整理真实数据引用路径与复现实验的步骤刻不容缓。
- **本轮工作目标**：复现 Stage1、Stage2 推理 → 记录指标 → 保存可视化图 → 梳理痛点并输出下一步路线。

## 2. 阶段设定与训练条件
- **Stage1 = 单车微调**  
  - **数据**：`configs/dataset/opv2v_ft.yaml`（单 agent，4 视角，`resolution_train=518×392`，固定相机列表 `camera0-3`）。  
  - **命令**：`python -m mapanything.train.training machine=local3090 dataset=opv2v_ft ... train_params.max_num_of_imgs_per_gpu=4 lr=5e-5 epochs=5`（详见 `/home/qqxluca/map-anything3/docs/opv2v_coop_training_summary.md`）。  
  - **输出目录**：`experiments/opv2v_ft_stage1`，产出 `checkpoint-best.pth` 作为后续协同训练的初始化。  
  - **核心特性**：每个 batch 仅包含单车视角；`OPV2VDataset` 会把每个相机的姿态转换到各自 ego（LiDAR）系再送入模型。
- **Stage2 = 协同固定视角微调**  
  - **数据**：`configs/dataset/opv2v_coop_ft.yaml`（同一时间戳下聚合多个车辆，`num_views=10`，`coop_min_agents=2`，`coop_min_num_views=4`，`main_agent='641'`）。  
  - **命令**：`python -m mapanything.train.training machine=local3090 dataset=opv2v_coop_ft ... train_params.max_num_of_imgs_per_gpu=8 lr=3e-5 epochs=3 distributed.world_size=5`。  
  - **输出目录**：`experiments/opv2v_coop_stage2_full`。  
  - **核心特性**：`OPV2VCoopDataset` 会把其他车辆的视角统一变换到主车（默认 641）ego frame，并允许 batch 内视角数量在 `min_num_views~max_available` 之间动态变化（`allow_variable_view_count=True`）。

> ✅ 需要注意：Stage2 的训练样本“始终包含多个车辆”，因此直接拿它来做“单车输入”推理时，模型会因为缺失额外 agent 的点云而产生明显偏移（见 §4 的现象分析）。这也是后面必须改造评估脚本的根源。

## 3. 数据、环境与资产
- **真实 OPV2V 数据**：  
  - RGB/YAML/PCD：`/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V`  
  - 深度：`/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/opv2v_depth`
- **Conda 环境**：`mapanything_reconstructed`（位于 `~/miniconda3/envs/`），已包含 `torch`, `open3d`, `matplotlib`, `hydra-core` 等依赖；进入仓库后执行 `pip install -e .`。
- **训练产物**（位于 `/home/qqxluca/map-anything3/experiments`）：
  - `opv2v_ft_stage1/checkpoint-best.pth`
  - `opv2v_coop_stage2_full/checkpoint-best.pth`
- **本地推理配置 JSON**：`map-anything/scripts/local_infer_config.json`（由 Hydra 配置自动导出，供离线脚本读取）。

## 4. 评估流程（可复现命令）
1. 激活环境：`source ~/miniconda3/bin/activate mapanything_reconstructed`
2. 进入评估代码树：`cd /home/qqxluca/vggt_series_4_coop/map-anything`
3. 运行 `scripts/color_compare.py`（附 `--no_viz` 以适配 SSH；若需 GUI，请通过 `ssh -Y` 或 `ssh -X` 启用 X11 转发）：

### Stage1（单车微调权重）
```bash
/home/qqxluca/miniconda3/envs/mapanything_reconstructed/bin/python \
  scripts/color_compare.py \
  --image_dir /media/.../OPV2V/train/2021_08_16_22_26_54/641 \
  --config_path /media/.../OPV2V/train/2021_08_16_22_26_54/641/000069.yaml \
  --gt_pcd_path /media/.../OPV2V/train/2021_08_16_22_26_54/641/000069.pcd \
  --checkpoint_path /home/qqxluca/map-anything3/experiments/opv2v_ft_stage1/checkpoint-best.pth \
  --hydra_config_path configs/train.yaml \
  --config_json_path scripts/local_infer_config.json \
  --config_overrides machine=local3090 dataset=opv2v_ft model=mapanything \
                      model/task=images_only model.encoder.uses_torch_hub=false loss=overall_loss \
  --no_viz \
  --save_pred_path visualization_outputs/stage1_000069_pred.pcd
```

### Stage2（协同微调权重）
同上，仅将 `--checkpoint_path` 指向 `opv2v_coop_stage2_full/checkpoint-best.pth`，并把 `dataset=opv2v_coop_ft`。输出保存为 `visualization_outputs/stage2_000069_pred.pcd`。

生成的 `.pcd` + 真实点云随后通过 Matplotlib 绘制离线对比（见 §6）。

### Cooperative（多车）评估
`scripts/color_compare.py` 现已支持 `--coop_*` 参数，可一次性读取同一时间戳下的多个车辆。示例（Seq=2021_08_16_22_26_54，Frame=000069，641 为主车，联合 650）：

```bash
/home/qqxluca/miniconda3/envs/mapanything_reconstructed/bin/python \
  scripts/color_compare.py \
  --coop_root /media/.../OPV2V \
  --coop_split train \
  --coop_sequence 2021_08_16_22_26_54 \
  --coop_frame 000069 \
  --coop_agents 641 650 \
  --coop_main_agent 641 \
  --coop_camera_ids 0 1 2 3 \
  --gt_pcd_path /media/.../641/000069.pcd \
  --checkpoint_path /home/qqxluca/map-anything3/experiments/opv2v_coop_stage2_full/checkpoint-best.pth \
  --hydra_config_path configs/train.yaml \
  --config_json_path scripts/local_infer_config.json \
  --config_overrides machine=local3090 dataset=opv2v_coop_ft model=mapanything \
                      model/task=images_only model.encoder.uses_torch_hub=false loss=overall_loss \
  --no_viz \
  --save_pred_path visualization_outputs/stage2_coop_000069_pred.pcd
```

> 将 `--checkpoint_path` 替换为 Stage1 权重即可得到“单车模型 + 多车输入”的对照结果。

## 5. 指标定义与样本范围
- **样本范围**：当前验证针对 `2021_08_16_22_26_54/641/000069` 单帧（共 4 个相机视角）。之所以从这帧开始，是因为它历史上用于多个 demo，便于老师快速对比；后续可以通过 shell loop 一次性扫多帧，文档中给出的命令即可复用。
- **训练阶段指标**：直接读取 `experiments/*/log.txt` 内的 JSON 行，取 `OPV2VDataset_loss_avg` 或 `OPV2VCoopDataset_loss_avg` 最低的条目（使用 Python 脚本过滤）；这些是模型训练过程中 MapAnything 内部总损失的统计。
- **推理姿态误差**：由 `scripts/color_compare.py` 第 566~638 行的 `log_camera_pose_errors` 计算。该函数遍历 `predictions[idx]["camera_poses"]` 与 YAML 里解析出的 `pose_C2W_cv`，以 view0 为参考系，分别输出绝对/相对平移误差（米、百分比）与旋转误差（度、百分比）。
- **点云差异**：`color_compare.py` 中 `predictions_to_pointcloud` + `T_CV_TO_UE` 将所有预测点云堆叠到 ego frame，并通过 `mask` 过滤无效像素；再由 `--save_pred_path` 写出 `.pcd` 供离线对比。
- **当前结论的局限**：因为只评估了单帧，数值主要用于 sanity check，不代表全量分布；要得到统计意义，需要批量遍历若干 `sequence/agent/frame` 并求均值/方差。

## 6. 定量结果
| 实验 | 输入视角 | 训练日志最优 loss | 推理姿态误差（平均） | 观察 |
| --- | --- | --- | --- | --- |
| Stage1（单车 → 单车） | 641 的 4 相机 | `1_000 @ loss_avg ≈ 2.52` | `0.22 m / 2.26°` | baseline，表现稳定。 |
| Stage1（单车 → 多车） | 641+650，各 4 相机 | 同上 | `11.77 m / 6.05°`（650 视角 >22 m） | 单车模型无法处理协同视角。 |
| Stage2（协同 → 单车） | 641 的 4 相机 | `200 @ loss_avg ≈ 2.62` | `3.82 m / 3.31°` | 缺失协同视角导致漂移。 |
| Stage2（协同 → 多车） | 641+650，各 4 相机 | 同上 | `4.06 m / 3.89°`（主车 3.8 m，协同 5–6 m） | 与训练设定一致，显著优于“强行单车”。 |

> 🔍 **结论**：Stage2 的价值在多车评估中才能体现；Stage1 结果可作为“单车参考线”，但不应直接推广到协同任务。后续需在 GPU 上批量跑更多帧，以获得统计更稳的结论。

### 6.1 批量评估 + 点云缓存/过滤（固定 10 帧，支持多高度复用）
- **固定帧集**：为避免“重新采样 → 结果不可比”的问题，`scripts/batch_eval.py` 现支持 `--frames_json`。我们把 `eval_runs/test_sample/summary_test.json` 里的 10 帧（`2021_08_20_21_10_24/{000113,000223}`, `2021_08_22_07_52_02/000195`, `2021_08_23_12_58_19/000266`, `2021_08_23_15_19_19/{000156,000182,000332}`, `2021_08_24_20_09_18/000280`, `2021_08_24_20_49_54/{000200,000262}`）锁定为 test 样本。  
- **点云缓存**：批量推理时添加 `--pc_save_dir eval_runs/pc_cache/test_sample`（同时开启 `--pc_metrics --pc_filter_z_min=-1 --pc_filter_z_max=4.2 --pc_filter_radius=120`）。脚本会在 `eval_runs/pc_cache/test_sample/<model>/<mode>/sequence_frame.npy` 写出预测点云（gitignored），供后续重算指标而不需再跑 GPU。  
- **指标重算脚本**：新建 `scripts/recompute_pc_metrics.py`（输入缓存目录 + GT PCD，输出新的 Chamfer/BEV 指标）。示例：  
  ```bash
  /home/qqxluca/miniconda3/envs/mapanything_reconstructed/bin/python \
    scripts/recompute_pc_metrics.py \
    --split test \
    --frames_json eval_runs/test_sample/summary_test.json \
    --pc_cache_dir eval_runs/pc_cache/test_sample \
    --output_root eval_runs/test_sample \
    --pc_filter_z_min -1 --pc_filter_z_max 1.8 --pc_filter_radius 120
  ```
- **高度对比**：先在 `z∈[-1,4.2] m` 下完成一次 GPU 推理（缓存点云 + 计算指标），随后直接在缓存上重算 `z∈[-1,1.8] m`。关键检测侧指标如下：

| 模型-模式 | Chamferᶠ(≤4.2 m) | Chamferᶠ(≤1.8 m) | BEV IoUᶠ(≤4.2 m) | BEV IoUᶠ(≤1.8 m) |
| --- | --- | --- | --- | --- |
| Pretrain-单车 | 2.735 | **2.094** | 0.0241 | **0.0168** |
| Stage1-单车 | 2.869 | **2.054** | 0.0319 | **0.0235** |
| Stage2-单车 | 3.105 | **2.237** | 0.0283 | **0.0236** |
| Pretrain-协同 | 3.015 | **2.109** | 0.0346 | **0.0240** |
| Stage1-协同 | 3.124 | **2.075** | 0.0467 | **0.0346** |
| Stage2-协同 | 3.086 | **2.296** | 0.0512 | **0.0416** |

> 观察：  
> 1. `z_max=1.8 m` 明显压低了“预测→GT”的 Chamfer（最多下降 30%），验证“漂浮层主要发生在 2 m 以上”。  
> 2. 但 `GT→预测` 与 BEV IoU 反而略降，说明我们砍掉了部分 GT 仍存在的高位稀疏点，导致覆盖率下降。下一步若要做检测评估，应对 GT 同步裁剪或把 `z_max` 做成可调。  
> 3. Stage1 在低高度下的 Chamferᶠ 与 BEV IoUᶠ 均略优于预训练；Stage2 虽然靠裁剪去掉了漂浮层，但整体仍落后。也就是说，**“深度/位姿上预训练 > 微调”的结论在低高度样本里依旧成立**，只是在检测友好度上 Stage1/Stage2 与预训练差距缩小。

## 7. 可视化输出
- **单车对比**：![Stage1 vs Stage2 Pred vs GT](assets/opv2v_eval/stage1_stage2_comparison.png)
- **多车对比**：![Stage1 vs Stage2 Coop](assets/opv2v_eval/stage1_stage2_coop_comparison.png)
- **过滤后点云（交互式）**：`visualization_outputs/html/` 下新增 `pretrain|stage1|stage2_{single,coop}_{raw,filtered}.html`，可直接在浏览器中旋转查看“预测 vs GT”。
- **俯视图对比**：`visualization_outputs/png/` 内含 `*_raw_top.png` / `*_filtered_top.png`（例如 `stage1_single_filtered_top.png`）。可以看到 Stage1 单车在过滤后仍保留 66.7% 的点并贴合地面，而 Stage2 仅剩 19.7% 且零散。协同模式下，Stage1 过滤后的点云几乎全部留存（97.3%），表明其预测主要集中在多视角交集区域。

如需交互式查看，可在本地（带显示的机器）运行 `open3d` 或 `CloudCompare` 打开 `visualization_outputs/*.pcd`（单车 & 多车版本均已保存）。

## 8. 已解决与待解决问题
- ✅ **复现链路**：确认 Stage1/Stage2 checkpoint + 配置 JSON + 真数据都能在当前仓库跑通，避免了“样例数据”误导。
- ✅ **无头可视化**：通过 `--no_viz` + Matplotlib 输出 PNG，绕过了 SSH 下 Open3D 无法弹窗的问题。
- ✅ **协同评估**：`color_compare.py` 已支持 `--coop_*`，可以直接比较多车视角。
- ⚠️ **CPU 推理速度**：在无 GPU 的会话中推理需要 ~30s/帧；后续应在具备 GPU 的节点上运行，或启用 `--memory_efficient` 版本。
- ⚠️ **日志/路径混乱**：老版本里仍有针对 `aws` 的默认配置，本次虽然通过 overrides 解决，但建议继续清理机器配置以避免踩坑。
- ⚠️ **深度监督稀疏**：测试帧的 GT depth mask 平均仅覆盖 2.8% 像素（最少 1.8%），且 2021_08_16_22_26_54/641/000069 的 `depth_mask_*.png` 显示天空/高处完全无真值；Stage1/Stage2 在这些区域的误差会被当前指标忽略或放大，需要在对比时明确“无监督区域”。
- ⚙️ **推理后过滤**：新增 `scripts/filter_pointcloud.py`，可对推理点云施加高度/半径约束并计算 Chamfer & BEV IoU。示例（Stage2 协同 + `z∈[-2,4] m`，`r<120 m`）把 110 万点裁成 56 万点，Chamfer(预测→GT) 从 2.34 m 降到 1.89 m，适合作为车辆检测前的“去漂浮层”预处理。

## 9. 下一步计划
1. **GPU 端批量评估**：沿用 `--coop_*` 命令，在 GPU 节点上批量跑多个 `sequence/agent`，输出均值/方差与完整可视化。
2. **X11 / 图形化评估**：在导师允许的情况下，在客户端开启 `ssh -Y` + `XQuartz/VcXsrv`，保留一次真实 Open3D 渲染截图，便于展示交互界面。
3. **Stage3（可变视角）预研**：等上述链路稳定后，再考虑重启 Stage3；重点关注梯度 checkpointing + num_views 动态带来的显存压力，必要时降分辨率或分批训练。
4. **数据校验脚本**：编写 `python -m mapanything.tools.check_opv2v --root ...`，提前检查缺失的图像/深度/YAML，减少训练时中断。

若老师需要查看更多帧或特定序列的评估，请告知具体 ID，我可以在现有流程上快速扩展。当前所有结果（命令、日志、图像）均已保存在仓库：`docs/opv2v_eval_report.md`、`docs/assets/opv2v_eval/*.png`、`visualization_outputs/*.pcd`。
