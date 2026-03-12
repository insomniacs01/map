# MapAnything OPV2V Fine-Tune Summary

> 目标：把本机 MapAnything 仓库里的训练/微调改动完整迁移到新的代码树，使 Codex 能在全新 clone 的 MapAnything 中“一次性”复现 OPV2V 单车 + 协同训练流程。

## 最高优先组件与依赖

1. **训练入口**：`mapanything/train/training.py`（Hydra 驱动）。所有实验都是用 `python -m mapanything.train.training ...` 触发，并依赖 Hydra `configs/**/*`.
2. **机器/路径配置**：`configs/machine/local3090.yaml` 定义了 OPV2V 图片 & 深度路径、预训练 checkpoint、实验输出目录等。若换机器，首先复制/更新这一份。
3. **数据集相关代码**
   - `mapanything/datasets/base/base_dataset.py:143-494` 新增 `allow_variable_view_count`、`min_num_views_allowed`，用于支持协同版 dataset 动态视角数量。任何搬运都要同时带上 `BaseDataset` 的这些字段与断言，否则 `OPV2VCoopDataset` 会直接 assert。
   - `mapanything/datasets/opv2v.py` 中的 `OPV2VDataset`（单车）与 `OPV2VCoopDataset`（协同）共存。协同版本依赖 `_convert_pose_to_opencv`、`cords_to_pose`、`load_frame_metadata` 等同文件内函数；不可只拷贝类定义而忘了辅助函数。
4. **Hydra 数据集配置**：`configs/dataset/opv2v_ft.yaml`（单车 Stage1）和 `configs/dataset/opv2v_coop_ft.yaml`（协同 Stage2/3）。协同配置将 `coop_min_num_views/coop_max_num_views` 等参数注入 dataset 构造器，必须保持字段名一致。
5. **训练工具链**：`mapanything/utils/train_tools.py:347-410` 的 `save_on_master` 原子写入 & `init_distributed_mode` timeout 扩展，保证多机/多卡 save & init 稳定。迁移时必须带上，否则分布式阶段写 checkpoint 会互踩。
6. **已有实验目录（参考用，不必提交）**：
   - `experiments/opv2v_ft_stage1`：单车微调所有日志与 checkpoint。
   - `experiments/opv2v_coop_stage2_full`：协同 8 视角版本。
   - `experiments/opv2v_coop_stage3_varviews`：协同可变视角尝试（当前只留下 `checkpoint-last.*`）。
   - 这些目录主要用于确认命令、超参、指标，可在新仓库中重新生成。

## 训练阶段拆解

### Stage1：OPV2V 单车微调（`experiments/opv2v_ft_stage1`）

- **用途**：产出协同行程的初始化 checkpoint（`checkpoint-best.pth`）。
- **配置**：
  - Hydra overrides（从 `console.log` errors 中可见）：
    ```bash
    python -m mapanything.train.training \
      machine=local3090 \
      dataset=opv2v_ft \
      model=mapanything \
      loss=overall_loss \
      model.pretrained=/home/qqxluca/map-anything3/checkpoints/facebook_map-anything-local.pth \
      train_params.max_num_of_imgs_per_gpu=4 \
      train_params.lr=5e-5 \
      train_params.epochs=5 \
      output_dir=/home/qqxluca/map-anything3/experiments/opv2v_ft_stage1 \
      distributed.world_size=4
    ```
  - `configs/dataset/opv2v_ft.yaml`：4 视角、`resolution_train=518×392`。
  - `train_params` 使用默认 set：`warmup_epochs=10`、`bf16 amp`、`linear warmup + half-cycle cosine`。
- **依赖**：`OPV2VDataset`（同文件 1P 版本）以及数据根路径（`local3090` 机型配置）。

### Stage2：协同固定 8 视角（`experiments/opv2v_coop_stage2_full`）

- **用途**：在多个车辆联合视角上延续微调。
- **命令示例**（Hydra 控制台打印的参数）：
  ```bash
  python -m mapanything.train.training \
    machine=local3090 \
    dataset=opv2v_coop_ft \
    model=mapanything \
    loss=overall_loss \
    model.pretrained=/home/qqxluca/map-anything3/experiments/opv2v_ft_stage1/checkpoint-best.pth \
    train_params.max_num_of_imgs_per_gpu=8 \
    train_params.lr=3e-5 \
    train_params.epochs=3 \
    output_dir=/home/qqxluca/map-anything3/experiments/opv2v_coop_stage2_full \
    distributed.world_size=5
  ```
- **关键数据参数**（`configs/dataset/opv2v_coop_ft.yaml`）：
  - `num_views: 10`（会被 `OPV2VCoopDataset` 限制为可用视角上限）。
  - `coop_min_agents=1`（但 Stage2 full 训练在日志里仍设 2）。
  - `coop_min_num_views=4`、`coop_max_num_views=null`（DataLoader 内由可用视角截断）。
  - `train_dataset`/`test_dataset` 都引用 `OPV2VCoopDataset`，把 Hydra 变量传给 `min_num_views`/`max_num_views`。
- **数据逻辑提醒**：
  - `OPV2VCoopDataset` 会以 `main_agent='641'` 为基准，把其他车辆坐标系转换到主车世界系。若要换主车，必须一起改 YAML 与任何可视化脚本。
  - `BaseDataset` 的 `allow_variable_view_count=True` 是在 `OPV2VCoopDataset.__init__` 里设置的（`mapanything/datasets/opv2v.py:209-232`）。如果忘拷贝该字段，dataset 会 fallback 到旧的固定视角断言导致崩溃。
- **训练工具要求**：`save_on_master` 的临时文件写入确保多进程保存 `checkpoint-best/final/last` 时不冲突；`TORCH_DIST_TIMEOUT` 可通过环境变量重写（默认 7200s），对多机 NCCL 初始化至关重要。

### Stage3：协同可变视角（`experiments/opv2v_coop_stage3_varviews`）

- **状态**：进行中，已有多次重启记录（`console.log` 中多次 “Building train dataset ...”）。虽然还未出稳定 ckpt，但复现需要以下设置：
  - `min_agents=1`，`min_num_views=4`，`max_num_views=None`，`num_views` 初始 12 后降至 10。
  - 模型端开启 gradient checkpointing（`console.log` 可见 `encoder.gradient_checkpointing=True` 等），否则 12 视角 batch 无法塞进显存。
  - `train_params.max_num_of_imgs_per_gpu=12`、`epochs=6`。
- **当前问题**：`experiments/opv2v_coop_stage3_varviews/checkpoint-last.corrupt` 表明 `save_on_master` 在训练被杀时仍可能留下 `.tmp` 文件。迁移时可在外层脚本中加 `rsync --partial` 或在新仓库里加入额外安全检查。

## 依赖关系速查

| 组件 | 位置 | 依赖 |
| --- | --- | --- |
| 数据源路径配置 | `configs/machine/local3090.yaml` | 训练脚本 (`mapanything/train/training.py`)、`OPV2V*Dataset` |
| 单车数据集定义 | `mapanything/datasets/opv2v.py` (`OPV2VDataset`) | `mapanything/datasets/base/base_dataset.py`, `numpy`, `PIL` |
| 协同数据集定义 | 同上 (`OPV2VCoopDataset`) | `BaseDataset` 新字段、`load_frame_metadata`、`cords_to_pose`、`_convert_pose_to_opencv` |
| 数据集配置 | `configs/dataset/opv2v_ft.yaml`, `configs/dataset/opv2v_coop_ft.yaml` | Hydra `dataset` group |
| 模型/训练配置 | `configs/model/*.yaml`, `configs/train_params/*.yaml`, `configs/loss/overall_loss.yaml` | `mapanything/train/training.py` |
| 训练工具 | `mapanything/utils/train_tools.py` | 所有分布式训练流程 |
| 方案文档 | `docs/opv2v_vehicle_id_adaptation.md` | 描述后续 Vehicle-ID embedding 变化（现阶段模型侧尚未改） |

## 迁移步骤建议

1. **同步 Hydra 配置**：把 `configs/machine/local3090.yaml`（或对应服务器版本）、`configs/dataset/opv2v_ft.yaml`、`configs/dataset/opv2v_coop_ft.yaml`、涉及的 `configs/model`/`configs/loss`/`configs/train_params` 子项整体拷贝，保证 Hydra 可以解析所有引用。
2. **同步数据集代码**：复制 `mapanything/datasets/base/base_dataset.py` 与 `mapanything/datasets/opv2v.py`（注意保留 import/辅助函数）。若新仓库已有不同版本，需要合并 `allow_variable_view_count` 以及 `OPV2VCoopDataset` 的初始化逻辑。
3. **同步训练工具**：复制 `mapanything/utils/train_tools.py` 中的 `save_on_master` 和 `init_distributed_mode` 改动；如有额外工具文件依赖本地路径，也一起迁移。
4. **确认预训练权重**：Stage1 依赖 `/home/qqxluca/map-anything3/checkpoints/facebook_map-anything-local.pth`；如目标机器路径不同，应在新的 machine 配置里更新 `root_pretrained_checkpoints_dir` 并把权重放入。
5. **复现 Stage1 → Stage2 → Stage3**：
   - 先在新仓库跑 Stage1，得到 `checkpoint-best.pth`。
   - 将 `model.pretrained` 指向 Stage1 结果，再跑 Stage2/Stage3。保持 `output_dir` 层级与原 repo 相同有利于脚本复用。
6. **日志 & 监控**：`experiments/*/log.txt` 中是 JSONL（每 epoch），`console.log` 记录 Hydra 配置打印。确保 `experiments` 目录写权限正常。

## 常见坑位与提示

- **视角断言**：忘记设置 `allow_variable_view_count=True` 时，`BaseDataset` 会强制 `len(views)==num_views` 引发 AssertionError。复制代码时务必保留此逻辑。
- **主车世界系**：`OPV2VCoopDataset` 把所有视角转换到主车（默认 `641`）。若要做多主车或随机主车，连同 `_choose_main_agent`、`label/instance` 生成逻辑都需要一起调整。
- **分布式超时**：新的集群若网络更慢，记得设置 `TORCH_DIST_TIMEOUT`（秒）。否则 `init_process_group` 可能 30min 后才报错。
- **保存中断**：若训练过程被 Ctrl+C 或 OOM 杀死，`*.pth.tmp` 可能残留。重新启动前要清理或写脚本自动 rename。

有了这份总结，Codex 只需按照“同步代码 → 更新 Hydra 配置 → Stage1 → Stage2/3 命令”即可在新的 MapAnything 仓库里复现训练微调部分，而无需再次翻查多处文件。
