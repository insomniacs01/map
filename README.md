# MapAnything FT（多车圆柱融合分支）

> 更新时间：2025-12-07 03:35 UTC

本仓库在 Meta 官方 MapAnything 基础上专注于 **OPV2V 多车圆柱相机** 场景。我们沉淀了数据聚合、可视化、训练脚本以及调参计划，方便团队在多台 3090 上连续迭代。官方 README 已归档至 `docs/archive/upstream_readme.md`，如需查阅原项目概览请移步该文件。

## 当前状态速览

- **数据**：`mapanything/datasets/opv2v_cyl.py` 负责把单车 4 个环视针孔相机拼成 360° 圆柱 panorama，并在 view 字典中设置 `camera_model="cyl"`、`virtual_camera`、`non_ambiguous_mask` 等字段。`scripts/opv2v_panorama_preview.py` 与 `scripts/visualize_opv2v_cyl_html.py` 提供样本抽查。
- **可视化**：`scripts/render_pointcloud_html.py` 已增量支持 per-view PNG、JSON 摘要、相机视角控制，可在 `eval_runs/.../val_idx*_pointcloud.html` 中直接查看 Plotly 点云/深度/Mask。
- **训练**：新增 `configs/loss/opv2v_pose_loss.yaml`，在 `FactoredGeometryScaleRegr3DPlusNormalGMLoss` 中开启 `compute_pairwise_relative_pose_loss`，将 `pose_quats_loss_weight/pose_trans_loss_weight` 提升至 3，并保留 `NonAmbiguousMaskLoss`。当前触发 10 卡训练会在加载 checkpoint 后因 GPU 被其他任务占满而 OOM，详情见 `docs/cyl/opv2v_cyl_coop_debug_plan.md`。
- **调试日志**：`docs/cyl/multi_car_training_plan.md` 描述 4n→n 圆柱方案；`docs/cyl/opv2v_cyl_coop_debug_plan.md` 记录逐项排查、脚本命令、失败原因；`docs/README.md` 汇总文档入口。
- **重要缓存**：OPV2V 多车 YAML 预筛选缓存已启用（避免每次启动都扫大量 YAML）。持久缓存默认在 `map-anything/cache/opv2v_scene_cache`，本地加速缓存在 `/tmp/mapanything_cache`（由 `MAPANYTHING_CACHE_DIR` / `MAPANYTHING_FAST_CACHE_DIR` 控制，自动双写/同步）。详细说明与预生成 prompt 见 `docs/opv2v_scene_cache.md`。

## 目录索引

| 主题 | 文档/脚本 | 说明 |
| --- | --- | --- |
| 圆柱训练方案 & 设计 | `docs/cyl/multi_car_training_plan.md` | 详细说明 4n→n 聚合、虚拟相机、步骤（最新更新带时间戳） |
| 调试 Checklist & 训练日志 | `docs/cyl/opv2v_cyl_coop_debug_plan.md` | 逐项排查、命令模板、异常记录 |
| 文档导航 | `docs/README.md` | Maintained/Archived 文档分类 |
| 可视化 | `scripts/visualize_opv2v_cyl_html.py`、`scripts/render_pointcloud_html.py` | 圆柱数据 & 点云 HTML 展示 |
| 训练入口 | `bash_scripts/train/finetuning/opv2v_cyl_coop.sh` | Multi-GPU torchrun 命令模板 |
| 环境文件 | `environment.mapanything_ft.yml` | Conda 依赖；需要 OPV2V 数据软链见下 |

## 快速开始

1. **准备环境**
   ```bash
   cd /path/to/vggt_series_4_coop/map-anything
   export ENV_ROOT=${ENV_ROOT:-$PWD/../.conda}
   export CONDA_ENVS_PATH=$ENV_ROOT/envs
   export CONDA_PKGS_DIRS=$ENV_ROOT/pkgs
   export PIP_CACHE_DIR=$ENV_ROOT/pip-cache
   export TMPDIR=$ENV_ROOT/tmp
   conda env create -n mapanything_ft -f environment.mapanything_ft.yml
   conda activate mapanything_ft
   export PYTHONPATH=$(pwd)
   ```
2. **挂载数据**
   - OPV2V 彩图：`map-anything/data/opv2v_images`（软链接到 OPV2V 原始目录）
   - 深度图：`map-anything/data/opv2v_depth`（软链接到生成的深度目录）
   - 若路径变动，请同步更新 `configs/machine/local3090.yaml`。
3. **验证圆柱样本**
   ```bash
   PYTHONPATH=$(pwd) conda run -n mapanything_ft python scripts/validate_opv2v_cyl_setup.py --num-samples 8 --splits train,val --verbose
   PYTHONPATH=$(pwd) conda run -n mapanything_ft python scripts/opv2v_panorama_preview.py --split validate --index 0 --output_dir eval_runs/opv2v_cyl_precheck/val_idx0
   ```
4. **训练/恢复（示例）**
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   conda run -n mapanything_ft python -m torch.distributed.run --standalone --nproc_per_node=10 \
       scripts/train.py \
       machine=local3090 \
       model=mapanything \
       dataset=opv2v_cyl_coop_ft_5gpu \
       train_params=opv2v_cyl_coop \
       loss=opv2v_pose_loss \
       model.model_config.pretrained_checkpoint_path=/home/qqxluca/map-anything3/experiments/mapanything/training/opv2v_cyl_coop/20251201_021942/checkpoint-best.pth \
       model.encoder.gradient_checkpointing=true \
       model.info_sharing.module_args.gradient_checkpointing=true \
       model.pred_head.gradient_checkpointing=true \
       hydra.run.dir=/home/qqxluca/map-anything3/experiments/mapanything/training/opv2v_cyl_posefix/\${now:%Y%m%d_%H%M%S} \
       hydra.job.name=opv2v_cyl_posefix
   ```
   > 若 GPU 正被 `/home/gongyan/anaconda3/envs/UniMODE/bin/python` 占满，会在加载 checkpoint 或第一轮 backward 报 OOM，需等待资源释放。详见 `docs/cyl/opv2v_cyl_coop_debug_plan.md`。

## VGGT（OPV2V）尺度微调

- **训练入口（带 metric scale head）**：`bash_scripts/train/finetuning/opv2v_coop_vggt_metric.sh`（默认 8 views/样本）
- **训练入口（不加尺度头，直接监督 metric pose/depth）**：`bash_scripts/train/finetuning/opv2v_coop_vggt_pose_metric.sh`（`norm_mode='?avg_dis'`，关闭 metric 样本归一化）
- **可视化（GT LiDAR/框 vs VGGT 点云）**：`scripts/viz_opv2v_vggt_pred_vs_gt_boxes_html.py`
- **可视化（把 GT 3D box 投影到每个相机视图）**：`scripts/viz_opv2v_boxes_on_images.py`

## 可视化与诊断

- **圆柱原始样本（RGB/深度/mask）**
  ```bash
  PYTHONPATH=$(pwd) conda run -n mapanything_ft python scripts/visualize_opv2v_cyl_html.py \
      --split validate --index 0 \
      --output-html eval_runs/opv2v_cyl_precheck/val_idx0/panorama.html
  ```
- **模型预测点云**
  ```bash
  PYTHONPATH=$(pwd) conda run -n mapanything_ft python scripts/render_pointcloud_html.py \
      --config scripts/local_infer_config.json \
      --checkpoint <your_checkpoint.pth> \
      --output-dir eval_runs/opv2v_cyl_precheck \
      --view-output-dir eval_runs/opv2v_cyl_precheck/val_idx0_views \
      --debug-summary-json eval_runs/opv2v_cyl_precheck/val_idx0_summary.json
  ```
  该脚本会输出：
  - `val_idx0_pointcloud.html`：Plotly 点云，支持轴对齐、体素抽样；
  - `*_view*.png`：渲染时的彩色点云截图；
  - `val_idx0_summary.json`：记录 pose/scale/深度统计，可追踪训练是否逼近目标。

## 进展 & 待办（2025-12-07）

| 状态 | 项目 | 说明 |
| --- | --- | --- |
| ✅ | 圆柱视角数据链闭环 | 数据脚本、mask 统计、HTML 可视化、预检查均已跑通。 |
| ✅ | 可视化脚本增强 | `render_pointcloud_html.py` 支持 per-view PNG、JSON 摘要及交互视角；用于定位 pose 漂移。 |
| ✅ | Pose Loss 配置 | `configs/loss/opv2v_pose_loss.yaml` 启用 pairwise pose loss，训练入口默认使用该配置。 |
| ⚠️ | 10 卡训练 | 目前因外部 `UniMODE` 进程占满 10×3090 导致预训练权重加载或 backward 时 CUBLAS OOM。需等待 GPU 空闲或与对方协调。 |
| 🚧 | 验证 loss < 5 | 训练尚未重新跑完；待 GPU 空出后按照 `docs/cyl/opv2v_cyl_coop_debug_plan.md` 的 checklist 继续。 |
| 🚧 | HTML 点云改进 | 需根据最新可视化结果继续调姿态/遮挡 mask，并在 `docs/cyl/multi_car_training_plan.md` 中记录。 |

## 常见问题

- **为什么训练步数会从 5000 变 4000？**  
  `ResizedDataset` 长度除以 `world_size` 得到 epoch step。GPU 数变化需要切换 `configs/dataset/opv2v_cyl_coop_ft.yaml` ↔ `_5gpu.yaml`（后者长度 50k）。细节参见 `docs/cyl/multi_car_training_plan.md §2.4`。

- **NonAmbiguousMaskLoss 总是 0？**  
  已在 `mapanything/train/losses.py` 加入 `pred_mask_mean` / `gt_mask_mean` 日志指标，若均值 < 0.05 需回查圆柱拼接的 valid mask 或调整 BCE `pos_weight`，详见调试计划文档。

- **HTML 内容更新后如何查看？**  
  直接刷新浏览器即可，因为新的可视化会覆盖 `eval_runs/.../*.html`。必要时用 `python -m http.server` 暴露目录。

- **HTML 里的点云看起来“被压扁/竖直拉伸/错位”？**  
  OPV2V 的 GT LiDAR `.pcd` 使用 CARLA/UE 坐标（Z-up），而 MapAnything 推理输出点云在 OpenCV 坐标（Y-down）。生成报告时需要把预测点云从 OpenCV 转回 CARLA；`scripts/make_batch_eval_pcd_html.py` 已自动处理，旧 HTML 建议重新生成。

## Git 工作流

1. 所有本地修改进入仓库根目录执行：
   ```bash
   git status
   git add <files>
   git commit -m "feat: xxx"
   git push origin main
   ```
2. 远程仓库：`git@github.com:MassimoQu/map-anything.git`（SSH key 已配置）。推送前请确认 `git status` 干净且 Hydra 生成的 log/ckpt 未误加入（`.gitignore` 已覆盖常见路径，如需新增请修改 `.gitignore`）。

如需更多细节，请阅读 `docs/cyl/*.md`。若文档与代码不符，请在修改时同步更新时间戳与表格状态。
