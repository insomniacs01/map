# OPV2V 统一流程记录

> 目标：在 **同一份 map-anything 代码树里打通 OPV2V 的微调（单车 + 协同）与推理/可视化**，避免“样例数据”“临时脚本”与真实流程混在一起导致混乱。

## 0. TL;DR
- **真实数据只放在 `/media/tsinghua3090/.../OPV2V` 与 `/media/.../opv2v_depth`**，仓库里 `tests/fixtures/sample_opv2v/`、`example/` 仅作为单元测试/入门示例；训练与可视化一律读取软链接或 machine 配置指向的真数据。
- Conda 环境固定在 `mapanything_reconstructed`，里面已经安装 `torch==2.1`, `open3d`, `hydra-core`, `rerun`, `matplotlib`, `scipy` 等依赖；若新增依赖请继续装在该 env 里。
- **Stage1 → Stage2 → Stage3** 的训练命令与输出目录要保持和 `docs/opv2v_coop_training_summary.md` 一致；协同阶段依赖 `mapanything/datasets/opv2v.py` 中的 `OPV2VCoopDataset`（带 `allow_variable_view_count`）。
- 可视化分成两类：① 纯离线产物（PNG/PCD）脚本，适合 SSH 无显示环境；② 需要 Open3D GUI 的脚本，配合 X11 转发或导出点云到本地查看。
- 当前已跑通的 demo：  
  `OPV2VDataset` loader (`1 scenes`)、`visualize_opv2v_pose.py`（输出 `example/opv2v_pose_vis.png`）。缺口：Open3D 仍需显示环境或改成 Offscreen。

---

## 1. 数据与路径规范

### 1.1 真实数据源
- **RGB + YAML + PCD**：`/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V`
- **深度 (.npy/.png)**：`/media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/opv2v_depth`
- 目录层次：`{split}/{sequence}/{agent}/{frame}_{cameraX}.png|.yaml` 与对应深度文件，满足 `mapanything/datasets/opv2v.py` 的读取假设。

### 1.2 仓库内的软链接推荐
在仓库根目录执行：

```bash
ln -s /media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V \
      map-anything/data/opv2v_images
ln -s /media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/opv2v_depth \
      map-anything/data/opv2v_depth
```

然后在 `configs/machine/local3090.yaml` 中将

```yaml
opv2v_images_root: /media/tsinghua3090/.../OPV2V
opv2v_depth_root:  /media/tsinghua3090/.../opv2v_depth
```

统一改成 `map-anything/data/opv2v_images` 与 `map-anything/data/opv2v_depth`，这样无论主机硬盘挂载路径如何变化，**所有 Hydra 配置都只依赖仓库相对路径**。

> `tests/fixtures/sample_opv2v/` 与 `example/` 仅保留用于单元测试与 CI。后续如需继续存放 fake 资产，请继续放在 `tests/fixtures/` 并加 README，避免误用。

### 1.3 数据集自检命令

使用 Conda 环境的 Python 直接实例化 dataset，确认能遍历真实文件：

```bash
/home/qqxluca/miniconda3/envs/mapanything_reconstructed/bin/python \
  map-anything/mapanything/datasets/opv2v.py \
  --root /media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/OPV2V \
  --depth_root /media/tsinghua3090/8626b953-db6f-4e02-b531-fceb130612da/home/opv2v_depth \
  --max_scenes 1
```

输出 `1 scenes` 表示 loader 能正确读取到 train split/agent 的 YAML、图像与深度。这一步必须在正式训练/可视化前完成。

---

## 2. Conda 环境

- 环境位置：`~/miniconda3/envs/mapanything_reconstructed`
- 激活方式：`source ~/miniconda3/bin/activate mapanything_reconstructed`
- 关键依赖（已安装）：`torch`, `torchvision`, `open3d`, `hydra-core`, `omegaconf`, `rerun`, `matplotlib`, `scipy`, `Pillow`, `tqdm`, `pyyaml`
- 若需新增包，使用 `pip install <pkg>`（已激活 env）或 `conda install -n mapanything_reconstructed <pkg>`；无 sudo 权限时也不要装到系统 Python。
- 进入仓库后执行 `pip install -e .`（一次性）以便脚本能直接 import `mapanything.*`。

---

## 3. 微调流程（Hydra）

> 训练入口统一使用 `python -m mapanything.train.training ...`，命令在 `docs/opv2v_coop_training_summary.md` 已给出。这里再压一遍要点，防止漏配。

### Stage1：单车 OPV2V 微调

```bash
python -m mapanything.train.training \
  machine=local3090 \
  dataset=opv2v_ft \
  model=mapanything \
  loss=overall_loss \
  model.pretrained=${machine.root_pretrained_checkpoints_dir}/facebook_map-anything-local.pth \
  train_params.max_num_of_imgs_per_gpu=4 \
  train_params.lr=5e-5 \
  train_params.epochs=5 \
  output_dir=${machine.root_experiments_dir}/opv2v_ft_stage1 \
  distributed.world_size=4
```

产物：`experiments/opv2v_ft_stage1/checkpoint-best.pth`，供协同阶段初始化使用。

### Stage2：协同固定视角

```bash
python -m mapanything.train.training \
  machine=local3090 \
  dataset=opv2v_coop_ft \
  model=mapanything \
  loss=overall_loss \
  model.pretrained=${machine.root_experiments_dir}/opv2v_ft_stage1/checkpoint-best.pth \
  train_params.max_num_of_imgs_per_gpu=8 \
  train_params.lr=3e-5 \
  train_params.epochs=3 \
  output_dir=${machine.root_experiments_dir}/opv2v_coop_stage2_full \
  distributed.world_size=5
```

- `configs/dataset/opv2v_coop_ft.yaml` 注入了 `coop_min_num_views`, `coop_max_num_views` 等关键参数；不可重命名。
- `OPV2VCoopDataset` 内部会保证所有视角转换到 `main_agent='641'` 的 ego frame；若需换主车同时修改 YAML + doc。

### Stage3：协同可变视角（实验中）

```bash
python -m mapanything.train.training \
  machine=local3090 \
  dataset=opv2v_coop_ft \
  dataset.coop_min_agents=1 \
  dataset.coop_min_num_views=4 \
  dataset.coop_max_num_views=null \
  dataset.num_views=10 \
  model=mapanything \
  loss=overall_loss \
  model.pretrained=${machine.root_experiments_dir}/opv2v_coop_stage2_full/checkpoint-best.pth \
  model.encoder.gradient_checkpointing=true \
  train_params.max_num_of_imgs_per_gpu=12 \
  train_params.lr=2e-5 \
  train_params.epochs=6 \
  output_dir=${machine.root_experiments_dir}/opv2v_coop_stage3_varviews \
  distributed.world_size=5
```

- 注意 gradient checkpointing，否则 12 视角 batch 会爆显存。
- `save_on_master` 仍可能在异常退出时遗留 `.corrupt` 文件；重启前先删干净。

### 训练目录规范
- `root_experiments_dir`（比如 `/home/qqxluca/map-anything3/experiments`）下按阶段划分：
  - `opv2v_ft_stage1`
  - `opv2v_coop_stage2_full`
  - `opv2v_coop_stage3_varviews`
- 同步保存 `console.log`, `log.txt`, `checkpoint-last/best/final.pth`。**禁止把这些输出迁入 repo**，仅作参考。

---

## 4. 可视化与 SSH 环境

### 4.1 离线/无窗口脚本
| 脚本 | 作用 | 备注 |
| --- | --- | --- |
| `scripts/visualize_opv2v_pose.py` | Matplotlib 输出相机姿态 + 车辆框体 PNG | 头一次运行需 `MPLBACKEND=Agg`，示例输出 `example/opv2v_pose_vis.png`（来自 `2021_08_16_22_26_54/641/000069`）。 |
| `scripts/depth_ge.py` | 读取 YAML + RGB，跑 MapAnything 推理，导出 `depth.npy/mask.npy`、可选 `.pcd` | 适用于批量导出再在本地可视化；需保证 `open3d` 可 import 但不需要窗口。 |
| `depth/depth.py` | 把 `.pcd` 投影成多相机深度 (PNG + NPY) | 可以用于生成参考深度或调试 pose。 |
| `scripts/tree.py` | 快速列出数据树，确认软链接是否指向真实数据 | |

运行示例（已验证）：

```bash
MPLBACKEND=Agg /home/qqxluca/miniconda3/envs/mapanything_reconstructed/bin/python \
  map-anything/scripts/visualize_opv2v_pose.py \
  --dataset_root /media/.../OPV2V \
  --split train \
  --sequence 2021_08_16_22_26_54 \
  --agent 641 \
  --frame 000069 \
  --output map-anything/example/opv2v_pose_vis.png
```

### 4.2 需要 Open3D GUI 的脚本
（`scripts/pcd_with_boxes.py`, `scripts/visualize.py`, `scripts/compare_pointclouds.py`, `scripts/devided_pointclouds.py` 等）

两种可行方案：
1. **SSH + X11 转发**  
   - 登录命令：`ssh -Y -C <user>@<host>` 或 `ssh -X`.  
   - 确保本地 XServer（macOS: XQuartz, Linux: 自带, Windows: Xming/VcXsrv）开启。  
   - 服务器需安装 `xauth`（无 sudo 则联系管理员预装）。若延迟大，可结合 `MESA_GL_VERSION_OVERRIDE=3.3` 强制软件渲染。
2. **导出点云/框体到文件 → 本地查看**  
   - 使用 `scripts/depth_ge.py --save_pred_dir outputs/pcd` 将预测写成 `.pcd`/`.ply`。  
   - 使用 `pcd_with_boxes.py --pred_pcd pred.pcd --gt_pcd ... --no_gui`（需改造，见 TODO）或直接在本地跑 `open3d`/`CloudCompare`。

> TODO：抽时间把 `pcd_with_boxes.py`、`compare_pointclouds.py` 改成 `VisualizerWithKeyCallback` + `--offscreen` 选项，便于自动化导出 `*.png`。

### 4.3 产物同步建议
- 所有 PNG/PCD 输出集中放到 `map-anything/visualization_outputs/<date>/...`
- 通过 `rsync -avh --progress <host>:map-anything/visualization_outputs ./vis_outputs` 拷回本地。
- 如果需要在线预览，可在本地 `open3d`、`potree` 或 `meshlab` 中打开 `.pcd`。

---

## 5. 清理与 TODO

1. **样例数据隔离**  
   - 把 `tests/fixtures/sample_opv2v/` 中的资产仅作为单测使用，README 已说明“仅限单测，绝不参与训练”。  
   - `configs/dataset/opv2v_ft_smoke.yaml` 保留，但文件头部注明“DEBUG ONLY”。
2. **路径与依赖统一**  
   - 所有脚本引用真实数据路径时都应该复用 `configs/machine/*.yaml` 的变量或 `map-anything/data/opv2v_*` 软链接，禁止写死 `/home/qqxluca/...`。  
   - `mapanything/utils/opv2v_viz.py`, `mapanything/utils/opv2v_pointclouds.py` 已经是公共模块，后续任何脚本若需要相同逻辑请 import，不要复制粘贴。
3. **Open3D 无头模式**  
   - 评估 `o3d.visualization.OffscreenRenderer`（>=0.17）或 `o3d.visualization.rendering.OffscreenRenderer`，写一个通用 `save_view_png(geoms, out_path)`，供 `pcd_with_boxes.py` 等复用。  
   - 若 Offscreen 方案难以兼容旧版本，就默认导出 `.pcd`/`.ply`，由本地查看。
4. **日志与配置回填**  
   - `docs/opv2v_vehicle_id_adaptation.md`、`docs/opv2v_coop_training_summary.md` 已覆盖模型/Vehicle-ID 方案；本文件负责串起“数据 → 训练 → 可视化”主线，后续若流程有改动优先更新这里。  

---

## 6. 已验证 Demo

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| 数据集加载 | `/home/qqxluca/miniconda3/envs/mapanything_reconstructed/bin/python map-anything/mapanything/datasets/opv2v.py --root ... --depth_root ... --max_scenes 1` | 输出 `1 scenes`，确认真实数据路径可读。 |
| 姿态可视化 | `MPLBACKEND=Agg ... scripts/visualize_opv2v_pose.py --sequence 2021_08_16_22_26_54 --agent 641 --frame 000069` | 生成 `map-anything/example/opv2v_pose_vis.png`（已保存，可用于 sanity check）。 |

后续若跑通 `scripts/depth_ge.py`（导出 `.pcd`）或 `pcd_with_boxes.py`（Open3D GUI），请把命令 + 输出补记在此表。

---

## 7. 待确认问题

1. **协同阶段的显存余量**：Stage3 仍未收敛，需要确认 `num_views`、分辨率和 batch size 之间的权衡，必要时考虑两阶段训练（先固定每车 2 视角，再逐步扩展）。
2. **Open3D 的 headless 替代方案**：若 X11 转发不稳定，是否可以改用 `rerun` 或 `pyvista` 做 Web 端可视化？需要评估投入。
3. **数据校验脚本**：是否需要一个 `python -m mapanything.tools.check_opv2v --root ...` 的小工具，把缺失的 YAML/深度/图像一次性列出来，避免训练时才发现？如有需要请提出。

确认以上事项后即可宣布“OPV2V 微调 + 可视化联合流程”正式收敛，并把该文档当作唯一入口给其他同事。
