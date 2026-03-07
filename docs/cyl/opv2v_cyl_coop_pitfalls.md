# OPV2V 圆柱协同训练踩坑大全（新手友好）

本文把目前为止在 OPV2V 圆柱协同（MapAnything + 检测头/几何微调）过程中踩过的坑按“本质”拆开讲清楚，并给出可复现的检查方法与对应代码/脚本入口。

> 适用范围：`OPV2VCoopCylindricalDataset`（圆柱全景）+ 多车协同 + 以 `view0` 为参考的相对位姿评估/监督。

---

## 0. 先建立一个“最小心智模型”

MapAnything 这条链路里，最容易踩坑的点是：**同一个名字在不同模块里代表的坐标系/尺度/参考帧不一致**。

建议新手先把数据流在脑子里固定成 6 步：

1) **Dataset** 输出 `views`（每个 view 是一个 dict）  
2) `BaseDataset` 自动补齐几何量：`ray_directions_cam / depth_along_ray / pts3d / pts3d_cam / camera_pose(_quats,_trans)`  
3) **Model** 读入 `views` 并预测 `preds`（每个 view 对应一个 pred dict）  
4) **Loss** 用 `batch+preds` 计算监督（可能是几何、mask、pose、det 等）  
5) **Eval/Metrics** 统计指标（必须和训练/可视化的采样规则一致）  
6) **Visualization** 把点云/框画出来（用于“人眼 sanity check”）

后面每个坑，本质都是“这 6 步里有一步的假设被破坏了”。

### 0.1 什么叫“正确的可视化基准”（先把 GT 画对）

**你后面所有的“预测是不是对了”的判断，都必须对照这张 GT 基准图。**

最稳的基准是：**真值 LiDAR 点云（GT pcd） + 真值 3D 框（GT boxes）叠在同一参考帧里**，并满足：

- 大地/路面点云呈现出明显的“rings”形态（激光扫描环）
- 大部分车辆框内能看到车辆表面点（至少不是“框里基本没点”）
- 点云整体方向/朝向合理（前方在 +X，右侧在 +Y，Z 向上；这是 OPV2V/CARLA 的约定）

推荐先用 3D HTML（交互）确认一次，再用 BEV PNG 快速迭代：

- GT 交互基准：`scripts/viz_opv2v_gt_pcd_boxes_html.py`
- 快速 BEV：`scripts/viz_opv2v_pred_bev_png.py`（也能画 GT）

如果你觉得“车点太稀疏”，先别怀疑坐标系——OPV2V 单车 LiDAR 远距离本来就稀疏；优先：
- 换更近/车更密集的帧（例如停车场/路口场景）
- 用 `--gt_lidar_mode fused` 把多车 GT LiDAR 叠加，密度会显著上升
- 确认不是可视化自己 `downsample/max_points` 把点采样没了（见 3.4）

### 0.2 一张图把 Pose/Geometry 分开（PredPose vs GTPose）

协同里最常见的“看着全错”，本质上只有两类：

- **Pose 错**：单 view 的点云形态其实对，但放到主车系的位置/朝向错了
- **Geometry 错**（depth/ray/scale/mask）：即使放在 GT pose 下点云形态也不对（扇形/长射线/塌缩）

因此我们所有可视化都推荐同时画两列：

- **PredPose**：用模型自己预测的 pose 放置点云（`pred['pts3d']`）
- **GTPose**：用 GT pose 放置点云（GT pose 作用在 `pred['pts3d_cam']` 上，只看几何不看 pose）

脚本入口：
- PNG：`scripts/viz_opv2v_pred_bev_png.py`
- HTML：`scripts/viz_opv2v_pred_vs_gt_boxes_html.py`

---

## 1. 坐标系大坑：MapAnything 用 OpenCV，OPV2V 用 CARLA

### 1.1 本质

**同一个 3D 点，如果你把它当成了另一套坐标系，就会出现：点云和框完全不对齐、车辆像‘漂’在空中、朝向怪异。**

- OPV2V（CARLA）常用定义（直觉版）：`X forward, Y right, Z up`
- OpenCV 相机坐标常用定义（直觉版）：`X right, Y down, Z forward`

因此必须有一个固定的基变换（仓库里用的就是这一套）：
- `data_processing/opv2v_pose_utils.py` 里的 `CARLA_TO_CAMERA_CV`
- 以及数据集里把 pose 统一转到 OpenCV 的 `_convert_pose_to_opencv()`（`mapanything/datasets/opv2v_cyl.py`）

### 1.2 新手怎么一眼判断“坐标系错了”

最简单的判据：**把真值 LiDAR 点云 + 真值 3D 框画在一起**，看“车的点云是否落在车框里”。

如果 GT 本身都对不上，那一定是坐标系/参考帧错了；不要继续训练（训练只会“学会错误的对齐”）。

推荐脚本（GT 基准）：
- `scripts/viz_opv2v_gt_pcd_boxes_html.py`（HTML 交互）
- `scripts/viz_opv2v_pred_bev_png.py`（BEV 快速 PNG，也能用来画 GT）

### 1.3 常见误区

- “我只转了 pose，没转点云/深度”：**不行**。同一条链路里所有 3D 量必须在同一坐标系里。
- “我把点云变换到了 main 车，但 box 还在 world/agent”：**不行**。点云/box 必须同参考帧。

### 1.4 行向量/列向量约定：为什么矩阵看起来像“转置/反了”

很多同学会卡在这：同一个 `CARLA_TO_CAMERA_CV`，有人写成 `R @ p`，有人写成 `p @ R`，结果完全不同。

**本质**：你在代码里到底把点当作 **列向量** 还是 **行向量**？

在本仓库的可视化脚本里（例如 `scripts/viz_opv2v_pred_bev_png.py`），点是 `(N,3)` 的 **行向量**，所以经常出现：
- `p_out = p_in @ R`（右乘）

一个最简单的“不会错”的自检是记住轴映射（OpenCV→CARLA）：

- OpenCV：`(x_right, y_down, z_forward)`
- CARLA：`(x_forward, y_right, z_up)`
- 因此 `p_carla = [z_cv, x_cv, -y_cv]`

只要你拿一个基向量测一下（例如 CV 的 `+Z` 应该变成 CARLA 的 `+X`），就能快速判断你的矩阵是否该转置/取逆。

---

## 2. 参考帧大坑：到底是“世界系 / 主车系 / view0 系”？

### 2.1 本质

协同里经常同时存在三种参考系：

- **world**：全局坐标（不用于直接可视化 GT box，除非你也把全部变换到 world）
- **main-agent ego**：你做融合/评估通常选的主车坐标
- **view0**：MapAnything 的多视图里默认第 0 个 view 是参考视图（很多相对 pose loss/metric 都是以它为基准）

坑点：如果 `view0` 不是你以为的主车（或者在训练中随机变化），**相对 pose 指标会非常不稳定**，甚至“看起来对但指标炸/看起来炸但指标对”。

### 2.2 解决方式（我们采用的）

- Dataset 侧提供 `ensure_main_agent_first=True`，保证 `view0` 就是主车（`mapanything/datasets/opv2v_cyl.py`）
- 多车超过 `num_views` 时，提供 `agent_selection_policy=nearest`，保证采样稳定（同文件）

对应 Hydra 透传已补齐：
- `map-anything/configs/dataset/opv2v/*/cyl_coop*.yaml`

### 2.3 多车圆柱模型到底怎么处理（最容易误解的点）

很多人第一反应是“把多车拼成一个更大的圆柱相机”，但我们实际采用的是更稳定的做法：

- **每辆车（每个 agent）各自把 4 个环视相机融合成 1 张圆柱 panorama**
- **多车协同 = 多个 panorama 作为多 view 输入给 MapAnything**

也就是说：
- `num_views` 指的是“参与协同的车辆数量”（不是 4 个相机数量）
- 一个 `view` 对应一个 `agent_id`（车），其 `img/depthmap/non_ambiguous_mask` 都来自该车的圆柱融合
- `view['camera_pose']` 在 coop 数据集里表示 **T_main_agent**（agent→主车坐标），并且统一到了 OpenCV 坐标

这件事有一个直接的 sanity check：
- 如果你启用了 `ensure_main_agent_first=True`，那么 `views[0]['agent_id']` 应该就是主车；
- 同时 `views[0]['camera_pose']` 应该接近单位阵（主车到主车）。

理解这一点，才能解释为什么“相对 pose”要用 `inv(T0)@Ti` 来算（见 3.1）。

### 2.4 Pose 表示也有坑：Quaternion 顺序 / 符号 / 归一化

如果你遇到“点云整体飞走”“pose_err_rel_mean 巨大”“明明看着差不多但突然炸”的情况，除了坐标系/参考帧，也要检查 pose 的表示本身：

**(A) Quaternion 顺序（XYZW vs WXYZ）**
- 本仓库里 `mapanything/utils/geometry.py:quaternion_to_rotation_matrix()` 明确约定：**(x, y, z, w)**（scalar-last）。
- 如果你把它当成 `(w, x, y, z)` 来用，旋转会完全错，典型现象就是点云/框朝向乱、位置漂移大。

**(B) q 与 -q 的“双覆盖”**
- `q` 和 `-q` 表示同一个旋转。
- 如果你的 loss 直接对 quaternion 做 L2，而不处理符号一致性，训练可能出现“来回翻转”或梯度不稳定。
- 更稳的做法是：用旋转矩阵/相对旋转角做 loss（或在 quat loss 里显式取 `min(||q-q_gt||, ||q+q_gt||)`）。

**(C) 归一化/近零/NaN**
- 微调早期 quaternion 可能被推到接近 0；直接 normalize 会产生 NaN，进一步把所有几何量污染掉（点云一片扇形/长射线）。
- `quaternion_to_rotation_matrix()` 已做了 `nan_to_num + eps clamp` 的鲁棒处理，但如果你在别处自己实现了 quat→R，一定要同样做防护。

新人建议：一旦 pose 指标异常，先用 `scripts/viz_opv2v_pred_bev_png.py` 看 PredPose vs GTPose；再用 `scripts/eval_opv2v_cyl_metrics.py` 的 `per_frame` 找出是“哪个 frame/哪个 agent”在拉爆。

---

## 3. “看着不错但 pose_err_rel_mean 很大”：评估范围不一致

### 3.1 pose_err_rel_mean 是什么（你到底在看什么）

在我们用的指标脚本 `scripts/eval_opv2v_cyl_metrics.py` 里：

- 先取 `view0` 为参考（通常就是主车）
- 对每个协同车 `i` 计算 GT 相对位姿：`T_rel_gt = inv(T0_gt) @ Ti_gt`
- 同样计算预测相对位姿：`T_rel_pred = inv(T0_pred) @ Ti_pred`
- 然后对 `T_rel_pred` vs `T_rel_gt` 计算：
  - 平移误差：`||t_pred - t_gt||`（单位 m，**3D 向量范数**）
  - 旋转误差：相对旋转角（单位 °）
- 最后对所有 `i>0`、所有评估帧求均值 → `pose_err_rel_mean=xx m / yy°`

所以它衡量的是“协同车相对主车的位姿”是否对齐；不是单车绝对定位。

### 3.2 本质（为什么会出现“图对但数爆炸”）

`pose_err_rel_mean` 这类指标如果把 **很远的协同车**（例如 100–300m）也算进来，会出现：

- 你只画 60m 范围 → 看着很对
- 指标统计全范围 → 远车误差巨大 → 数字“爆炸”

这不是模型突然坏了，而是**你比较的集合不一致**。

另外两个“更隐蔽但真实存在”的原因：

- 你只看 BEV（XY），但误差是 3D 范数；如果主要偏在 Z（例如高度漂移），BEV 看着还行但数会变大  
- 均值会被“少数极端错的 agent/帧”拉爆；建议看 `metrics.json` 里的 `per_frame` 找出最差的帧/agent 再定位

### 3.3 解决方式：训练/评估/可视化三者统一

我们新增并使用了：
- `max_agent_distance`：按主车坐标系 XY 距离过滤协同车（`mapanything/datasets/opv2v_cyl.py`）
- `agent_selection_policy=nearest`：优先选最近的协同车，避免“本次训近车，下次训远车”

并把同一套参数同时用于：
- 训练采样（Hydra dataset config 或 quick finetune 脚本参数）
- 指标脚本 `scripts/eval_opv2v_cyl_metrics.py`
- 可视化脚本 `scripts/viz_opv2v_pred_bev_png.py`

示例：把三者都固定在 60m：
- 采样：`--max_agent_distance 60 --agent_selection_policy nearest`
- 画图 crop：`--radius_max 60`

### 3.4 可视化参数也会“骗你”（点云稀疏/看起来错）

如果你看到“点云太稀疏/像没学到”，先排除下面这些“可视化自己造成的错觉”：

- `--mask_thresh`：阈值太高会把大量点过滤掉（早期建议先用 `0~0.3` 看全貌，再逐步提高）
- `--max_points`/downsample：如果你下采样到几千点，车辆表面很容易看起来“只有几粒点”
- `--z_min/--z_max`：切得太严会把车顶/地面直接裁没（BEV 只看 XY，更容易忽略 Z 被裁）
- `--radius_max`：你画 60m，但车在 65m；你会以为“框里没点”

---

## 4. 圆柱相机（cyl）的大坑：depth 的语义必须一致

### 4.1 本质

圆柱全景不是针孔相机：每个像素射线方向不同，depth 的含义也容易混。

两种常见 depth 定义：
- `depth_z`：沿相机 Z 轴的深度（OPV2V 原始针孔深度通常是这个）
- `depth_along_ray`：沿像素射线方向的距离（MapAnything 里很多地方用这个）

如果你拿 `depth_z` 当成 `depth_along_ray`，3D 点会整体“鼓起来/塌下去”，可视化会怪到离谱。

### 4.2 我们的处理（原则）

Dataset 在融合 4 相机到圆柱 panorama 时，显式把 OPV2V 的 `depth_z` 转成圆柱射线上的 `depth_along_ray`：
- 见 `CylindricalPanoramaBuilder.fuse()`（`mapanything/datasets/opv2v_cyl.py`）

新人建议：任何时候只要改了 `vertical_fov / elevation_center / azimuth_offset`，都先跑 GT 可视化基准再训。

---

## 5. “样本经常报错/训练特别慢”：约束不满足导致 __getitem__ retry 风暴

### 5.1 本质

设置了 `num_views=4` 且启用了 `max_agent_distance=60` 后，某些帧可能在 60m 内凑不够 4 辆车：
- `Requested 4 views but only 2 available`

如果 Dataset 允许 retry（`max_num_retries>0`），训练会不断随机换 idx 重试：
- GPU 看起来很闲、训练很慢、日志刷大量 retry

### 5.2 解决方式：初始化时预过滤

我们在数据集 `_load_data()` 结束后增加了预过滤：当设置了 `max_agent_distance` 时，直接丢掉“凑不够 required_views”的 scene（`mapanything/datasets/opv2v_cyl.py`）。

本质是：**把“运行时异常”变成“离线数据筛选”**，训练才会稳定。

---

## 6. 非歧义 mask（non_ambiguous_mask）的大坑：模型可以靠“全 0”作弊

### 6.1 本质

`non_ambiguous_mask` 监督的目的是让模型学会哪些像素是可靠几何区域。

但如果正样本比例很低/或 BCE 设计不当，模型可能输出全 0：
- loss 反而很小
- 几何监督失效
- 可视化点云稀疏到“像完全错了”

### 6.2 新手该怎么看

训练时重点盯这两个量（loss 已在训练日志里打印）：
- `NonAmbiguousMaskLoss_pred_mask_mean`
- `NonAmbiguousMaskLoss_gt_mask_mean`

正常情况下两者应在同一量级（例如 0.2~0.4），如果 `pred_mask_mean` 长期接近 0 或 1，就要先处理 mask。

附带工程坑：如果你把整个模型 `model.train()`，冻结模块里的 BN 统计也会被更新，mask 分支很容易崩。  
所以在 quick finetune 脚本里我们只对少数模块打开 train-mode（见 `scripts/finetune_opv2v_cyl_pose_quick.py` 的注释）。

---

## 7. “双卡 A800 想跑满，但总是被杀”：CPU RAM OOM ≠ GPU OOM

### 7.1 本质

这台机器是 2×A800 80GB，但 **系统内存只有 32GiB**。  
OPV2V 圆柱样本本身很大（每 view 包含 `pts3d/pts3d_cam/rays/masks` 等），DataLoader 一旦：
- `num_workers` 过大
- `prefetch_factor` 过大

就会在 CPU 端堆积大量 batch，触发 OOM，进程直接被系统 `SIGKILL`（日志里常见 exitcode -9）。

### 7.2 我们的落地设置

- DataLoader：`prefetch_factor=1`（`mapanything/datasets/__init__.py`）
- 推荐先从 `num_workers=2` 起步（`configs/dataset/opv2v_cyl_coop_ft_r60_nearest.yaml`）
- 先确保“稳定跑”，再一点点加吞吐

### 7.3 新手排查顺序

1) `free -h` 看 RAM 是否紧张  
2) `nvidia-smi` 看 GPU 是否真的忙  
3) `ps aux --sort=-%mem` 看是不是别的训练占了内存/显存  

---

## 8. 网络坑：torch hub / huggingface 下载会失败（需代理）

### 8.1 本质

即使你觉得“我用的是本地 checkpoint”，模型初始化时仍可能触发：
- torch hub 拉 DinoV2 repo/权重
- huggingface 相关请求

在本环境如果外网不稳定，会报 `RemoteDisconnected`。

### 8.2 解决方式（本仓库约定）

按仓库根目录 `AGENTS.md`：

```bash
cd /J6P-perception/yijinxiong_workspace/yijinxiong/clash && ./status.sh || ./start.sh
source /J6P-perception/yijinxiong_workspace/yijinxiong/clash/env.sh
```

或单条命令加：
`http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 ...`

---

## 9. Hydra/命令行坑：`${...}` 一定要加引号

### 9.1 本质

Hydra override 里经常用 `${dataset.xxx}`，但 bash 会把 `${...}` 当作 shell 变量展开，导致：
- `bad substitution`

### 9.2 解决方式

把包含 `${...}` 的参数用单引号包住，例如：

```bash
dataset.train_dataset='+ 40_000 @ ${dataset.opv2v.train.dataset_str}'
```

---

## 10. 一个“反例”复盘：为什么 DDP smoke 会把 pose 训炸？

我们做过一轮 DDP smoke（`experiments/opv2v_cyl_coop_ddp_smoke_r60_v8`），指标显示 pose 完全崩掉。  
本质原因是：**用“几何综合 loss”去从一个 pose 很差的初始化开始学，很容易先把 pose 学歪**，导致点云整体飞走。

更稳的方式是两阶段：
1) **先 pose-only 快速收敛**（相对位姿监督）  
2) 再做 geometry finetune（LiDAR 投影监督/深度监督），此时 pose 已经在正确 basin 附近

对应 quick 工具链：
- pose：`scripts/finetune_opv2v_cyl_pose_quick.py`
- geometry（LiDAR）：`scripts/finetune_opv2v_cyl_geom_quick.py`

---

## 11. 推荐的“每次训练都照做”的新手 Checklist

1) **先画 GT 基准**（点云 + 框）  
2) 固定评估/可视化范围（推荐 60m）  
3) 固定采样规则：`ensure_main_agent_first=True` + `agent_selection_policy=nearest` + `max_agent_distance=60`  
4) **确认 view0 就是主车**：看 `views[0]['agent_id']`，以及 `views[0]['camera_pose']` 是否接近单位阵（2.3）  
5) 训练中每隔一段：
   - 跑 `scripts/eval_opv2v_cyl_metrics.py` 输出 json
   - 跑 `scripts/viz_opv2v_pred_bev_png.py` 输出 png
    - （强烈推荐）同时看 PredPose vs GTPose，两列一起判断 pose/geometry（0.2）
6) 如果“图很好但数很差”，先检查是不是把远车算进去了/统计范围不一致（第 3 节）
7) 如果“点云很稀疏/像全错”，先检查是不是 `mask_thresh/downsample/z-range/radius` 造成的错觉（第 3.4 节）
8) 如果进程莫名其妙 exitcode -9，先查 CPU RAM（第 7 节）

---

## 12. 关键文件索引（便于你快速定位）

- 数据集与坐标变换：`mapanything/datasets/opv2v_cyl.py`
- 评估脚本（对齐采样/视野）：`scripts/eval_opv2v_cyl_metrics.py`
- 可视化（BEV PNG）：`scripts/viz_opv2v_pred_bev_png.py`
- 可视化（PredPose vs GTPose HTML）：`scripts/viz_opv2v_pred_vs_gt_boxes_html.py`
- GT 可视化基准：`scripts/viz_opv2v_gt_pcd_boxes_html.py`
- quick pose 微调：`scripts/finetune_opv2v_cyl_pose_quick.py`
- quick geometry 微调（LiDAR 投影监督）：`scripts/finetune_opv2v_cyl_geom_quick.py`
- 几何微调打法/复盘：`docs/cyl/opv2v_cyl_lidar_geom_finetune.md`
- 检测头踩坑大全：`docs/cyl/opv2v_cyl_detection_pitfalls.md`
- 多车训练方案（圆柱 4n→n 背景）：`docs/cyl/multi_car_training_plan.md`
- 训练调试清单（环境/步数/恢复/DDP）：`docs/cyl/opv2v_cyl_coop_debug_plan.md`
- 双卡数据配置（60m + nearest）：`configs/dataset/opv2v_cyl_coop_ft_r60_nearest.yaml`
- 双卡训练参数：`configs/train_params/opv2v_cyl_coop_a800_2gpu.yaml`
