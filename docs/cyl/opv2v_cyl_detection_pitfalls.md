# OPV2V 圆柱协同：检测头微调踩坑大全（新手友好）

> 目标：在 OPV2V 上把 **3D 检测框 AP/mAP** 拉起来（核心），点云重建/几何只做辅助；同时要求可视化里 Pred 框和 GT 框尽量对齐、不要“满屏垃圾框”。
>
> 本文是这次迭代里遇到的坑的“**现象 → 本质 → 如何确认 → 怎么修/怎么绕**”总结；尽量从数据/代码/评测/可视化/训练动态多个角度，帮助新手建立稳定的排错框架。

---

## 0. 先建立一个最小心智模型（否则很容易“修错方向”）

你看到的“检测框漂移 / AP 近似 0 / 满屏框 / 只剩一个框”往往不是模型学不会，而是**接口契约不一致**导致“预测是对的但解释方式错了”。

把整个链路拆成 5 个边界，每个边界都有典型坑：

1) **数据集 → 输入字段（keys）**  
`camera_pose/camera_poses`、`ray_directions`、`depth_z`、GT boxes（坐标系 + yaw 单位）  

2) **坐标系（CARLA/UE vs OpenCV）**  
OPV2V GT LiDAR/GT boxes 在 **CARLA/UE**：`X forward, Y right, Z up`  
MapAnything 内部几何/点在 **OpenCV**：`X right, Y down, Z forward`  

3) **det head 的栅格定义（x_range/y_range/voxel_size）**  
训练时用的是哪个网格，评测解码时必须完全一致。  

4) **解码（heatmap/reg → boxes）与后处理（NMS/阈值）**  
score 阈值、local-peak、pre-topk、NMS IoU、max_dets 都会改变“框数/观感/AP”。  

5) **可视化过滤**  
可视化脚本的 `pred_min_score`/`pred_max_boxes` 不是评测阈值；它能让你看到“只有一个绿框”，但那不代表模型只预测了一个框。

---

## 1. 这次到底做了什么（为了让你能把坑和改动对上号）

### 1.1 加了/训练了一个密集 BEV 检测头（CenterNet 风格）

- **Head**：`mapanything/mapanything/models/mapanything/detection_head.py`  
  - `DifferentiableBEVRasterizer`：把点云（ego CARLA）栅格化为 BEV 特征（默认 `density/mean_z`，可选 `high_ratio/var_z`）。  
  - `BEVCenternetHead`：卷积 trunk → 输出 `heatmap`（sigmoid）和 `reg`（8 通道）。  
    - `reg` 语义（训练/解码一致）：`[dx, dy, log_l, log_w, z, log_h, sin_yaw, cos_yaw]`。

- **Det wrapper**：`mapanything/mapanything/models/mapanything/det_model.py`  
  把 MapAnything 的多视角预测点（OpenCV）对齐到 ego，再转换为 CARLA/UE，喂给 BEV head。

- **Loss**：`mapanything/mapanything/train/losses.py` → `BEVCenternetDetLoss`  
  - heatmap：CenterNet 风格 focal loss（`focal_alpha/focal_beta`）。  
  - reg：只在正样本网格（GT box center）上做 `L1`。  
  - 额外加了一个可选项：`hard_neg_topk/hard_neg_weight`（显式惩罚 topK 最高分负样本）用于压掉“密密麻麻中等分数假阳性”。

### 1.2 把“满屏框/只剩一个框”的问题压到可控

解码+后处理主要在 `scripts/batch_eval.py`：

- `decode_bev_centernet()`：  
  - 用 **local-maximum peak**（CenterNet-style）而不是 “所有格子都解码”，避免一上来几千个框。  
  - `pre_nms_topk`：NMS 前只保留 topK，限制复杂度。  
  - **点支撑过滤（support filtering）**：利用 BEV 特征（density/high_ratio/var_z）过滤空旷区域的高分噪声。  
  - oriented BEV IoU NMS：`_nms_bev_oriented()`。

### 1.3 修了两个“会让 AP 直接假死”的根因坑

#### 坑 A：推理阶段输入 key 被改写，det 对齐读不到 pose（指标≈0，框漂移）

- **本质**：`model.infer()` 的预处理会把 `camera_poses` 转成 `camera_pose_quats` / `camera_pose_trans` 并删除原 key；  
  但 det 对齐逻辑最初只读 `camera_pose/camera_poses`，推理/评测时就拿不到 GT pose，导致点对齐错。
- **修复**：`mapanything/mapanything/models/mapanything/det_model.py` 的 `_view_pose_for_detection()`  
  - 兼容 `camera_pose`、`camera_poses`、以及 `camera_pose_quats/trans`（XYZW order）三种输入。

#### 坑 B：评测解码网格默认值与 det head 配置不一致（训练正常但评测 AP≈0）

- **本质**：训练用了 `x_range=[-50,120]` 的 wide head，但 `batch_eval.py` 的 CLI 默认仍是 `x_min=0,x_max=120`；  
  同一张 heatmap 被按“不同坐标系的网格”解码，预测中心整体错位 → TP 全没了 → AP 直接假死。
- **修复**：`scripts/batch_eval.py` 在 `main()` 里自动读取 `configs/model/det_head/<det_head_cfg>.yaml`：  
  当 CLI 参数仍是默认值时，自动把 `det_x/y/voxel_size` 对齐到 det head 配置，避免 silent corruption。

---

## 2. 踩坑大全（现象 → 本质 → 如何确认 → 怎么修/怎么绕）

### 2.1 数据/路径类：看起来像“代码坏了”，其实是“数据没被正确读到”

#### 坑：数据解压后的目录层级不一致，导致脚本找不到文件

- **现象**：`FileNotFoundError`（找不到 `.yaml/.png/.pcd/.npy`），或评测采样帧数为 0。  
- **本质**：OPV2V tar 解压后可能多一层目录；脚本默认从 `map-anything/data/opv2v` 与 `map-anything/data/opv2v_depth` 读。  
- **如何确认**：看目录是否满足：  
  - `data/opv2v/<split>/<sequence>/<agent>/<frame>.yaml`  
  - `data/opv2v/<split>/<sequence>/<agent>/<frame>_<cam>.png`  
  - `data/opv2v/<split>/<sequence>/<agent>/<frame>.pcd`  
  - `data/opv2v_depth/<split>/<sequence>/<agent>/<frame>_<cam>_depth.npy`
- **怎么修**：  
  - 解压后把软链接指向正确根目录：`map-anything/data/opv2v`、`map-anything/data/opv2v_depth`。  
  - 避免拷贝大目录，优先软链接。

#### 坑：从仓库根目录启动训练/评测，导致 `model.pretrained=experiments/...` 找不到

- **现象**：训练一开始就 `FileNotFoundError: experiments/.../checkpoint-final.pth`，但你明明在磁盘上看到了 ckpt。  
- **本质**：多数 bash 脚本默认 `cd map-anything` 后再运行 `python scripts/train.py`；  
  你如果在更上层目录执行（例如 `vggt_series_4_coop/`），相对路径 `experiments/...` 就会指向错误位置。  
- **如何确认**：报错路径里缺少 `map-anything/` 前缀，或者 `pwd` 不是 `.../map-anything`。  
- **怎么修**（两选一）：  
  - 进入目录再跑：`cd map-anything && python scripts/train.py ...`  
  - 或者把 `model.pretrained` 改成绝对路径/带 `map-anything/` 前缀的路径。

### 2.2 网络/依赖类：torch hub 下载失败导致训练/加载“莫名其妙”

#### 坑：torch hub 访问外网不通（卡住/下载失败/随机报错）

- **现象**：首次启动训练时下载权重失败，或间歇性失败。  
- **本质**：外网不可直连，需要走本地代理；或者 torch hub cache 不完整。  
- **怎么修**：按仓库根目录 `AGENTS.md`：  
  - `cd /J6P-perception/yijinxiong_workspace/yijinxiong/clash && ./status.sh || ./start.sh`  
  - `source /J6P-perception/yijinxiong_workspace/yijinxiong/clash/env.sh`  
  - 或训练脚本里显式关掉 hub：例如 `scripts/batch_eval.py` 已强制 `model.encoder.uses_torch_hub=false`。

### 2.3 训练资源类：训练被杀/不收敛，常见是“机器状态”而非算法

#### 坑：训练过程中被系统杀（Signal 9/15），看起来像不稳定

- **现象**：训练中断，日志里出现 `Killed`、`Signal(9)`、`Signal(15)`。  
- **本质**：GPU/CPU 内存被别的进程占满，或 dataloader workers 过多导致系统 OOM。  
- **怎么修**：  
  - 先 `nvidia-smi` / `ps aux | rg train.py` 清理后台作业。  
  - 降低 `dataset.num_workers`、batch size，必要时用 `accum_iter` 做梯度累积。  

### 2.4 坐标系类：框/点云看起来“差一点点”，其实是“差一个转置/差一个坐标系”

#### 坑：CARLA/UE 与 OpenCV 坐标混用（框和点永远对不齐）

- **现象**：Pred 点云/Pred 框整体“旋转/翻转/平移”偏移；AP 极低；但训练 loss 不一定爆炸。  
- **本质**：OPV2V GT 的点/框在 CARLA（X前Y右Z上）；模型几何在 OpenCV（X右Y下Z前）。  
- **如何确认（强烈推荐新手必做）**：先只画 GT：  
  - `scripts/viz_opv2v_gt_pcd_boxes_html.py`（GT LiDAR + GT 框叠加）  
  如果 GT 自己都不对齐，那后面都不用看。  
- **怎么修**：统一约定“每个张量当前在哪个坐标系”，并把转换封装在 1~2 个地方；不要在脚本里临时 `@` 来 `@` 去。  
  - 这次的“可视化叠加”明确把预测点转换回 CARLA：`scripts/make_batch_eval_pcd_html.py`。

#### 坑：行向量/列向量约定不清（矩阵要不要转置？）

- **现象**：你写了一个“看起来对”的转换矩阵，结果方向对了但数值还是错，或者左右翻转。  
- **本质**：同一个 3x3 旋转矩阵，**列向量**写法与 **行向量**写法相差一个转置；而代码里常见 `points @ R`（行向量）与 `R @ points`（列向量）混用。  
- **怎么修**：写在代码注释里，并用 1 个简单向量做单元自检：  
  - 例如 CARLA 的 `+Z` 应该对应 OpenCV 的 `-Y`（因为 OpenCV Y 向下）。  

### 2.5 输入字段类：最隐蔽、也最容易把你带沟里（“看起来模型不行”）

#### 坑：`infer()` 会删除 `camera_poses`（det head 拿不到 pose，点对齐错）

- **现象**：训练脚本/数据集没问题，但只要你走 `model.infer()` 路径（评测/可视化），AP 突然≈0，框漂移明显。  
- **本质**：`mapanything/utils/inference.py:preprocess_input_views_for_inference()` 会把 `camera_poses` 拆成 `camera_pose_quats/trans` 并 `del processed_view["camera_poses"]`。  
- **如何确认**：在推理前打印 `view.keys()`，看 `camera_poses` 是否还在。  
- **怎么修**：  
  - det 侧兼容两种 key（已在 `det_model.py` 修）。  
  - 或你的评测/脚本不要依赖“原始 key 名称”，而是封装一个“取 pose”的函数。

#### 坑：评测脚本默认会把 GT pose 输入剥离掉（你以为在测 det，其实在测另一个问题）

- **现象**：你训练时用 `posed_sfm` + GT pose 对齐，评测时 AP/可视化突然很差。  
- **本质**：`scripts/batch_eval.py` 默认 `strip_external_calibration_inputs(processed_views)`，会移除外部 GT pose，让模型以 image-only 方式跑；  
  但 det head 如果依赖 GT pose（`point_pose_source=gt`），这会直接破坏点对齐。  
- **怎么修**：评测 det head 时务必加：  
  - `--keep_camera_poses --model_task posed_sfm`  

### 2.6 解码/评测类：AP 低不一定是模型，常常是“解释方式不一致”

#### 坑：评测 task 与训练 task 不一致（AP 可能差一个数量级）

- **现象**：训练用 `images_only`，评测不小心走了默认 `posed_sfm`，AP/可视化突然变得很差（或反过来“虚高”）。  
- **本质**：`model/task` 决定是否把 GT camera pose/几何输入给模型，点云分布会发生明显变化；  
  det head 只训练了其中一种分布，换任务配置相当于“换了输入域”。  
- **如何确认**：看评测日志里 `geometric_input_config`：  
  - `overall_prob=0, cam_prob=0` → `images_only`  
  - `overall_prob=1, cam_prob=1` → `posed_sfm`  
- **怎么修**：训练/评测保持同一个 `--task`。  
  - det head 训练用 `images_only` 就评测 `--task images_only`；  
  - 如果训练改成 `posed_sfm`，评测也要显式 `--task posed_sfm`。

#### 坑：det head 配置与评测解码网格不一致（AP 假死）

- **现象**：loss 在降、可视化看着“差不多”，但 AP 很低甚至 0。  
- **本质**：heatmap 的 (x,y) 像素到米的映射由 `(x_min,y_min,voxel_size)` 决定；你训练/解码必须一致。  
- **如何确认**：随便取一个预测中心，把它的 (ix,iy) 手算成 (x,y) 看是否落在合理范围（比如车应该在前方 0~80m）。  
- **怎么修**：  
  - 指定 `--det_head_cfg` 并让 `batch_eval.py` 自动对齐网格（已加）。  
  - 如果你手动改了 CLI 参数，确保它们与训练配置一致。

#### 坑：可视化阈值≠评测阈值（“只剩一个绿框”通常是你自己过滤掉了）

- **现象**：HTML 里只看到 1 个绿框（Pred），红框（GT）很多；你以为模型只预测了一个框。  
- **本质**：`scripts/make_batch_eval_pcd_html.py --pred_min_score` 默认 0.3，会把大量 0.05~0.2 的预测框直接过滤掉；  
  但评测可能是 `--det_score_thresh 0.05`，两者不是一回事。  
- **怎么修**：  
  - 做 debug 时建议 `pred_min_score` 跟评测一致（例如都用 0.05），或者至少把两个值写在页面标题/日志里。

#### 坑：`eval_opv2v_cyl_det_metrics.py` 导入失败（ModuleNotFoundError: scripts.batch_eval）

- **现象**：运行 `scripts/eval_opv2v_cyl_det_metrics.py` 直接报 `ModuleNotFoundError: scripts.batch_eval`。  
- **本质**：Python 环境里往往有一个第三方包名也叫 `scripts`，会覆盖仓库内 `map-anything/scripts/` 目录；  
  结果 `import scripts.batch_eval` 被解析到 site-packages 的 `scripts`，里面没有 `batch_eval.py`。  
- **如何确认**：`python -c "import scripts; print(scripts.__file__)"` 会指向 `site-packages/scripts/__init__.py`。  
- **怎么修**：  
  - 在脚本里显式把 `map-anything/scripts` 插到 `sys.path` 头部，并直接 `from batch_eval import ...`。  
  - 或者临时加环境变量：`PYTHONPATH=$PWD/scripts:$PYTHONPATH`（让本地 scripts 先被命中）。

#### 坑：dense heatmap 直接解码会产生“无数框”（NMS 也救不了）

- **现象**：推理可视化里满屏框；precision 极低。  
- **本质**：如果把每个超过阈值的像素都解码成框，本质上是在把 heatmap 当“逐像素分割”，而 CenterNet 本来就要求 **peak** 才是目标中心。  
- **怎么修**（已在 `decode_bev_centernet()` 做）：  
  - local-maximum peak 筛选（maxpool 比较）  
  - pre-NMS topK 截断  
  - 支撑过滤（density/high_ratio/var_z）  
  - 合理的 `det_score_thresh` 与 `det_nms_iou`

### 2.7 Loss/训练动态类：同样的 head，loss 细节能决定“满屏框”还是“框干净”

#### 坑：heatmap 初始 bias 太高 → 一开始全是高分噪声

- **现象**：训练初期大量中等分数框，且很难压下去。  
- **本质**：heatmap head 最后一层 bias 决定初始输出的 prior objectness。  
- **怎么修**：把 bias 设更负（例如 `-4.0`），让初始 objectness 更低：  
  - `configs/model/det_head/bev_centernet_wide_v2.yaml: heatmap_bias: -4.0`

#### 坑：focal 超参不合适，会出现“看着收敛但框不干净”

- **现象**：loss 下降，但预测框数量还是很多，AP 提升慢。  
- **本质**：dense 任务里负样本远多于正样本；focal 的 `alpha/beta` 会改变对“中等分数负样本”的惩罚力度。  
- **怎么修**：这次有效的组合（仅供起点）：  
  - `configs/loss/opv2v_det_only_wide_v4.yaml`：`heatmap_weight=4.0, focal_alpha=1.0, focal_beta=4.0`  

#### 坑：硬负样本惩罚（hard neg）用过头会“矫枉过正”

- **现象**：框数量骤减，甚至只剩 0~1 个框；recall 掉得厉害。  
- **本质**：hard-neg 会强行压掉最高分负样本；如果权重/TopK 太大，会把“困难正样本”周边也一起压没。  
- **怎么修**：  
  - 先从 `hard_neg_topk=0` 开始，靠 peak+NMS+阈值把框收敛到“能看”；  
  - 再小步引入 hard-neg（例如 topk=50/100、weight=0.05/0.1）并同时监控 recall。  

### 2.8 关键瓶颈：GT 框里没有预测点 → 3D 检测 recall 会“天花板”

> 这是我认为这条路线里**最容易被忽略、但又最致命**的一点：  
> 如果 det head 的输入是“重建点云/点图（points）”，那么它的 recall 受限于一个硬条件：**GT 车框里必须有足够多的预测点**。  
> 否则无论 det head 多强，都只能在“没点的地方瞎猜”，最终 AP/recall 会卡死。

- **现象**：  
  - 你会看到少数近处车辆框很准（IoU 高），但大部分车辆完全不出框；  
  - 或者 det head 产出很多框，但真正覆盖 GT 的 TP 很少（FP 很多、FN 很多）。

- **本质**：  
  - “点云重建是辅助”在这里会被迫变成一个**必要条件**：  
    用点做检测，等价于先做一个“隐式的 3D 可见性/几何重建”，没有重建到目标的点，就没有足够信息定位 3D box。

- **如何确认（最直接的 sanity check）**：  
  - 统计每个 GT box 内的预测点数量（ego CARLA/UE 坐标系一致时）：  
    - 如果大量 GT box 的 in-box 点数为 0（或极少），那 det recall 低就是必然结果。  
  - 例如在一次代表帧上，`27` 个 GT 车框里有 `21` 个框内预测点数为 `0`（这时 det head 只能最多找回剩下那几个有点的车）。

- **怎么修 / 怎么绕**：  
  - **修（走“点云→BEV 检测”这条路）**：提高车辆的重建覆盖率（让更多 GT 框内有点）。常用方向：  
    - 增强车辆相关的几何监督（例如更强的 `VehicleBoxPointAlignmentLoss` / 更合理的采样策略）；  
    - 减少会“把车辆当 outlier 丢掉”的机制（例如过强的 top-n% 排除、过强的 mask/置信度过滤）。  
  - **绕（改变 det head 输入）**：把 det head 从“纯点云”升级为“图像特征 lift 到 BEV”（LSS/BEVFormer 类），即使点云稀疏也能做检测；点云重建退回辅助。

### 2.9 高度过滤（z filter）非常容易设错：OPV2V ego 坐标原点在 LiDAR，高度是负的

- **现象**：一开 `point_z_min/point_z_max`，检测直接崩（AP 断崖式下降），或者几乎不出框。  
- **本质**：OPV2V 的 ego/车体坐标系原点在 LiDAR（不是地面），所以：  
  - 地面点的 `z` 通常在 `-1.8 ~ -2.0` 左右（取决于 LiDAR 安装高度）；  
  - 车辆点大多也在负 `z` 区间（例如车顶可能接近 `-0.5` 左右）。  
  因此像 `z_min=0.2` 这种“看起来合理”的阈值，实际上是在**把车辆点几乎全过滤掉**。
- **如何确认**：随便取一帧 GT LiDAR `.pcd`，打印 `z` 分位数；你会看到中位数在 `-1.x`。  
- **怎么修**：  
  - 不要凭感觉拍 `z` 阈值；先看点云 `z` 分布，再决定要不要滤、滤到哪。  
  - 如果目标是“去地面”，常见做法是把 `z_min` 设到 `-1.5` 左右（保留车身上半部分），并配合 density support（否则车中心可能没点）。  

---

## 3. 一个新手也能跑通的“闭环检查”流程（推荐照着做）

> 目的：避免你还没修坐标系/输入字段就开始调 loss，最后陷入“越调越乱”。

1) **GT 基准先过**（只看真值，确认坐标可信）  
   - `PYTHONPATH=$(pwd) python scripts/viz_opv2v_gt_pcd_boxes_html.py ...`

2) **只评测 1~5 帧，强制保留 GT pose 输入**（确认 det head 逻辑链路打通）  
   - `PYTHONPATH=$(pwd) python scripts/batch_eval.py --det_metrics --keep_camera_poses --model_task posed_sfm --sample_size 5 ...`

3) **保存代表帧 + 画 GT/Pred boxes**（肉眼校验“框数是否合理、是否对齐”）  
   - `batch_eval.py` 加 `--save_representative`  
   - `PYTHONPATH=$(pwd) python scripts/make_batch_eval_pcd_html.py --eval_root <...> --draw_gt_boxes --draw_pred_boxes --pred_min_score 0.05`

4) **如果 AP≈0：先假设是“坐标/网格/pose key”问题，而不是模型**  
   优先排查：  
   - 是否忘了 `--keep_camera_poses`  
   - `det_head_cfg` 是否和训练一致  
   - `det_x/y/voxel_size` 是否被覆盖成错的  
   - det_model 是否拿到了 pose（`camera_pose_quats/trans` 兼容问题）

---

## 4. 当前这轮的最好结果（作为你对齐的参照点）

> 注意：这是 `test` 抽样 50 帧、IoU=0.5、`score_thresh=0.05`、`max_dets=100` 的结果；不是全量 mAP。

当前最好（e2e v5：更强 hard-neg + vehicle loss 不回传 scale）：

- checkpoint：  
  `map-anything/experiments/mapanything/training/opv2v_det_e2e_posegeom_det_v5_hardneg_scale_detach/20260116_170846/checkpoint-final.pth`
- 评测汇总：  
  `map-anything/eval_runs/det_e2e_v5_eval_t005/summary_test.json`  
  - `det_ap_iou=0.1992`，`precision=0.1942`，`recall=0.2280`，`mean_iou=0.6975`
- 可视化（含 GT 红框 + Pred 绿框）：  
  - `map-anything/eval_runs/det_e2e_v5_eval_t005/html/index.html`  
  - 更干净（更少框）的版本：`map-anything/eval_runs/det_e2e_v5_eval_t005/html_slim/index.html`

历史对照（det head-only）：  
`map-anything/eval_runs/det_head_only_single_stage1_wide_v2_focalalpha1_eval_posefix_xyfix/summary_test.json`（`det_ap_iou=0.1568`）

---

## 5. 本质总结：这些坑为什么高频？

因为 3D 检测在这个项目里并不是“从头到尾一个坐标系/一个接口”，而是“在多个模块边界上拼接”：

- 数据集定义了一套坐标与字段名；
- `infer()` 又定义了一套推理输入格式（会改 key）；
- det head 训练定义了一套 BEV 网格；
- 评测/可视化又定义了各自的阈值与解码方式。

只要其中任意两处不一致，就会出现“模型其实学到了一些东西，但你看到的框完全不对”的错觉。

因此排错的第一原则永远是：**先保证契约一致（keys/坐标/网格/阈值），再谈模型与 loss。**
