# MapAnything 可视化核心工作流拆解

> 目标：把当前仓库里支撑可视化/评估的脚本与依赖关系一次性讲清楚，方便在一份“干净的” map-anything 代码库里复现，不必再临时东补西补。

## 1. 整体范围
- **数据**：以 `example/` 与仓库根目录下新增的 `000069.yaml`、`example/configs/*.yaml` 等 Carla/OPV2V 标注为入口，配套有多视角 RGB (`example/photo`)、深度 (`example/depth`、`depth/*.npy|.png`) 以及实测点云 (`example/*.pcd`).
- **模型**：本地缓存位于 `scripts/local_models/`（`config.json` + `model.safetensors`）。另有多个 `.pth` 检查点 (`checkpoint-best.pth` 等) 直接用于 Hydra 配置内。
- **可视化脚本**：集中在 `scripts/` 与 `depth/`，围绕 MapAnything 推理结果生成/比对点云、绘制框体与相机姿态、展示深度图。
- **坐标转换**：项目普遍依赖 UE/Carla ↔ OpenCV 的变换矩阵与 intrinsics 重标定逻辑，分别实现在 `scripts/depth_ge.py`, `scripts/color_compare.py`, `scripts/pose_compare.py`, `depth/*.py` 等模块中。

## 2. 关键依赖
| 组件 | 用途 | 由谁使用 |
| --- | --- | --- |
| `torch`, `torchvision` | 模型推理与 CUDA 选择 | 所有 MapAnything 推理脚本 |
| `open3d` | 加载/保存/渲染点云、绘制坐标系与框体 | `scripts/color_compare.py`, `compare_pointclouds.py`, `devided_pointclouds.py`, `pcd_with_boxes.py`, `visualize.py`, `depth/*.py` |
| `PIL.Image` | RGB/深度读取 | 所有预处理脚本 |
| `scipy.spatial.transform.Rotation` | Carla euler 角转旋转矩阵 | `scripts/color_compare.py` |
| `yaml`, `numpy`, `matplotlib` | 配置解析、统计、深度可视化 | `depth/*.py`, `scripts/depth_ge.py`, `pcd_with_boxes.py` |
| MapAnything 内部工具：`mapanything.utils.image`、`mapanything.utils.hf_utils.hf_helpers` | 输入预处理、Hydra/权重加载 | 全部推理脚本 |

> ✅ **复现提示**：保证这些依赖在同一个虚拟环境里，且 `mapanything` 包已通过 `pip install -e .` 或等效方式可用。`open3d` 与 `scipy` 在很多服务器默认未装，需要提前处理。

## 3. 数据与目录约定
1. **YAML 配置** (`example/configs/*.yaml`, `000069.yaml`): 包含 `cameraX`、`lidar_pose`、`vehicles` 等信息。大部分脚本假设图像/深度文件命名为 `[{yaml_stem}]_[camera].png`。
2. **图像/深度** (`example/photo`, `example/depth`): `scripts/depth_ge.py`、`color_compare.py`、`pose_compare.py` 都会在这些目录下查找匹配的 RGB/深度。
3. **点云** (`example/*.pcd`, `depth/*.npy|.png`): 
   - `scripts/compare_pointclouds.py`、`devided_pointclouds.py`、`color_compare.py` 需要 `.pcd` 基准。
   - `pcd_with_boxes.py` 可以同时加载 `.pcd/.ply/.bin/.npy`。
4. **模型/配置**：
   - `scripts/local_models/` 供 `.from_pretrained()` 直接读取。
   - `configs/train.yaml` + `checkpoint-best.pth` 提供 Hydra 模式下的加载 (`color_compare.py`).

> ✅ **复现提示**：在新的仓库里保持相同的目录结构，或在文档中显式说明如何通过参数将路径重定向；否则很多脚本的默认拼接规则会失效。

## 4. 共享逻辑（跨脚本调用）
1. **UE/Carla ↔ OpenCV 基变换** (`ue_c2w_to_opencv_c2w`): 在 `scripts/depth_ge.py`, `scripts/color_compare.py`, `scripts/pose_compare.py`、`depth/*.py` 里均有实现，用同一矩阵 `S = [[0,1,0],[0,0,-1],[1,0,0]]` 或其转置。
   - **⚠️ 常见坑：点云/BEV 指标必须先统一坐标约定**  
     OPV2V 的 GT LiDAR `.pcd`（以及 YAML `vehicles` 框体等）使用 CARLA/UE 约定（X 前、Y 右、Z 上），而 MapAnything 推理输出（`pred['pts3d']` / `pred['depth_z']` 反投影得到的点云）默认处于 OpenCV 约定（X 右、Y 下、Z 前）。  
     因此在做 **GT LiDAR 叠加可视化**、`z_min/z_max` 高度过滤、BEV IoU/Chamfer 等指标时，需要先把预测点从 OpenCV 转回 CARLA：`p_carla = S^T · p_cv`（等价行向量写法：`p_carla = p_cv @ S`）。  
     仓库内 `scripts/make_batch_eval_pcd_html.py` 已在生成 HTML 时自动做这一步（对旧产物也会启用启发式检测），避免“点云看起来被压扁/竖直拉伸”的错觉。
2. **深度真值的约定（OPV2V depth）**  
   `opv2v_depth/*_depth.npy` 是稀疏 `depth_z`（沿相机光轴的 Z 深度），与 MapAnything 输出的 `pred['depth_z']` 语义一致；不需要像点云那样做额外“轴交换”，只要相机位姿在同一约定下（CARLA ↔ OpenCV）一致即可。
2. **Intrinsics 重标定** (`rescale_intrinsics_to_image`): 用于在推理前将 YAML 内主点/焦距按真实图像分辨率缩放。
3. **视图预处理** (`mapanything.utils.image.preprocess_inputs`): 所有基于 YAML 的推理脚本在组装 `views` 后调用，返回模型可直接消费的张量。
4. **点云颜色/掩码对齐**：`color_compare.py`, `pose_compare.py` 等脚本会将 `pred['img_no_norm']` 与 `pred['mask']` 对齐，用颜色填充点云；默认 fallback 为蓝色。
5. **Open3D 可视化骨架**：
   - `compare_pointclouds.py`/`devided_pointclouds.py`: 双窗口 + 坐标系渲染。
   - `pcd_with_boxes.py`: 两窗共享 `o3d.visualization.Visualizer`，额外叠加 `TriangleMesh` 坐标系与 bounding boxes。

> ✅ **复现提示**：这些函数最好抽成公共模块（例如 `scripts/utils/coord.py`），否则复制脚本时容易遗漏尺度/坐标细节。

## 5. 可视化脚本拆解
### 5.1 `scripts/color_compare.py`
- **作用**：最完整的“预测 vs. 真实”比对工具，支持单/多 YAML、外部位姿、Hydra 加载 `.pth`、保存/上色点云、姿态误差统计。
- **依赖链**：
  1. `initialize_mapanything_local` (Hydra) + `scripts/local_models/config.json` 或 `.pth`。
  2. `load_views_from_config(s)` → `preprocess_inputs`。
  3. 外部姿态 → 生成 `camera_info_list`，供 `log_camera_pose_errors` 与点云对齐。
  4. 颜色与坐标转换：`pts3d_cam`→`pose_C2W_cv`→（可选）锚定到 ego/world，再经 `T_CV_TO_UE` 转到 UE 坐标，用 `open3d` 画出预测与真值（含颜色）。
- **可选参数**：
  - `--aggregate_frame [auto|ego|world]` 控制点云融合坐标。
  - `--max_height` 过滤异常点。
  - `--save_pred_path` 输出 `.pcd/.ply`。
  - `--no_viz` 允许在 headless 模式下只生成数据。
- **复现要点**：确保 Hydra 配置（`configs/train.yaml`）与 checkpoint 路径可被解析；外部 YAML 需含 `cords`、`intrinsic`、`extrinsic` 字段。

### 5.2 `scripts/compare_pointclouds.py` & `scripts/devided_pointclouds.py`
- **作用**：最小可用的本地对比脚本，只需 `--image_dir` 与 `--gt_pcd_path`。
- **行为**：
  1. 调 `MapAnything.from_pretrained('./local_models')`；
  2. `load_images` → `model.infer`；
  3. 将 `pred['pts3d']` 叠成单个点云，应用固定的 `R_x @ R_y` 旋转，使之与地面真值坐标系一致；
  4. 用 `Visualizer` 打开两个窗口。`devided_pointclouds.py` 版本额外加了 `VisualizerWithKeyCallback` 和 `C` 键显示/隐藏相机位姿。
- **复现要点**：确保模型权重已在 `scripts/local_models/` 下；如需换路径，修改 `model_path` 默认值。

### 5.3 `scripts/pcd_with_boxes.py`
- **作用**：在预测/真值点云上叠加 YAML 中的车辆框体 + 相机坐标系，支持不同输入格式与坐标切换。
- **流程**：
  1. `parse_yaml_data` 读取 `vehicles`、`lidar_pose`、`cameraX.cords`；
  2. `x_to_world`/`transform_boxes_to_ego_frame` 将框体转到自车坐标；
  3. `create_camera_geometries` 把每个相机位姿渲染为 `TriangleMesh` 坐标系，允许 `--pred_frame camera0` 等选项；
  4. `visualize_two_windows` 在两个窗口里展示预测/真值点云 + 一致的框体和相机轴。
- **复现要点**：YAML 里的 `vehicles` 角度必须齐全；若要支持其他标注，需要保持 `[location]/[center]/[extent]/[angle]` 语义。

### 5.4 `scripts/pose_compare.py`
- **作用**：批量遍历 `--config_dir` 下的所有 YAML，使用图像+位姿推理并聚合点云；同时可将预测保存为 `.ply/.pcd`。
- **特色**：
  - `parse_yaml_inputs` 会加载 `intrinsic/extrinsic` 并做 `ue→opencv` 转换；
  - 推理阶段默认开启混合精度、掩码、边缘裁剪；
  - 聚合后将点云从 OpenCV 坐标旋回 UE (`T_cv_to_ue`)，以便与地面真值一致。
- **复现要点**：`mapanything` 包需可 import；`sys.path.append(str(Path(__file__).resolve().parent / 'map-anything'))` 是针对当前目录结构的 hack，若在新仓库拆分，需要替换成正式安装方式。

### 5.5 `scripts/depth_ge.py`
- **作用**：多模态推理（RGB + 可选深度）+ 保存点云，核心逻辑同 `pose_compare.py`，但额外支持 `--depth_dir`、`--save_pred`、`--out_path`。
- **复现要点**：
  - 深度命名规则 `[stem]_[camera]_depth.png`；
  - `initialize_mapanything_local` 需要 `scripts/local_models` 或 `.pth`；
  - 输出可直接被 `pcd_with_boxes.py` 或其他 Open3D 工具消费。

### 5.6 其他辅助脚本
| 文件 | 功能 |
| --- | --- |
| `visualize.py` | 单一 `.pcd` 可视化（快速 sanity check）。 |
| `tree.py` | 目录结构巡检，确保数据集拷贝完整。 |
| `depth/depth.py` | 点云 → 多相机深度图、包围盒可视化，含保存 16-bit PNG/`npy`。 |
| `depth/batch_depth.py` 及同目录其他脚本 | 大规模批处理/清理深度资产（missing depth 检查、PNG 清理、可视化输出）。 |
| `gpu_test.py` | CUDA 环境连通性检测。 |

## 6. 训练与远程生产的衔接
- 训练/微调在外部服务器完成，**关键产物**是 `.pth`/`.safetensors` checkpoint。每次训练完需将 checkpoint 下载回本地，供以上脚本读取。
- `scripts/color_compare.py` 与 `scripts/depth_ge.py` 都支持通过 `--checkpoint_path` 或 `--model_path` 切换权重。
- 为保持再现性，建议：
  1. 约定 checkpoint 命名（例如 `checkpoint-finalYYYYMMDD.pth`）。
  2. 在文档中记录对应 Hydra config/override。
  3. 上传至版本控制或对象存储，并在脚本文档里注明下载路径。

## 7. 在新仓库中复现的步骤清单
1. **准备环境**：安装 `torch`, `open3d`, `scipy`, `Pillow`, `pyyaml`, `matplotlib`, `tqdm`，并可 import `mapanything`。
2. **同步资产**：把 `example/`、`depth/`、`scripts/local_models/`、`checkpoint-*.pth` 等目录原样迁移，或调整脚本参数指向新的位置。
3. **抽象公共模块**（推荐）：
   - `coord_utils.py`：放 `ue_c2w_to_opencv_c2w`, `T_CV_TO_UE` 等；
   - `io_utils.py`：封装 `.pcd` 读取、颜色解码；
   - `visualization.py`：统一双窗口/键盘交互逻辑。
   这样 Codex 能快速组合成新的命令行工具，而不会遗漏共享逻辑。
4. **验证路径**：
   - 使用 `scripts/compare_pointclouds.py`（最小依赖）跑通一次；
   - 再跑 `scripts/color_compare.py --config_path example/configs --image_dir example/photo --gt_pcd_path example/641_000069.pcd`，确认外部姿态/颜色/保存功能正常；
   - 启动 `scripts/pcd_with_boxes.py` 检查框体/相机姿态渲染。
5. **记录差异**：若新的 map-anything 版本在 API 或张量命名上有变动（例如 `pred['pts3d_world']`），需要在此文档补充“适配说明”，确保 Codex 在未来有明确的替换点。

通过以上整理，即便在仅包含“核心可视化”代码的新仓库里，Codex 也能据此理解：有哪些输入资产、脚本之间如何协作、关键的坐标/颜色处理逻辑是什么，从而完整复刻当前项目的可视化能力。
