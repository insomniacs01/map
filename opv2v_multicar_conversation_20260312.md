# OPV2V 多车协同问题排查与修复对话整理（本轮）

> 整理日期：2026-03-12  
> 当前目录：`/Users/macbookpro/repos/map`  
> 远端工作目录：`/home/qqxluca/map-anything3`  
> 远端主机：`THU2204_10_3090`

---

## 1. 本轮目标

本轮对话的核心目标是：

1. 继续排查并修复 `map-anything3` 中 OPV2V 多车协同训练效果不佳的问题。
2. 不只看“能不能跑”，而是要找出“为什么点云效果不好”的真实原因。
3. 每解决一个问题就训练、评测、分析，逐步逼近一版真正可用的多车协同方案。
4. 将有效修复同步提交到 GitHub：`insomniacs01/map`。

---

## 2. 用户提出的核心问题与初始判断

用户最早描述的问题是：

- 单车 4 相机协同感知训练微调已经完成；
- 多车协同仍然有很大问题；
- 训练后点云可视化“完全不能看”；
- 怀疑原因包括：
  - 车与车之间距离太远，共同特征太少；
  - 同车相机之间距离很近，而跨车相机距离很远，存在明显尺度差异；
  - 希望通过“打标签”的方式让模型知道同车更近、异车更远；
  - 还怀疑模型对“车辆”这一对象理解不足，想通过更多车辆图像帮助模型理解协同对象。

对这些想法的分析结论是：

- “距离太远、overlap 太少”这个方向是对的，但不是唯一根因；
- “同车打相同标签”这个想法方向也对，但如果只是改 `label` 字段，本身不会自动形成几何约束；
- 真正更关键的问题在于：
  - 数据采样是否保留了同车 rig 结构；
  - 模型是否显式知道 `agent identity` / `camera identity`；
  - 损失函数里是否真的在约束跨车相对位姿与世界坐标一致性；
  - 训练和评测链路本身是否稳定，不会被 OOM / 数据无效重试 / 指标错误误导。

---

## 3. 早期排查后形成的根因框架

在阅读 `~/map-anything3` 代码与配置后，逐步形成了以下判断：

### 3.1 不是单一的“车太远”问题

多车协同效果差并不是单一由“距离太远”造成，而是多项问题叠加：

1. **数据层问题**  
   多车多相机视角在采样时容易被打平，破坏同车 4 相机的刚体 rig 结构。

2. **模型层问题**  
   模型缺少显式的车辆身份和相机身份信息，导致它更难区分“这是同一辆车上的不同相机”还是“这是不同车之间的视图”。

3. **损失层问题**  
   如果没有显式启用 pairwise relative pose loss、world-frame point consistency 等约束，仅靠隐式特征对齐很难稳定学好多车几何关系。

4. **训练基础设施问题**  
   包括显存利用不充分、验证阶段放大 batch 导致 OOM、无效样本重试拖慢训练、评测脚本指标定义错误等问题。

### 3.2 用户原始想法中哪些是对的

- “同车应该更近、异车应该更远”：方向是对的；
- 但不应只停留在标签层，应该通过：
  - 结构化采样；
  - 身份嵌入；
  - 几何损失；
  - curriculum 采样；
  来落地。

- “让模型理解车辆”：方向部分正确；
- 但当前更核心的不是语义识别“车辆是什么”，而是建立**稳定的跨视角几何关系**。

---

## 4. 本轮之前已经完成的关键修复基础

在本轮继续推进之前，前面已经完成了一批关键改造，这些是后续工作的基础：

### 4.1 数据与采样侧

- 为 `OPV2VCoopDataset` 增加结构化采样；
- 支持按“2 车 × 每车 4 相机”的方式返回样本；
- 支持距离过滤、完整 rig 约束、agent 选择策略等；
- 支持输出 identity metadata。

### 4.2 模型侧

- 增加 `agent_identity_embedding`；
- 增加 `camera_identity_embedding`；
- 在 encoder feature 上注入身份偏置；
- 保持与旧 checkpoint 兼容。

### 4.3 损失与配置侧

- 新增协同训练损失配置；
- 显式启用 pairwise relative pose loss；
- 显式启用 world-frame points loss；
- 新增 OPV2V 协同模型、数据、训练参数配置。

### 4.4 基础设施侧

- 修复验证阶段 OOM；
- 增加可单独控制的验证显存预算；
- 修复评测导出链路；
- 增加正式训练和对照实验配置；
- 将执行计划与问题分析整理进 `docs/`。

---

## 5. GPU、训练吞吐与稳定性相关对话整理

用户多次追问 GPU 与训练状态，主要包括：

- 训练需要多久；
- 是否还有空闲显卡；
- 能不能多卡一起用；
- 为什么显卡没有跑满；
- 为什么先“吃满单卡再加卡”；
- 训练是不是崩了，崩了就修复继续跑；
- 训练轮次过少、数据过少会不会没说服力；
- 原始多车并非完全坏掉，而是效果不好，希望真正找出效果差的原因。

围绕这些问题，本轮前后的结论包括：

### 5.1 GPU 没跑满的真实原因

不是 dataloader 卡死，也不是模型根本没训练，而是：

- `dataset.num_views = 8`
- `train_params.max_num_of_imgs_per_gpu = 8`
- 实际 batch size 会被折算为 `max_imgs // num_views`

也就是说，当时每卡每步真实样本数只有 `1`，自然导致：

- 显存只占到约 `8.6GB`；
- GPU 计算波峰很短；
- `nvidia-smi` 看起来就像没有吃满。

### 5.2 单卡吃满后的稳定档位

后续探针确定：

- `bs160` 是当前 `224x126 / 8 views / lowmem / encoder frozen` 下的稳定高吞吐档位；
- 显存约 `19.7GB`；
- `176` 及以上开始 OOM。

结论：

- 先把单卡吃满再谈多卡，是更稳妥的路线；
- 不是不能多卡，而是要先在单卡上验证配置本身稳定与有效，不然多卡只会把问题放大。

### 5.3 验证阶段 OOM 的真正原因

训练本身在 `bs160` 下稳定，但验证阶段曾 OOM。根因是：

- 训练 batch size 被折算为 `20` 个多视角样本；
- 验证逻辑却被固定写成训练的 `2x`；
- 导致验证 batch 变成 `40`，显存直接冲爆。

修复方式：

- 新增 `max_num_of_test_imgs_per_gpu`；
- 验证不再盲目 `2x`；
- 进入验证前清理缓存。

修复后已能完整跑过验证集并继续训练。

---

## 6. 近邻版 `near40` 的实验与本轮最关键的错误发现

### 6.1 固定 20 帧正式评测

为了比较 `base` 与 `near40`，做了固定 `20` 帧、固定快照的正式评测：

- 原始评测目录：  
  `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/eval_bs160_epoch0_test20_seed42`

原先基于这份结果得到的粗判断是：

- `near40` 在 pose/depth 上略优；
- 但尺度似乎更差；
- 几何指标并没有同步提升；
- 因此 `near40` 是明显 trade-off。

### 6.2 本轮发现：原先的 `scale_err` 评判是错的

本轮最关键的新发现是：

- `scripts/batch_eval.py` 里用的 `scale_err = mean(|s-1|)` 是 legacy 口径；
- 这个定义并不适用于 OPV2V；
- 对 OPV2V 而言，GT 的尺度目标不是固定 `1`，而是每帧自己的 `g`；
- 因此真正应该比较的是 `s / g`，也就是**相对 GT 的尺度误差**。

换句话说：

- 之前“`near40` 尺度更差”的结论，实际上是被错误指标误导出来的。

---

## 7. 本轮完成的代码修复：修正 OPV2V 尺度评测口径

本轮对 `scripts/batch_eval.py` 做了针对性修复，目标是把尺度评测改成 OPV2V 可用的正确口径。

### 7.1 新增/修正的内容

1. 在 `EvalResult` 中补充了正确的尺度相关字段：
   - `scale_log_err`
   - `scale_eq_rel_err`
   - `scale_ratio_mean`
   - `scale_gt_factor`
   - `scale_to_gt_err`
   - `scale_to_gt_log_err`
   - `scale_to_gt_eq_rel_err`
   - `scale_to_gt_ratio_mean`
   - 以及对应 median / p90 等统计量

2. 增加从 GT views 估计 GT scale factor 的逻辑。

3. 增加从预测结果估计 predicted scale factor 的逻辑。

4. 新增 `scale_metric_details(...)` 以统一输出：
   - legacy 指标（保留兼容）
   - 正确的 `scale_to_gt_*` 指标

5. 在汇总 `summary_test.json` 与 `coop_metrics.csv` 时加入新的尺度统计字段。

### 7.2 语法验证

修复后已经运行：

```bash
/home/qqxluca/miniconda3/envs/mapanything_ft/bin/python -m py_compile scripts/batch_eval.py
```

结果通过。

---

## 8. 本轮完成的正式重跑与结果

### 8.1 修正口径后的正式重跑

修正后，对同一批 `20` 帧进行了 corrected scale rerun：

- 新评测目录：  
  `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/eval_bs160_epoch0_test20_seed42_scalefix`

期间一度出现 SSH 连接中断，但实际评测进程在远端继续跑完，最终生成：

- `summary_test.json`
- `base/coop_metrics.csv`
- `near40/coop_metrics.csv`

### 8.2 修正后的正式结论

修正口径后，`near40` 与 `base` 的关系变得更清晰：

#### `near40` 更好的部分

- pose 略优：
  - `pose_abs: 20.31 -> 20.06`
  - `pose_rot: 5.09 -> 4.83`

- depth 略优：
  - `depth_rmse: 16.51 -> 16.31`
  - `depth_mae: 9.35 -> 9.26`

- scale 也更优（这是本轮纠正的重点）：
  - `scale_to_gt_err: 0.8016 -> 0.7147`
  - `scale_to_gt_log_err: 1.7260 -> 1.3611`
  - `scale_to_gt_eq_rel_err: 5.5281 -> 3.5328`

#### `near40` 没有更好的部分

- 跨车几何并没有同步改善：
  - `chamfer_filtered_pred_to_gt: 1.569 -> 1.588`
  - `bev_iou_filtered: 0.2677 -> 0.2655`

#### 帧级统计

对 `20/20` 帧逐帧比较后发现：

- `scale_to_gt_err`：`near40` 在 `20/20` 帧更好；
- `scale_to_gt_log_err`：`near40` 在 `20/20` 帧更好；
- `scale_to_gt_eq_rel_err`：`near40` 在 `20/20` 帧更好。

### 8.3 因此得到的新结论

此前“`near40` 尺度更差”的判断是错误的。  
修正后，真实结论变为：

- `near40` 提升了 `pose / depth / scale`；
- 但没有带来更好的跨车几何融合。

这意味着：

- “减少跨车距离”确实能改善一部分优化问题；
- 但它不能单独解决“点云几何融合质量”这一最终问题。

---

## 9. 为什么“同车打同一标签”仍然不是当前最有效修复

围绕用户提出的“给同车图像打同一标签”的想法，本轮再次确认：

- 当前代码里的 `label` 主要用于 view naming / bookkeeping；
- 并不会自动变成“同标签更近、异标签更远”的几何监督；
- 因此单纯改标签，并不能直接解决多车几何对齐问题。

真正有效的手段仍然是：

1. 结构化采样；
2. `agent/camera identity embedding`；
3. pairwise / world-frame 几何损失；
4. overlap-aware / curriculum 数据分布设计；
5. 后续必要时再加 rig-consistency / baseline consistency 约束。

---

## 10. `near40` 距离截断的真实代价

本轮还整理了训练集里距离截断带来的统计变化：

- 原始同帧 `>=2 agents`：`6374`
- `<=40m` 后可用于 `2 agents x 4 cams` 结构化采样：`4690 / 6374 = 73.6%`
- 原始 `>=3 agents` 占比：`61.5%`
- `<=40m` 后 `>=3 agents` 占比：`25.5%`
- 最近邻车间距分位数：
  - `P50 = 27.1m`
  - `P75 = 41.2m`
  - `P90 = 59.6m`

这说明：

- `40m` 并没有把数据直接砍没；
- 但它显著削弱了多车场景中的“多主体丰富度”；
- 同时也减少了长基线尺度样本；
- 所以只训近邻会牺牲一部分真正困难但重要的多车几何情况。

---

## 11. 因此引出的下一步策略：`mix40`

为避免 `near40` 过于偏向近邻样本，本轮之前已进一步设计并启动了混合 curriculum：

- `4096 near40 + 4096 full-range`

目标是：

- 保留近距离样本带来的 overlap 优势；
- 同时保住 full-range 样本提供的尺度与长基线分布。

相关配置：

- `configs/dataset/opv2v_coop_ft_structured_stagea_mix40.yaml`

训练目录：

- `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/opv2v_coop_stagea_long_bs160_mix40_run1`

在本轮结束前查询到：

- 原先的 `mix40` 训练 PID `3104211` 已经不在了；
- 需要下一步继续检查它是正常结束、被系统杀掉，还是报错退出。

---

## 12. 文档与执行计划的同步更新

本轮不仅改了代码，也把文档结论全部同步纠正，避免后续继续被错误指标误导。

### 12.1 更新的远端文档

1. `docs/opv2v_multicar_execution_plan_20260311.md`
2. `docs/opv2v_multicar_fix_20260311.md`

主要更新内容：

- 明确说明此前 `scale_err` 是 legacy 口径；
- 将尺度主判断改为 `scale_to_gt_*`；
- 把 corrected rerun 的结果写入文档；
- 把已经完成的部分打勾；
- 将下一步待做事项改成基于正确尺度指标继续推进。

---

## 13. GitHub 提交与推送

本轮完成后，已将这次尺度修复相关改动提交并推送到 GitHub。

### 13.1 提交信息

- Commit: `55964bc`
- Message: `Fix OPV2V batch eval scale metrics`

### 13.2 推送目标

- 仓库：`insomniacs01/map`
- 分支：`opv2v-coop-multicar`

### 13.3 本次只提交了哪些文件

仅提交了本轮真正相关的 3 个文件：

1. `scripts/batch_eval.py`
2. `docs/opv2v_multicar_execution_plan_20260311.md`
3. `docs/opv2v_multicar_fix_20260311.md`

未把无关改动（如 `scripts/color_compare.py` 等）混入本次提交。

---

## 14. 本轮结束时的状态总结

### 14.1 已经确认的事实

1. 多车协同效果不好不是单一“车太远”问题，而是结构、身份、损失、分布和基础设施问题叠加。
2. “同车打标签”本身不会自动形成有效几何监督。
3. `near40` 的真实作用是：
   - 提升 `pose / depth / scale`
   - 但没有改善跨车几何融合
4. 之前基于 `scale_err = |s-1|` 得出的“尺度更差”结论是错误的。
5. OPV2V 的尺度评测现在已经改成正确的 `scale_to_gt_*` 口径。

### 14.2 当前尚未解决完的问题

1. 还没有最终证明哪条训练路线的点云几何最好；
2. 还没有拿到一版“完全没问题”的正式 checkpoint；
3. `mix40` 训练当前需要继续检查状态并续跑/分析；
4. 如果 `mix40` 之后仍然几何不好，需要继续补：
   - 跨车 baseline consistency
   - rig consistency
   - 更强的 curriculum
   - 更严格的点云可视化验收

---

## 15. 本轮对话的最终落点

本轮最重要的产出不是“又多跑了一次实验”，而是**把一个关键错误判断改正过来了**：

- 原先以为 `near40` 让尺度变差；
- 现在确认这其实是评测脚本口径错了；
- 修正后发现 `near40` 实际上改善了尺度；
- 真正没改善的是跨车几何融合。

这使得下一阶段的优化方向更加明确：

- 不要再把主要精力放在“修尺度”上；
- 应该把重点放在**跨车几何融合约束**与**训练分布设计**上。

---

## 16. 建议的下一步（承接本轮）

如果继续沿着本轮的结论往下推进，推荐顺序是：

1. 检查 `mix40` 训练为什么停止；
2. 修复并续跑 `mix40`；
3. 用固定帧集比较 `base / near40 / mix40`；
4. 重点看：
   - `scale_to_gt_*`
   - `chamfer_filtered_*`
   - `bev_iou_filtered`
   - 真实点云可视化
5. 如果 `mix40` 仍然几何不行，再补：
   - 跨车 baseline / rig-consistency loss
   - 更强 curriculum
   - 必要时更强的可视化与帧级诊断工具

---

## 17. 附：本轮涉及的关键路径

### 17.1 远端代码与文档

- `/home/qqxluca/map-anything3/scripts/batch_eval.py`
- `/home/qqxluca/map-anything3/docs/opv2v_multicar_execution_plan_20260311.md`
- `/home/qqxluca/map-anything3/docs/opv2v_multicar_fix_20260311.md`

### 17.2 远端评测结果

- 旧评测：  
  `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/eval_bs160_epoch0_test20_seed42/summary_test.json`

- 修正尺度后的新评测：  
  `/media/tsinghua3090/66c73fca-acad-4d88-a5b9-47aa246d1d02/qqxluca/map-anything3_experiments/eval_bs160_epoch0_test20_seed42_scalefix/summary_test.json`

### 17.3 Git 提交

- `55964bc` — `Fix OPV2V batch eval scale metrics`

---

以上即为本轮完整对话与工作过程的整理版。

---

## 18. 后续补充澄清（2026-03-12）：当前点云几何评测口径仍有关键定义问题

在继续远端排查 `scripts/batch_eval.py` 后，又确认了一个**不应忽视的评测定义问题**：

### 18.1 当前 `chamfer_filtered_* / bev_iou_filtered` 的公式本身不是主要问题

- `Chamfer` 当前实现使用的是标准的双向最近邻平均距离；
- `BEV IoU` 当前实现使用的是标准的俯视占据栅格交并比；
- 因此它不像此前 `scale_err = |s-1|` 那样，存在一眼可见的“公式定义直接错了”的问题。

换言之：

- 当前点云几何指标的主要风险**不在公式本身**；
- 而在于**它到底在拿什么 prediction 和什么 GT 做比较**。

### 18.2 当前 `single` 与 `coop` 模式下的 prediction 含义不同

经继续核对远端代码，当前评测脚本的输入组织方式是：

1. `single` 模式：
   - 只读取 `main_agent` 的 `4` 个相机视角；
   - 因此生成的 prediction 本质上是**主车单车点云**。

2. `coop` 模式：
   - 会读取 `info.coop_agents` 中所有车辆的相机视角；
   - 并且先把协同车视角都变换到 `main_agent` 坐标系；
   - 最终将多视角预测的 `pts3d` 聚合成一个 prediction point cloud；
   - 因此这里的 prediction 本质上是**多车协同融合后的点云**。

这意味着：

- `single` 下的 point cloud prediction 是“自车预测”；
- `coop` 下的 point cloud prediction 是“协同预测”；
- 二者在语义上并不是同一个评测对象。

### 18.3 当前 `GT` 只取主车 `.pcd`，不是协同真值点云

进一步确认到，当前 `batch_eval.py` 在计算点云几何指标时：

- 读取的 GT 路径是：`images_root / split / sequence / main_agent / frame.pcd`；
- 也就是**只取主车自己的 LiDAR 点云**；
- 并没有把 `coop_agents` 的 `.pcd` 一起对齐到主车坐标系后再融合。

因此当前 `coop` 模式下的点云几何指标，实际比较的是：

- **协同预测点云**  vs  **主车自车 GT 点云**

而不是：

- **协同预测点云**  vs  **协同真值点云**

### 18.4 这意味着当前 `coop` 点云指标不能直接作为“协同重建质量”的主结论

这一步非常关键。

如果当前 `coop prediction` 预测出了：

- 主车本车 LiDAR 没有扫到；
- 但协同车视角确实看到了；
- 且这些区域本来就是多车协同应该补出来的信息；

那么在现有评测口径里，这部分新增几何**可能会被当成多余点或错误点**，从而：

- 拉高 `chamfer_pred_to_gt / chamfer_filtered_pred_to_gt`；
- 压低 `bev_iou_raw / bev_iou_filtered`。

因此，当前 `coop` 模式下的 `chamfer_filtered_* / bev_iou_filtered` 更适合回答的是：

- **协同输入有没有把主车本地地图搞坏**

而不是直接回答：

- **协同输出是否更接近真实的多车联合场景**

所以本轮后续补充结论应明确改成：

- 当前 `coop` 点云几何指标**仍可作为辅助诊断**；
- 但它**不应该单独作为“协同点云质量是否真的更好/更差”的主依据**。

### 18.5 本轮补充讨论后确认：更关键的问题不在“GT 要不要过滤”，而在“GT 范围不对”

围绕点云评测又进一步讨论后，当前应优先明确的是：

- 相比“GT 是否也要做同样的高度/半径过滤”，
- **更关键的定义问题是：`coop` prediction 现在对应的 GT 仍只有主车 `.pcd`，并不是协同真值点云。**

也就是说：

- 即便先不改 GT 过滤口径；
- 只要 `coop` 还在和 `main_agent` 单车 `.pcd` 比；
- 那么它作为“协同重建主评测”的语义就依然是不完整的。

### 18.6 因此更合理的后续评测方案

后续更合理的做法应该是把点云几何评测拆成两套：

1. **主评测：`coop prediction` vs `fused coop GT`**
   - 读取所有 `coop_agents` 的 `.pcd`；
   - 使用各自 `lidar_pose` 对齐到 `main_agent` 坐标系；
   - 融合成真正的协同真值点云；
   - 再计算 Chamfer / BEV IoU。

2. **辅助评测：`coop prediction` vs `ego GT`**
   - 保留当前主车 `.pcd` 对照；
   - 用来回答“协同是否破坏主车本地地图”。

这样可以把两个问题分开：

- 一个指标回答“协同有没有补出真实多车几何”；
- 另一个指标回答“协同有没有把主车本地结果搞乱”。

### 18.7 这条补充澄清对当前工作方向的影响

这意味着，后续如果看到：

- `pose / depth / scale` 继续改善；
- 但当前 `coop` 的 `chamfer_filtered_* / bev_iou_filtered` 继续变差；

不能立刻下结论说：

- “模型真的把协同点云越训越坏了”。

更谨慎的说法应当是：

- **现有几何主评测口径可能在惩罚协同补出来、但主车单车 LiDAR 本来没有覆盖到的区域。**

因此，本轮后续追加的一个关键共识是：

- 当前多车协同点云效果的最终判断，必须在引入 `fused coop GT` 后再做更正式的确认。

## 19. 本轮补充：可视化链路修正与当前诊断状态（2026-03-12）

### 19.1 本轮新增确认：之前那两张彩色 3D HTML 不能作为有效结论

用户指出此前导出的彩色 3D 点云“肉眼看就完全不对”，这一判断是成立的。

本轮继续核对后确认：

- 之前那两张彩色 HTML 虽然能显示点云；
- 但**它们使用的点云生成/放置口径本身就不对**；
- 因此不能据此直接判断“模型几何真的坏到了什么程度”。

换言之，之前那批彩色 HTML 的问题，首先是：

- **可视化方法错了**；
- 而不是可以直接据此下结论说“模型本身已经被完全训坏”。

### 19.2 本轮确认的根因：之前可视化把 coop 点云放在了错误的参考系里

本轮重新梳理后确认，之前那批可视化主要用了类似下面这条错误思路：

- 直接使用 `predictions_to_pointcloud(...)` 产出的聚合点云；
- 再和 OPV2V 的 GT 点云放在一起看。

但在 OPV2V coop 场景里，这样做有两个关键问题：

1. `predictions_to_pointcloud(...)` 本质上是在用 `depth_z + camera_poses` 直接重建**模型自己的世界系点云**；
2. coop 下真正要看的，是**主车 ego/LiDAR 系**中的结果，而不是一个“模型自己内部 gauge/world frame”下的聚合点云。

因此之前那批图里，出现“点云像一团奇怪 blob / 位置和朝向都很怪”的现象，本质上是：

- **可视化参考系错了**；
- 所以看起来当然会很怪。

### 19.3 本轮进一步确认：OPV2V coop 点云必须拆成 `PredPose` / `GTPose` 两列看

这一点本轮也重新核实清楚了。

对于协同点云，必须把问题拆成两部分：

1. **PredPose**
   - 用模型自己预测的 pose 放置点云；
   - 回答的是：模型最终“部署口径”下输出长什么样。

2. **GTPose**
   - 用 GT 相机 pose 去放置 `pred['pts3d_cam']`；
   - 回答的是：如果不看 pose，只看几何本体（depth/ray/scale/mask），点云形态是否合理。

这样才能区分：

- 到底是 **pose 出问题**；
- 还是 **geometry 本身就有问题**。

如果不拆这两列，单看一张混在一起的图，很容易把两类错误混淆。

### 19.4 本轮继续确认：GT 必须使用 `fused coop GT`，而不是主车单车 GT

这和前面第 18 节结论保持一致，但这轮在可视化链路里又被重新落实了一次。

本轮修正后的可视化明确使用：

- **主车 ego 系中的 fused coop GT LiDAR**；
- 而不是只取 `main_agent/frame.pcd` 的 ego-only GT。

具体做法是：

- 读取所有参与 coop 的 agent 的 `.pcd`；
- 用各自 `lidar_pose` 对齐到 `main_agent` 坐标系；
- 再融合成真正的协同真值点云。

因此这轮新导出的 HTML，在 GT 定义上已经比旧版正确得多。

### 19.5 本轮新增产物：已新写一份“正确口径”的 coop RGB 可视化脚本

本轮新建脚本：

- `scripts/viz_opv2v_coop_rgb_predpose_gtpose_html.py`

这个脚本的设计目标是专门解决本轮暴露出来的问题：

- 使用 **coop raw input**；
- 生成 **PredPose / GTPose** 两列；
- 使用 **fused coop GT**；
- 做 **OpenCV -> CARLA** 坐标统一；
- 保留 **RGB 上色**；
- 输出可旋转的 Plotly 3D HTML。

额外处理：

- 远端机器上 DINOv2 的 `torch.hub` 访问不稳定；
- 该脚本里额外补了一个本地 hub repo 定位逻辑，用远端缓存目录：
  - `/home/qqxluca/.cache/torch/hub/facebookresearch_dinov2_main`
- 因此下一轮如果需要继续在远端重跑这个脚本，建议继续沿用这份脚本，不要退回旧版错误流程。

### 19.6 本轮已生成的“正确版” HTML 文件

远端生成路径：

- `/home/qqxluca/map-anything3/outputs/vis_fix/2021_08_23_16_06_26_000079_predpose_gtpose_rgb.html`
- `/home/qqxluca/map-anything3/outputs/vis_fix/2021_08_24_20_09_18_000132_predpose_gtpose_rgb.html`

已同步回本地：

- `tmp_vis_html_fixed/2021_08_23_16_06_26_000079_predpose_gtpose_rgb.html`
- `tmp_vis_html_fixed/2021_08_24_20_09_18_000132_predpose_gtpose_rgb.html`
- 索引页：`tmp_vis_html_fixed/index.html`

### 19.7 本轮关于“页面打不开”的最终结论

本轮也顺手把页面打开方式问题彻底确认了一次：

- **不是端口转发问题**；
- **不是远端页面没拉回来**；
- 主要是 Safari 对这类大 Plotly HTML 在 `file://` 直开时经常出现白屏/空白。

因此目前稳定打开方式是：

1. 在本地目录起服务：
   - `cd /Users/macbookpro/repos/map/tmp_vis_html_fixed && python3 -m http.server 8001`
2. 浏览器打开：
   - `http://127.0.0.1:8001/index.html`

这轮已经实际这样做过，并确认 `http://127.0.0.1:8001/...` 路径是可打开的。

### 19.8 本轮从“正确版可视化标题摘要”里读到的初步信号

虽然这一轮的重点是把可视化链路先纠正，但在新导出的 HTML 标题里，已经能看到一组很重要的相对 pose 摘要：

#### 帧 A：`2021_08_23_16_06_26 / 000079`

主车 `243` 的四个相机：

- 相对 pose 误差基本都接近 `0m / 0~1°`

协同车 `252`：

- 大约在 `1.6m ~ 4.0m`
- 角度大约在 `0.1° ~ 2.0°`

协同车 `261`：

- 大约在 `1.8m ~ 5.1m`
- 角度大约在 `0.6° ~ 2.9°`

#### 帧 B：`2021_08_24_20_09_18 / 000132`

主车 `3174` 的四个相机：

- 基本仍接近 `0m / 0~1°`

协同车 `3183`：

- 有一部分视角误差较小（约 `0.6m ~ 0.9m`）；
- 但也出现了更明显的跨车 pose 偏差，最大大约到 `2.7m / 4.8°`。

### 19.9 因而本轮的阶段性判断：当前更像是“跨车 pose 漂移”在主导问题

基于目前这轮已经确认的内容，更合理的阶段性判断是：

- 旧图之所以“看起来离谱”，首先是因为**可视化链路本身错了**；
- 把可视化口径修正后，当前能明确看到的一个主要问题是：
  - **跨车相对 pose 误差确实存在，而且量级已经足以把协同点云整体拉歪。**

也就是说，现在更像是：

- **主车内四个视角相对稳定；**
- **协同车对主车的相对 pose 仍有几米级偏差；**
- 这会直接导致 `PredPose` 那一列看起来错位、糊成一团、重叠异常。

### 19.10 但本轮仍未最终排除：geometry 本身可能也还有问题

虽然当前已经能较明确地怀疑“pose 是大头”，但这轮还不能把 geometry 完全宣告无罪。

原因是：

- 目前虽然已经修正到 `PredPose / GTPose / fused GT` 的正确可视化口径；
- 但还没有把这两个页面再做成：
  - 按 agent 分色；
  - 或者按 `PredPose vs GTPose` 做更细粒度的数值比较（例如只看 `GTPose` 下的 Chamfer / BEV IoU）。

所以更准确的说法应当是：

- **目前已经确认：旧可视化流程错了；**
- **修正后初步显示：跨车 pose 漂移是当前主要可疑点；**
- **但 geometry 是否也同时存在明显缺陷，还需要继续拆分检查。**

### 19.11 本轮已经完成什么、下一轮最该做什么

本轮已经完成：

1. 证实旧彩色 HTML 可视化口径有问题；
2. 重新梳理出 coop 场景下正确的点云可视化定义；
3. 新写正确版脚本：`PredPose / GTPose / fused GT / RGB`；
4. 远端成功生成两帧 corrected HTML；
5. 已同步回本地并提供稳定打开方式（本地 HTTP）。

下一轮最建议继续做的事：

1. **人工直接看 corrected HTML**
   - 重点看：左列 `PredPose` 和右列 `GTPose` 差多少；
   - 如果右列明显比左列好很多，那么基本就能锁定：**主问题是 pose**。

2. **再做一版 agent 分色 HTML**
   - 不再按 RGB 上色；
   - 改成每辆车一个颜色；
   - 这样能一眼看出到底是哪台协同车漂得最厉害。

3. **如果要进入定量阶段**
   - 在 `GTPose` 口径下补几何指标；
   - 这样可以把“geometry 质量”从“pose 误差”里进一步剥离出来。

### 19.12 给下一轮对话的直接接手提示

如果下一轮对话要继续，不建议再从旧的 `tmp_vis_html/` 那批文件开始看；
建议直接从这批 corrected 页面开始：

- `tmp_vis_html_fixed/index.html`
- 本地打开方式：`http://127.0.0.1:8001/index.html`

并优先围绕下面这个问题继续：

- **在 corrected 可视化里，`GTPose` 是否明显优于 `PredPose`？**

如果答案是“明显优于”，那么后续工作重点应切到：

- **coop 跨车 pose 对齐问题**

如果答案是“`GTPose` 依然很差”，那后续才应该继续重点查：

- **geometry/depth/ray/scale/mask 本身的问题**

