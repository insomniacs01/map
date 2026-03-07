# 协同场景 Baseline Distance 分桶协议（训练 & 评测）

Last updated: 2026-03-03

本文目标：把“跨车距离跨度大导致几何/尺度收敛差”的问题，落成一套 **可复现、可迁移到新数据集** 的分桶与评测规范，使得：
- 同一模型/不同模型在 **严格公平合同** 下可比；
- bucket breakdown 作为诊断与回归测试常态化；
- bucket 既能服务 **训练（课程学习/重加权/分桶 finetune）**，也能服务 **评测（主合同 + 压力合同 + 桶回归）**。

> 本协议只定义 *bucket 与合同*，不重新定义 scale/pose/depth/det 指标口径：指标解释以 `scale_metric_notes.md` 和 `goal_based_test_plan.md` 为准。

---

## 0. 核心结论（为什么要分桶）

在 OPV2V/多车协同里，跨车 baseline 距离存在重尾（tail 很少但很难），并且在 `norm_mode=avg_dis` 的定义下会强烈影响 coop 的 GT scale 因子与 pose 误差：
- 如果不控制 bucket 覆盖，随机小合同会 **欠采样 tail**，导致“看起来变好/变坏”不稳定。
- 如果训练时把不同 baseline 难度混在一起，模型往往会被 tail 的梯度/数值不稳定拖垮（尤其是 scale/pose）。

证据与背景详见：
- `goal_based_test_plan.md`（Test-set design review + det 子集方差）
- `../../docs/coop_scale_strategy_options_eval.md`（baseline distance 与 g_coop/误差强相关的量化）

---

## 1. 分桶对象：baseline distance（跨车距离）

### 1.1 baseline distance 的定义（跨数据集通用）

对每个多车样本（同一 timestamp 的多车多视角），选择一个 **main agent**（参考车），其余协同车为 **coop agents**。

定义每个协同车到主车的水平距离（推荐）：
- `d_i = || (x_i, y_i) - (x_main, y_main) ||_2`（单位：米）

当一次样本包含多个协同车（>2 agents）时，建议用 **最难 baseline** 作为该样本的 bucket key：
- `baseline_distance_m = max_i d_i`

这样 bucket 能反映“本帧联合重建/融合里最远的跨车约束”，对尺度/跨车 pose 的稳定性更敏感。

> OPV2V 当前 pinhole 2-agent/8-view 主线里，`baseline_distance_m` 退化为 main 与 pair agent 的距离。

### 1.2 bucket edges（默认）

默认采用 5 桶（细分 near）：
- B0: `[0, 15)`
- B1: `[15, 30)`
- B2: `[30, 60)`
- B3: `[60, 100)`
- B4: `[100, +inf)`

对应的标准 edges（米）：
- `bucket_edges_m = [0, 15, 30, 60, 100, 1e9]`

训练时可用 4 桶粗粒度（便于样本量足够）：
- near: `[0, 30)`
- mid: `[30, 60)`
- far: `[60, 100)`
- tail: `[100, +inf)`

对应的**标准 preset 名称**（跨脚本统一）：
- `fine_eval`：`[0, 15, 30, 60, 100, inf]`
- `coarse_train`：`[0, 30, 60, 100, inf]`

相关工具对齐：
- `scripts/bucket_frames_by_baseline_distance.py --preset {fine_eval|coarse_train}`
- `scripts/report_index_bucket_stats.py --preset {fine_eval|coarse_train}`
- `scripts/make_frames_contract_from_index.py --bucket_preset {fine_eval|coarse_train}`

---

## 2. 统一“合同（frames contract）”规范：可复现 + 可迁移

### 2.1 统一 JSON schema（跨数据集）

**contract 文件**是所有公平对比的起点：任何模型对比都必须在同一份 contract 上跑。

推荐 schema（必要字段 + 可选字段）：
```json
{
  "split": "test",
  "seed": 42,
  "sample_size": 500,
  "pair_policy": "index_nearest",
  "require_assets": true,
  "bucket_edges_m": [0, 15, 30, 60, 100, 1000000000],
  "bucket_target_counts": [150, 200, 103, 18, 29],
  "frames_hash_md5_v2": "...",
  "frames": [
    {
      "sequence": "...",
      "frame": "000088",
      "main_agent": "1045",
      "coop_agents": ["1045", "1054"],
      "baseline_distance_m": 49.57,
      "baseline_bucket": "B2_[30,60)"
    }
  ]
}
```

关键要求：
- **baseline_distance_m**：强烈建议写进 contract（这样后续分桶不依赖数据集文件，跨数据集通用）。
- **frames_hash_md5_v2**：必须记录（hash 包含 main/coop pairing，避免 pairing 漂移造成“伪公平”）。

OPV2V 已有可直接复用的 contract 生成器：
- `scripts/make_opv2v_frames_contract.py`

### 2.2 bucket_report（从 contract 派生的切桶产物）

给定一个 contract，我们会生成 `bucket_report.json` + 每个桶的子 contract（仍然是同一 schema 的子集）。

工具：
- `scripts/bucket_frames_by_baseline_distance.py`

说明：
- 如果 contract 已包含 `baseline_distance_m`，脚本会直接使用，不会读取 OPV2V 的 YAML。
- 若缺失，才会回退到 OPV2V YAML 计算（仅作为兼容旧合同）。

---

## 3. “主合同 vs 压力合同 vs 桶回归”：每个桶在评测中的作用

### 3.1 评测必须同时覆盖两种分布

为了既“代表真实部署”又“不会漏掉 tail 回归”，建议固定两份 test 合同：

1) **C-main（代表性主合同）**
- pairing policy 与训练一致（通常 `index_nearest`）
- bucket 按真实分布出现（不强行平衡）

2) **C-stress（压力合同 / bucket 覆盖合同）**
- 仍然同一 pairing policy（例如 `index_nearest`）
- 显式指定 `bucket_target_counts`，保证 B3/B4 有足够帧数

OPV2V 已有现成合同（推荐）：
- 代表性：
  - `eval_runs/frames_test500_nearest_seed42.json`
  - `eval_runs/frames_test200_nearest_seed42.json`（更省）
- 压力/覆盖：
  - `eval_runs/frames_test500_nearest_stress_seed42.json`
  - `eval_runs/frames_test200_nearest_stress_seed42.json`

大合同层级（用于 promotion/final）：
- `eval_runs/frames_test2170_nearest_seed42_full.json`：**final decision set（默认也作为 promotion 口径）**  
  - OPV2V test split 只有 2170 帧，full 的成本通常可接受；优先用 full 得到稳定结论。
- 可选中档 gate（仅当 full2170 成本过高时才需要）：例如 `eval_runs/frames_test1000_nearest_stress_seed42.json`（Test1000-stress），或自行生成 Test1500-stress（更有“省时意义”）。
- `eval_runs/frames_test2000_nearest_seed42.json` / `eval_runs/frames_test2000_nearest_stress_seed42.json`：**near-full proxy**  
  - 2000 与 2170 区分度很低（92% overlap + bucket shift 极小），不建议作为独立决策 gate。

补充（为什么你会觉得 “2000stress 不像 stress”）：  
stress 的“压测效果”会随着 `N → full` 迅速变弱，因为 far/tail 桶的可用帧数本来就很少（存在上限）。
- `Test500-stress` vs `full2170`：bucket ratio shift 最大约 `3.225pp`（差异明显）
- `Test2000-stress` vs `full2170`：bucket ratio shift 最大约 `0.353pp`（差异很小）

### 3.2 桶回归（bucket breakdown）怎么用

对每次 eval（同一 contract 的一次推理）都应输出：
- overall（主结论）
- per-bucket breakdown（诊断 + 回归）

**注意**：
- det 的 bucket AP 仅用于诊断/回归（不要简单平均 bucket AP 作为 full AP 的替代）。
- 若你需要“桶等权”的宏平均指标，应当定义一个 **桶等权采样的 contract**（即 C-stress），再在该 contract 的 overall 指标上做主结论。

### 3.3 合同区分度检查（必须落盘）

任何 quick/promote/final 合同组合都应先做区分度量化并归档：

```bash
PYTHONPATH=$(pwd) python scripts/report_contract_distinguishability.py \
  --contract_a eval_runs/frames_test2000_nearest_seed42.json \
  --contract_b eval_runs/frames_test2170_nearest_seed42_full.json \
  --label_a test2000_main \
  --label_b full2170 \
  --out_json eval_runs/contract_distinguish_test2000_main_vs_full2170.json \
  --out_md eval_runs/contract_distinguish_test2000_main_vs_full2170.md
```

最低检查项：
- overlap：`intersection/union/Jaccard` + subset 关系；
- baseline buckets（`[0,15,30,60,100,inf]`）的 count/ratio shift；
- 若 quick 合同是 full 的子集，且桶比例差异很小，则该 quick 合同只能算“提速代理集”，不能作为独立决策 gate。

### 3.4 如何构造“真正不同”的 quick/promotion 集

1) **默认推荐：stress quick + full final**  
   quick 用 stress（如 Test500-stress）做 tail 回归，promotion/final 一律看 full2170。

2) **需要严格分离时：用 disjoint bucket 子合同**  
   先从 full 合同切桶（子合同天然不重叠），再选不同桶组做 quick 与 promotion 诊断：
   - `scripts/bucket_frames_by_baseline_distance.py --preset fine_eval --frames_json <full.json> ...`
   - 用 `report_contract_distinguishability.py` 验证 overlap 与 bucket shift。

---

## 4. 训练如何使用 bucket（3D 基础模型收敛导向）

典型用法（从便宜到昂贵）：

1) **Bucket-only finetune**：只在某个桶（例如 tail）做短步微调，观察 tail 是否改善且 main 不退化。
2) **Curriculum**：near -> mid -> far -> tail 逐步加难，防止早期被 tail 拖垮。
3) **Mixture reweight**：全量训练但提高 tail 的采样权重/scale loss 权重（只改一个因素更可控）。

实现落地点（OPV2V 已支持）：
- 数据集支持 `min_agent_distance/max_agent_distance` 过滤（只约束 *跨车*，主车永远保留）：
  - `mapanything/datasets/opv2v.py`
  - 入口配置：`configs/dataset/opv2v_coop_ft_2a8v_full.yaml`
- **Contract-driven 分桶训练**（推荐用于“严格按 bucket key 划分训练集/回归集”，避免 min/max 距离过滤改变 pairing 语义）：  
  - `OPV2VCoopDataset(..., frames_json=...)` 支持直接读取 frames contract 并锁定 `main/pair`（用于可复现的桶训练/桶回归）。  
  - 典型流程：先生成 full-train contract，再切桶得到 train buckets（near/mid/far/tail），分别做 finetune / curriculum：  
    - full train contract：`eval_runs/frames_train6374_nearest_seed42_full.json`  
    - 切桶（推荐用 coarse preset，直接得到 4 桶 near/mid/far/tail）：  
      - `scripts/bucket_frames_by_baseline_distance.py --preset coarse_train --frames_json <train_full.json> --out_dir <bucket_dir> --self_check`  
      - 输出文件名会带 lane：`..._near_0_30m.json / ..._mid_30_60m.json / ..._far_60_100m.json / ..._tail_100_inf_m.json`  
    - 训练时把 `frames_json=<bucket_contract.json>` 写进 dataset config（Hydra dataset_str 里加 `frames_json=...`）。
- 分桶训练矩阵脚本（4 桶 coarse）：
  - `scripts/run_plan1_distance_bucket_matrix.sh`

---

## 5. 标准评测流程（严格公平可比）

### 5.1 Geometry（Test500/200）

1) 选定 contract（C-main 与 C-stress 都跑）：
- `eval_runs/frames_test500_nearest_seed42.json`
- `eval_runs/frames_test500_nearest_stress_seed42.json`

2) 用同一协议跑 `batch_eval.py`（部署口径建议 `model_task=calibrated_sfm`；不喂跨车 GT pose）：
- 具体命令/阈值以 `goal_based_test_plan.md` 为准。

3) 对同一份 eval 输出做 bucket breakdown（no inference）：
- `scripts/bucket_frames_by_baseline_distance.py` 生成 bucket 子合同 + `bucket_report.json`
- `scripts/report_geom_by_baseline_buckets.py` 聚合出 per-bucket 表格
  - 注：为避免同一 `sequence/frame` 在不同 main/pair 下产生歧义，推荐使用带 `main_agent/coop_agents` 列的 per-frame CSV（2026-02-28+ 的 `batch_eval.py` 默认写出）。

一键入口（推荐）：
- 单合同（eval + buckets）：
  - `scripts/eval_geom_ckpt_contract_with_buckets.sh`
- 双合同套件（默认 nearest + nearest_stress）：
  - `scripts/eval_geom_ckpt_bucket_suite.sh`

### 5.2 Detection（Test50/200/500）

det 的公平性更依赖 **固定 decode cfg** 与 **固定 frames contract**：
- decode cfg 锁定逻辑见 `goal_based_test_plan.md` 与 `scripts/eval_det_ckpt_test50.sh`
- 需要评估 sampling variance 时，用：
  - `scripts/report_det_ap_subset_sensitivity.py`

---

## 6. 新数据集如何接入这套协议（你需要提供什么）

为了让“同一套 bucket 协议”可跨数据集复用，新数据集只需要做到：

1) 能生成统一 schema 的 contract JSON（推荐实现一个 `make_<dataset>_frames_contract.py`）；
2) 每个 frame 写入 `baseline_distance_m`（以及可选的 `baseline_bucket`）。

之后可以直接复用本仓库的通用工具链：
- `bucket_frames_by_baseline_distance.py`（自动优先使用 contract 的 baseline_distance_m）
- `batch_eval.py`（输入 `--frames_json` 即可）
- `report_geom_by_baseline_buckets.py` / `report_det_ap_by_baseline_buckets.py`

如果新数据集没有全局坐标/无法得到 baseline distance（例如跨车外参完全未知且无 GT），则需要先定义一个可计算的 proxy（例如 GPS/里程计/伪 GT），否则“按 baseline 难度”分桶无法落地。

---

## 7. Fairness Checklist（每次对比都必须写在报告里）

- contract 路径 + frames 数
- `frames_hash_md5_v2`（若合约 frames 里带 `dataset` 字段，建议同时记录 `frames_hash_md5_v3`）
- pairing policy（nearest/random/…）
- bucket_edges_m（以及是否 stress/balanced）
- eval 协议：`model_task` + `keep_camera_poses/keep_main_agent_poses`
- det：`det_decode_cfg`（iou_thresh/score_thresh/nms/max_dets/网格范围/voxel_size）
