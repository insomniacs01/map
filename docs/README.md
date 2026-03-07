# map-anything/docs 导航（精简版）

> 本仓库目前有两条“需要维护”的主线：
> 1) **OPV2V pinhole 几何→检测**（VGGT/MapAnything 基础几何：pose/depth/scale + BEV det/e2e）
> 2) **圆柱 4n→n 聚合路线**（cyl/）
>
> 上游历史 OPV2V 评测/训练/可视化文档放在 `archive/upstream/`，仅供参考（数字可能过期）。

## 主线 A：OPV2V pinhole 几何→检测（需要维护）
- `goal_based_test_plan.md`：**验收标准 + 固定合同 + PASS/FAIL 状态**（最权威）
- `scale_metric_notes.md`：scale 指标口径说明（避免 `|s-1|` 误导 OPV2V）
- `coop_bucket_protocol.md`：baseline distance 分桶协议（主合同/压力合同/桶回归；训练&评测统一）
- `coop_contract_index_schema.md`：跨数据集通用的 index parquet schema（先 index，再 contract；支持灵活分桶训练/评测）
- `../../docs/coop_eval_standard.md`：评测标准化总纲（合同/分桶/产物索引；包含主线 ckpt bucket-suite 对比入口）
- `pose_depth_scale_execution_plan.md`：详细执行笔记（偏历史；以 test plan 为准）
- `requirements_and_issues_overview.md`：需求/问题综述（用于对齐问题空间）
- `opv2v_scene_cache.md`：OPV2V 多车 YAML 预筛选缓存（避免每次启动都扫 YAML）
- `opv2v_eval_gallery_howto.md`：Metric Viz / Det Compose 画廊的**阅读方式 + 一键重跑命令**（可视化闭环入口）
- `experiment_logbook.md`：训练/排障 logbook（要求每轮实验落盘证据）

## 主线 B：圆柱 4n→n（需要维护）
- `cyl/multi_car_training_plan.md`：圆柱全景聚合方案与实现细节。
- `cyl/opv2v_cyl_coop_debug_plan.md`：圆柱训练调试 checklist（环境/步数/遮挡 mask 等）。
- `cyl/opv2v_cyl_coop_pitfalls.md`：圆柱协同（重建/姿态）踩坑大全（GT 基准、坐标系/参考帧、指标与可视化对齐）。
- `cyl/opv2v_cyl_lidar_geom_finetune.md`：LiDAR 投影监督的几何微调打法（从“可视化全错”到“点云重合”）。
- `cyl/opv2v_cyl_detection_pitfalls.md`：检测头微调踩坑大全（坐标系/pose key/解码网格/阈值与可视化错觉）。
- `cyl/results_overview.md`：当前各方案的指标与可视化索引（便于横向对比/快速打开产物）。
- `codex_debug_workflow.md`：失败自动生成 Codex 调试 prompt 的流程与管理规范。
- `assets/`：配套图示（若仅服务上游文档，可后续迁入 `archive/`）。

## 归档
- `archive/upstream/`：上游 OPV2V 评测/训练/可视化文档备份（image-only、Vehicle-ID、可视化等，数字可能过期）。
- `archive/` 其它文件：旧报告/流程，慎用厘米级数字。

新增或更新圆柱方案时，请同步维护 `cyl/`；上游通用流程建议参考 map-anything 分支，以免双份漂移。
