# OPV2V Scene Cache（多车筛选缓存）

> 目的：避免每次启动都扫描 + 解析大量 YAML（多车距离筛选），尤其在 NAS/IO 慢环境下会卡 10–60 分钟。

## 当前缓存位置（已统一）

- 持久缓存目录（NAS）：`/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/cache/opv2v_scene_cache`
- 本地加速缓存（/tmp）：`/tmp/mapanything_cache`
- 环境变量（可选覆盖）：
  - `MAPANYTHING_CACHE_DIR=/path/to/cache`（持久化主缓存）
  - `MAPANYTHING_FAST_CACHE_DIR=/tmp/mapanything_cache`（本地加速缓存）
  - `MAPANYTHING_META_CACHE_DIR=/path/to/meta_cache`（可选：缓存 YAML 元数据，减少每次 __getitem__ 的 YAML IO）

**说明**：缓存文件名包含参数哈希（split + 视角/距离/筛选策略等）。只要这些参数不变，后续训练会自动复用缓存。代码会：
1) 优先读取本地 `/tmp` 缓存（更快）  
2) 若 /tmp 不存在，则读取 NAS 并自动同步一份到 /tmp  
3) 若需要重建缓存，会在 **NAS 与 /tmp** 同时写入一份
4) 若设置 `MAPANYTHING_META_CACHE_DIR`，会缓存 YAML 元数据（lidar_pose、相机外参/内参、车辆信息等），后续训练直接读取 meta cache，避免每个样本反复读 YAML

## 给“另一台服务器/另一个 AI”的 Prompt（可直接复制）

> 目标：在 IO 快的机器上**预生成** OPV2V coop 场景缓存，然后把缓存目录同步到 NAS，之后所有训练节点直接复用，启动即可跳过长时间 YAML 扫描。

```
你是一个自动化运维助手。请在机器上预生成 OPV2V 多车协同数据的 scene 缓存，用于 MapAnything 的 coop 训练加速。

要求：
1) 缓存目录统一放在：
   /J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/cache/opv2v_scene_cache
2) 用 MAPANYTHING_CACHE_DIR 指向该目录；
3) 需要生成 split=train/validate/test 的缓存；
4) 参数必须与训练完全一致（否则 hash 会变化导致不命中）；
5) 建议同时生成多种 max_agent_distance（比如 30/40/60m）以便后续做 curriculum 或 ablation。

请执行以下脚本（可直接复制）：

cd /J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything
export PYTHONPATH=$(pwd)
export MAPANYTHING_CACHE_DIR=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/cache/opv2v_scene_cache
export MAPANYTHING_FAST_CACHE_DIR=/tmp/mapanything_cache
export MAPANYTHING_META_CACHE_DIR=/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/cache/opv2v_meta_cache

python - <<'PY'
from pathlib import Path
from mapanything.datasets.opv2v import OPV2VCoopDataset

ROOT = "/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/data/opv2v_images"
DEPTH = "/J6P-perception/yijinxiong_workspace/vggt_series_4_coop/map-anything/data/opv2v_depth"

base_kwargs = dict(
    ROOT=ROOT,
    depth_root=DEPTH,
    camera_ids=(0,1,2,3),
    pair_agents=True,
    pair_agent_policy="nearest",
    include_agents=None,
    main_agent=None,
    main_agent_policy="first",
    min_agents=2,
    min_num_views=8,
    max_num_views=8,
    num_views=8,
    variable_num_views=False,
    resolution=(448,252),
    principal_point_centered=False,
    transform="colorjitter+grayscale+gaublur",
    data_norm_type="dinov2",
    aug_crop=16,
    seed=777,
    max_num_retries=1,
)

for max_dist in (30, 40, 60):
    for split in ("train", "validate", "test"):
        ds = OPV2VCoopDataset(
            **base_kwargs,
            split=split,
            max_agent_distance=max_dist,
        )
        print(f"[OK] split={split} max_dist={max_dist} scenes={len(ds)}")
PY
```

产出：`opv2v_coop_<split>_<hash>.json` 若存在即说明缓存成功。  
然后把整个目录同步给其他训练机即可复用（rsync/scp 均可）。

## 常见问题

- **参数改了但不生效？**  
  缓存 key 由配置决定（分辨率、视角数、pair_agents、max_agent_distance 等）。一旦变动会生成新 hash，不会命中旧缓存。
- **缓存在哪被使用？**  
  `mapanything/datasets/opv2v.py` 内部会自动读取 `MAPANYTHING_CACHE_DIR`（持久化）并优先使用 `MAPANYTHING_FAST_CACHE_DIR`（本地加速）。
- **如何确认命中？**  
  训练日志里会在 dataset 初始化时显示“Building train dataset …”，但若 cache 已命中，会跳过长时间 YAML 扫描（速度明显快）。
- **为什么“帧索引”不够？**  
  因为是否可用取决于**每帧每车的 pose 距离和可视化数量**，这些信息在 YAML 里。缓存保存的是**已经过滤后的 scene 列表**，用来避免每次都扫描全部 YAML 并计算距离。
