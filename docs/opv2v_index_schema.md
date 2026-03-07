# OPV2V Index Schema（全量索引规范）

> 目标：只扫描一次 OPV2V YAML，后续训练/筛选**不再扫数据**，任意距离阈值可快速过滤。

## 位置

- 建议放在 OPV2V 根目录：`<OPV2V_ROOT>/opv2v_index/`
- 文件：
  - `index_train.parquet`
  - `index_validate.parquet`
  - `index_test.parquet`
  - `schema.json`
  - `manifest.json`

## Schema（字段定义）

**版本**：`opv2v_index_v1`

每行代表一个 `scene+frame` 的“协同快照”，字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `split` | string | `train/validate/test` |
| `sequence` | string | 场景序列名 |
| `frame` | string | 帧编号（五位数字字符串） |
| `agents` | list[string] | 本帧参与的 agent id（排序后） |
| `main_agent` | string | 默认主车（`sorted(agents)[0]`） |
| `pair_agent` | string/null | 最近的可用配对车（基于资产完整性） |
| `num_agents` | int | 本帧有效 agent 数 |
| `num_agents_with_assets` | int | 有完整图像/深度的 agent 数 |
| `agent_pose` | dict | `agent_id -> [x,y,z,roll,yaw,pitch]` |
| `agent_xy` | dict | `agent_id -> [x,y]` |
| `agent_has_assets` | dict | `agent_id -> bool` |
| `agent_num_boxes` | dict | `agent_id -> int`（车辆框数量） |
| `agent_has_boxes` | dict | `agent_id -> bool` |

> 说明：`agent_pose` 是“世界坐标系下的 LiDAR pose”，允许后续按任意阈值计算相对距离。

## 为什么这样设计

- **动态阈值筛选**：距离阈值只是一个过滤条件，不需要重新扫 YAML。  
- **跨数据集复用**：其他协同数据集只需提供相同字段即可复用筛选逻辑。

## 推荐筛选逻辑（示意）

1) 读索引  
2) 对每行取 `main_agent` 的 `agent_xy`  
3) 计算其它 agent 到 `main_agent` 的距离  
4) 保留距离在 `max_agent_distance` 内且 `agent_has_assets==True` 的 agent  
5) 若满足 `min_agents` / `num_views` → 该帧可用

## 迁移到其它协同数据集

只要你能为每帧构造同样字段（尤其是 pose/资产可用性），这套索引逻辑即可复用，不依赖 OPV2V 原始结构。

