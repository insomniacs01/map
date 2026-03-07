# OPV2V 点云可视化操作指南

> 适用于在 SSH/VS Code Remote-SSH 场景下快速查看 `.pcd`/Plotly HTML 点云。无需 GPU 渲染窗口，直接借助浏览器即可交互式旋转、缩放。

## 1. 背景原理
- `scripts/color_compare.py` 以及批量评估脚本会在 `map-anything/visualization_outputs/` 下输出预测 `.pcd`。为了绕过 X11/OpenGL 依赖，我们将这些 `.pcd` 转换为 Plotly 交互式 HTML（位于 `visualization_outputs/html/*.html`）。  
- Plotly 仅依赖 WebGL，由浏览器负责渲染，因此只要把 HTML 文件通过 HTTP 服务暴露给本地浏览器即可查看。  
- VS Code Remote-SSH 支持端口转发：在服务器上运行 `python -m http.server` 后，VS Code 会自动提示端口映射，把远端 `8090` 转成本地 `127.0.0.1:<forwarded_port>`，本地浏览器即可访问。

## 2. 操作步骤
1. 打开 VS Code 终端（确保已在仓库中）：
   ```bash
   cd /home/qqxluca/vggt_series_4_coop/map-anything/visualization_outputs/html
   python -m http.server 8090
   ```
   - 终端会输出 `Serving HTTP on 0.0.0.0 port 8090 ...`。
   - VS Code 右下角会弹出 “Forwarded port 8090” 提示，点击即可复制本地访问地址（通常是 `http://127.0.0.1:8090/`）。

2. 在本地浏览器中访问对应地址，例如：
   ```
   http://127.0.0.1:8090/stage2_coop_000069_pred.html
   ```
   即可看到“预测（红色） vs GT（灰色）”的 3D 点云，对照不同模型/模式只需切换文件名。

3. 查看完成后，回到 VS Code 终端按 `Ctrl+C` 终止 `http.server`。

## 3. 常用 HTML 文件
| 文件名 | 内容说明 |
| --- | --- |
| `pretrain_stage_single_000069_pred.html` | 预训练模型（单车）与 GT |
| `stage1_000069_pred.html` | Stage1 单车与 GT |
| `stage2_000069_pred.html` | Stage2 单车与 GT |
| `pretrain_stage_coop_000069_pred.html` | 预训练模型（协同输入）与 GT |
| `stage1_coop_000069_pred.html` | Stage1 协同输入与 GT |
| `stage2_coop_000069_pred.html` | Stage2 协同输入与 GT |

若需要新的帧或模型，可复用下述思路：读取 `.pcd` → 用 Plotly 生成 HTML。示例（可直接放入 `python - <<'PY'` 中执行）：
```python
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

def load(path):
    cloud = o3d.io.read_point_cloud(path)
    return np.asarray(cloud.points), np.asarray(cloud.colors) if cloud.has_colors() else None

pred_pts, _ = load("visualization_outputs/stage2_coop_000069_pred.pcd")
gt_pts, _ = load("/media/.../000069.pcd")
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=gt_pts[:,0], y=gt_pts[:,1], z=gt_pts[:,2], mode="markers",
                           marker=dict(size=1, color="rgba(180,180,180,0.6)"), name="GT"))
fig.add_trace(go.Scatter3d(x=pred_pts[:,0], y=pred_pts[:,1], z=pred_pts[:,2], mode="markers",
                           marker=dict(size=1, color="rgba(255,0,0,0.8)"), name="Prediction"))
plot(fig, filename="visualization_outputs/html/stage2_coop_custom.html", auto_open=False)
```
执行完成后即可在浏览器访问 `stage2_coop_custom.html`。

## 4. 注意事项
- 端口 8090 仅在当前终端会话有效，断开 SSH 或 `Ctrl+C` 后需要重新执行。
- 访问地址由 VS Code 转发决定，若跳出的链接里端口不是 8090（例如 `48090`），以 VS Code 提示的地址为准。
- 若使用其他 SSH 客户端或本地浏览器无法访问，可考虑 `ssh -L 8090:localhost:8090` 手动建隧道，再访问 `http://127.0.0.1:8090/`.
