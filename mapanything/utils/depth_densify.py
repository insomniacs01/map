from __future__ import annotations

import numpy as np


def _sorted_offsets(radius: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))
    # Prefer closer neighbors first (Manhattan distance), then deterministic tie-break.
    offsets.sort(key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0]), abs(t[1])))
    return offsets


def densify_sparse_depthmap_nearest(depthmap: np.ndarray, *, radius: int) -> np.ndarray:
    """Densify a sparse depthmap by copying the nearest valid depth within a pixel radius.

    This is a conservative label densification intended for sparse LiDAR-projected depth.
    It only fills pixels that are within `radius` (Chebyshev neighborhood) of an originally
    valid pixel, and never extrapolates beyond that neighborhood.

    Valid pixels are defined as finite and strictly positive (`depth > 0`).

    Args:
        depthmap: (H, W) float array.
        radius: Pixel radius. 0 disables densification.

    Returns:
        A new (H, W) float array with additional filled depths.
    """
    if radius <= 0:
        return depthmap
    if depthmap.ndim != 2:
        raise ValueError(f"Expected depthmap shape (H, W), got {depthmap.shape}")

    depth_src = depthmap
    out = depthmap.copy()

    valid = np.isfinite(depth_src) & (depth_src > 0.0)
    filled = valid.copy()
    if not valid.any():
        return out

    H, W = depth_src.shape
    offsets = _sorted_offsets(int(radius))

    for dy, dx in offsets:
        # Compute overlap window between source (shifted) and destination.
        if dy >= 0:
            src_y0, src_y1 = 0, H - dy
            dst_y0, dst_y1 = dy, H
        else:
            src_y0, src_y1 = -dy, H
            dst_y0, dst_y1 = 0, H + dy

        if dx >= 0:
            src_x0, src_x1 = 0, W - dx
            dst_x0, dst_x1 = dx, W
        else:
            src_x0, src_x1 = -dx, W
            dst_x0, dst_x1 = 0, W + dx

        if src_y1 <= src_y0 or src_x1 <= src_x0:
            continue

        src_slice = (slice(src_y0, src_y1), slice(src_x0, src_x1))
        dst_slice = (slice(dst_y0, dst_y1), slice(dst_x0, dst_x1))

        src_valid = valid[src_slice]
        if not src_valid.any():
            continue

        dst_filled = filled[dst_slice]
        fill_mask = (~dst_filled) & src_valid
        if not fill_mask.any():
            continue

        out_view = out[dst_slice]
        out_view[fill_mask] = depth_src[src_slice][fill_mask]
        out[dst_slice] = out_view

        dst_filled[fill_mask] = True
        filled[dst_slice] = dst_filled

        if filled.all():
            break

    return out
