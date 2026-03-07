from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BEVGridSpec:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    voxel_size: float

    @property
    def x_min(self) -> float:
        return float(self.x_range[0])

    @property
    def x_max(self) -> float:
        return float(self.x_range[1])

    @property
    def y_min(self) -> float:
        return float(self.y_range[0])

    @property
    def y_max(self) -> float:
        return float(self.y_range[1])

    @property
    def width(self) -> int:
        return int(math.ceil((self.x_max - self.x_min) / self.voxel_size))

    @property
    def height(self) -> int:
        return int(math.ceil((self.y_max - self.y_min) / self.voxel_size))


class DifferentiableBEVRasterizer(nn.Module):
    def __init__(
        self,
        *,
        x_range: Sequence[float] = (0.0, 120.0),
        y_range: Sequence[float] = (-50.0, 50.0),
        voxel_size: float = 0.5,
        max_points: int = 200_000,
        eps: float = 1e-6,
        log_density: bool = True,
        include_high_ratio: bool = False,
        z_high_threshold: float = 0.5,
        include_var_z: bool = False,
    ) -> None:
        super().__init__()
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError("x_range/y_range must be length-2 sequences.")
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0.")
        self.grid = BEVGridSpec(
            x_range=(float(x_range[0]), float(x_range[1])),
            y_range=(float(y_range[0]), float(y_range[1])),
            voxel_size=float(voxel_size),
        )
        self.max_points = int(max_points)
        self.eps = float(eps)
        self.log_density = bool(log_density)
        self.include_high_ratio = bool(include_high_ratio)
        self.z_high_threshold = float(z_high_threshold)
        self.include_var_z = bool(include_var_z)

    @property
    def out_channels(self) -> int:
        # Base features: [density, mean_z]
        channels = 2
        if self.include_high_ratio:
            channels += 1
        if self.include_var_z:
            channels += 1
        return channels

    def forward(
        self, points_xyz: torch.Tensor, point_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            points_xyz: (B, N, 3) points in ego CARLA/UE convention (X forward, Y right, Z up).
            point_weights: (B, N) optional weights/masks (0 to ignore).

        Returns:
            bev_feat: (B, 2, H, W) = [density, mean_z].
        """
        if points_xyz.ndim != 3 or points_xyz.shape[-1] != 3:
            raise ValueError("points_xyz must have shape (B, N, 3).")

        points_xyz = points_xyz.float()
        batch_size, num_points, _ = points_xyz.shape
        if point_weights is None:
            point_weights = torch.ones(
                (batch_size, num_points), device=points_xyz.device, dtype=torch.float32
            )
        else:
            if point_weights.shape != (batch_size, num_points):
                raise ValueError("point_weights must have shape (B, N).")
            point_weights = point_weights.float()

        if self.max_points > 0 and num_points > self.max_points:
            # Subsample points for efficiency.
            #
            # Important: do *not* sample uniformly over all points when many points have
            # zero weights (e.g., after z filtering / mask weighting), otherwise the
            # effective signal collapses. Prefer sampling from non-zero weight points,
            # and fall back to uniform when no weights are available.
            weights_clamped = point_weights.clamp(min=0.0)
            nonzero = weights_clamped > 0

            keep_idx_list: list[torch.Tensor] = []
            for b in range(int(batch_size)):
                valid_idx = torch.nonzero(nonzero[b], as_tuple=False).reshape(-1)
                if valid_idx.numel() == 0:
                    # Fallback: uniform sampling over all points.
                    keep_idx = torch.randperm(num_points, device=points_xyz.device)[: self.max_points]
                    keep_idx_list.append(keep_idx)
                    continue

                if valid_idx.numel() >= self.max_points:
                    w = weights_clamped[b].index_select(0, valid_idx)
                    w_sum = float(w.sum().item())
                    if w_sum <= 0:
                        # Fallback: uniform sampling over nonzero indices.
                        perm = torch.randperm(valid_idx.numel(), device=points_xyz.device)[: self.max_points]
                        keep_idx_list.append(valid_idx.index_select(0, perm))
                    else:
                        probs = w / w.sum()
                        sel = torch.multinomial(probs, num_samples=self.max_points, replacement=False)
                        keep_idx_list.append(valid_idx.index_select(0, sel))
                else:
                    # Not enough valid points: keep them all, then pad by sampling (with replacement)
                    # from the valid set to reach max_points.
                    pad = self.max_points - int(valid_idx.numel())
                    pad_sel = valid_idx[
                        torch.randint(valid_idx.numel(), (pad,), device=points_xyz.device)
                    ]
                    keep_idx_list.append(torch.cat([valid_idx, pad_sel], dim=0))

            keep_idx_batched = torch.stack(keep_idx_list, dim=0)  # (B, max_points)
            points_xyz = points_xyz.gather(
                1, keep_idx_batched[..., None].expand(-1, -1, 3)
            )
            point_weights = point_weights.gather(1, keep_idx_batched)
            num_points = self.max_points

        x = points_xyz[..., 0]
        y = points_xyz[..., 1]
        z = points_xyz[..., 2]

        finite = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
        in_range = (
            (x >= self.grid.x_min)
            & (x <= self.grid.x_max)
            & (y >= self.grid.y_min)
            & (y <= self.grid.y_max)
        )
        weights = point_weights * finite.float() * in_range.float()

        u = (x - self.grid.x_min) / self.grid.voxel_size
        v = (y - self.grid.y_min) / self.grid.voxel_size

        ix0 = torch.floor(u).to(torch.long)
        iy0 = torch.floor(v).to(torch.long)
        dx = (u - ix0.float()).clamp(0.0, 1.0)
        dy = (v - iy0.float()).clamp(0.0, 1.0)

        ix1 = ix0 + 1
        iy1 = iy0 + 1

        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        height = self.grid.height
        width = self.grid.width
        grid_size = height * width

        pts_flat = points_xyz.reshape(-1, 3)
        z_flat = pts_flat[:, 2]
        weights_flat = weights.reshape(-1)
        batch_ids = (
            torch.arange(batch_size, device=points_xyz.device)
            .repeat_interleave(num_points)
            .to(torch.long)
        )
        grid_offsets = batch_ids * grid_size

        sum_w = torch.zeros(
            (batch_size * grid_size,), device=points_xyz.device, dtype=torch.float32
        )
        sum_z = torch.zeros_like(sum_w)
        sum_z2 = torch.zeros_like(sum_w) if self.include_var_z else None
        sum_w_high = torch.zeros_like(sum_w) if self.include_high_ratio else None

        def _accumulate(
            ix: torch.Tensor,
            iy: torch.Tensor,
            w_ij: torch.Tensor,
            *,
            weights_flat_in: torch.Tensor,
            sum_w_out: torch.Tensor,
            sum_z_out: torch.Tensor | None,
            sum_z2_out: torch.Tensor | None,
        ) -> None:
            ix_flat = ix.reshape(-1)
            iy_flat = iy.reshape(-1)
            w_flat = (w_ij.reshape(-1) * weights_flat_in).clamp(min=0.0)
            valid = (
                (w_flat > 0)
                & (ix_flat >= 0)
                & (ix_flat < width)
                & (iy_flat >= 0)
                & (iy_flat < height)
            )
            if not torch.any(valid):
                return
            flat_idx = (iy_flat * width + ix_flat) + grid_offsets
            flat_idx = flat_idx[valid]
            w_flat = w_flat[valid]
            sum_w_out.scatter_add_(0, flat_idx, w_flat)
            if sum_z_out is not None:
                z_sel = z_flat[valid]
                sum_z_out.scatter_add_(0, flat_idx, w_flat * z_sel)
                if sum_z2_out is not None:
                    sum_z2_out.scatter_add_(0, flat_idx, w_flat * z_sel * z_sel)

        _accumulate(
            ix0,
            iy0,
            w00,
            weights_flat_in=weights_flat,
            sum_w_out=sum_w,
            sum_z_out=sum_z,
            sum_z2_out=sum_z2,
        )
        _accumulate(
            ix1,
            iy0,
            w10,
            weights_flat_in=weights_flat,
            sum_w_out=sum_w,
            sum_z_out=sum_z,
            sum_z2_out=sum_z2,
        )
        _accumulate(
            ix0,
            iy1,
            w01,
            weights_flat_in=weights_flat,
            sum_w_out=sum_w,
            sum_z_out=sum_z,
            sum_z2_out=sum_z2,
        )
        _accumulate(
            ix1,
            iy1,
            w11,
            weights_flat_in=weights_flat,
            sum_w_out=sum_w,
            sum_z_out=sum_z,
            sum_z2_out=sum_z2,
        )

        sum_w = sum_w.view(batch_size, height, width)
        sum_z = sum_z.view(batch_size, height, width)
        mean_z = sum_z / (sum_w + self.eps)

        density = torch.log1p(sum_w) if self.log_density else sum_w
        channels = [density, mean_z]

        if self.include_high_ratio:
            z_high = z >= self.z_high_threshold
            weights_high = point_weights * finite.float() * in_range.float() * z_high.float()
            weights_high_flat = weights_high.reshape(-1)
            _accumulate(
                ix0,
                iy0,
                w00,
                weights_flat_in=weights_high_flat,
                sum_w_out=sum_w_high,  # type: ignore[arg-type]
                sum_z_out=None,
                sum_z2_out=None,
            )
            _accumulate(
                ix1,
                iy0,
                w10,
                weights_flat_in=weights_high_flat,
                sum_w_out=sum_w_high,  # type: ignore[arg-type]
                sum_z_out=None,
                sum_z2_out=None,
            )
            _accumulate(
                ix0,
                iy1,
                w01,
                weights_flat_in=weights_high_flat,
                sum_w_out=sum_w_high,  # type: ignore[arg-type]
                sum_z_out=None,
                sum_z2_out=None,
            )
            _accumulate(
                ix1,
                iy1,
                w11,
                weights_flat_in=weights_high_flat,
                sum_w_out=sum_w_high,  # type: ignore[arg-type]
                sum_z_out=None,
                sum_z2_out=None,
            )
            sum_w_high_grid = sum_w_high.view(batch_size, height, width)  # type: ignore[union-attr]
            high_ratio = (sum_w_high_grid / (sum_w + self.eps)).clamp(0.0, 1.0)
            channels.insert(1, high_ratio)

        if self.include_var_z:
            sum_z2_grid = sum_z2.view(batch_size, height, width)  # type: ignore[union-attr]
            mean_z2 = sum_z2_grid / (sum_w + self.eps)
            var_z = (mean_z2 - mean_z * mean_z).clamp(min=0.0)
            channels.append(var_z)

        bev_feat = torch.stack(channels, dim=1)
        return torch.nan_to_num(bev_feat, nan=0.0, posinf=0.0, neginf=0.0)


class _ConvGNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class BEVCenternetHead(nn.Module):
    def __init__(
        self,
        *,
        x_range: Sequence[float] = (0.0, 120.0),
        y_range: Sequence[float] = (-50.0, 50.0),
        voxel_size: float = 0.5,
        max_points: int = 200_000,
        include_high_ratio: bool = False,
        z_high_threshold: float = 0.5,
        include_var_z: bool = False,
        hidden_channels: int = 64,
        num_conv: int = 3,
        heatmap_bias: float = -2.19,
        heatmap_support_gate: bool = False,
        support_kernel: int = 5,
        support_var_z_tau: float = 0.1,
        support_detach: bool = True,
    ) -> None:
        super().__init__()
        self.rasterizer = DifferentiableBEVRasterizer(
            x_range=x_range,
            y_range=y_range,
            voxel_size=voxel_size,
            max_points=max_points,
            include_high_ratio=include_high_ratio,
            z_high_threshold=z_high_threshold,
            include_var_z=include_var_z,
        )
        self.grid = self.rasterizer.grid

        trunk: list[nn.Module] = []
        in_ch = int(self.rasterizer.out_channels)
        for _ in range(int(num_conv)):
            trunk.append(_ConvGNReLU(in_ch, int(hidden_channels)))
            in_ch = int(hidden_channels)
        self.trunk = nn.Sequential(*trunk)

        self.heatmap_head = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.reg_head = nn.Conv2d(in_ch, 8, kernel_size=1)

        nn.init.constant_(self.heatmap_head.bias, float(heatmap_bias))
        self.heatmap_support_gate = bool(heatmap_support_gate)
        self.support_kernel = int(support_kernel)
        self.support_var_z_tau = float(support_var_z_tau)
        self.support_detach = bool(support_detach)

        if self.support_kernel <= 0:
            raise ValueError("support_kernel must be > 0")
        if self.support_var_z_tau <= 0:
            raise ValueError("support_var_z_tau must be > 0")

    def _support_map(self, bev_feat: torch.Tensor) -> torch.Tensor | None:
        if not self.heatmap_support_gate:
            return None
        if bev_feat.ndim != 4:
            return None
        # If var_z is enabled, it is appended as the last channel.
        if not getattr(self.rasterizer, "include_var_z", False):
            return None
        var_z = bev_feat[:, -1]
        pad = self.support_kernel // 2
        var_pool = F.max_pool2d(
            var_z[:, None],
            kernel_size=self.support_kernel,
            stride=1,
            padding=pad,
        )
        support = var_pool / (var_pool + self.support_var_z_tau)
        return support.detach() if self.support_detach else support

    def forward(
        self, points_xyz: torch.Tensor, point_weights: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        feat = self.rasterizer(points_xyz, point_weights=point_weights)
        trunk = self.trunk(feat)
        heatmap = torch.sigmoid(self.heatmap_head(trunk))
        support = self._support_map(feat)
        if support is not None:
            heatmap = heatmap * support
        reg = self.reg_head(trunk)
        # Detach BEV features so downstream visualization / post-processing does not
        # keep large autograd graphs alive during end-to-end finetuning.
        return {"heatmap": heatmap, "reg": reg, "bev_feat": feat.detach()}
