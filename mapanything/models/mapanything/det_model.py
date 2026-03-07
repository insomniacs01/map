from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from mapanything.models.mapanything.detection_head import BEVCenternetHead
from mapanything.models.mapanything.model import MapAnything
from mapanything.utils.geometry import quaternion_to_rotation_matrix


class MapAnythingDet(MapAnything):
    def __init__(self, det_head_config: Optional[Dict[str, Any]] = None, **kwargs):
        self.det_head_config = det_head_config or {}
        super().__init__(**kwargs)

        cfg: Dict[str, Any] = dict(self.det_head_config) if self.det_head_config else {}
        self.det_enabled = bool(cfg.pop("enabled", False))
        self.det_stop_gradient = bool(cfg.pop("stop_gradient", False))
        self.det_use_pred_mask = bool(cfg.pop("use_pred_mask", True))
        self.det_use_pred_conf = bool(cfg.pop("use_pred_conf", False))
        self.det_conf_clip = cfg.pop("conf_clip", 10.0)
        self.det_conf_power = float(cfg.pop("conf_power", 1.0))
        self.det_point_z_min = cfg.pop("point_z_min", None)
        self.det_point_z_max = cfg.pop("point_z_max", None)
        self.det_point_pose_source = str(cfg.pop("point_pose_source", "gt")).lower()
        if self.det_point_pose_source not in ("gt", "pred"):
            raise ValueError("det_head_config.point_pose_source must be one of: ['gt', 'pred']")
        # When using predicted poses/points, you may still want to anchor the global frame to a
        # known reference (e.g., main-agent rig pose for view0). This makes det AP meaningful
        # under deployment-like evals where cross-agent GT poses are not provided.
        self.det_pred_pose_align_to_view0_gt = bool(cfg.pop("pred_pose_align_to_view0_gt", False))

        if self.det_conf_clip is not None:
            self.det_conf_clip = float(self.det_conf_clip)
        if self.det_point_z_min is not None:
            self.det_point_z_min = float(self.det_point_z_min)
        if self.det_point_z_max is not None:
            self.det_point_z_max = float(self.det_point_z_max)

        if self.det_enabled:
            self.det_head = BEVCenternetHead(**cfg)
        else:
            self.det_head = None

        # Carla/UE (X forward, Y right, Z up) -> OpenCV (X right, Y down, Z forward)
        # For row-vector points: p_carla = p_cv @ CARLA_TO_CV
        self.register_buffer(
            "_CARLA_TO_CV",
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _view_pose_for_detection(self, view: Dict[str, Any]) -> torch.Tensor | None:
        """Return a (B, 4, 4) pose tensor for aligning points into ego/world.

        Training datasets provide `camera_pose`, while `model.infer()` inputs provide `camera_poses`.
        We support both, and also accept (quats, trans) tuples (XYZW order) for inference.
        Note that `model.infer()` internally converts `camera_poses` into separate
        `camera_pose_quats` / `camera_pose_trans` keys, so we handle that case too.
        """

        pose = view.get("camera_pose")
        if pose is None:
            pose = view.get("camera_poses")
        if pose is None:
            quats = view.get("camera_pose_quats")
            trans = view.get("camera_pose_trans")
            if quats is None or trans is None:
                return None
            if not torch.is_tensor(quats):
                quats = torch.as_tensor(quats)
            if not torch.is_tensor(trans):
                trans = torch.as_tensor(trans)
            pose = (quats, trans)

        if isinstance(pose, tuple):
            if len(pose) != 2:
                return None
            quats, trans = pose
            if not torch.is_tensor(quats) or not torch.is_tensor(trans):
                return None
            if quats.ndim != 2 or quats.shape[-1] != 4:
                return None
            if trans.ndim != 2 or trans.shape[-1] != 3:
                return None
            rot = quaternion_to_rotation_matrix(quats)
            mat = torch.eye(4, device=rot.device, dtype=rot.dtype).unsqueeze(0).repeat(rot.shape[0], 1, 1)
            mat[:, :3, :3] = rot
            mat[:, :3, 3] = trans
            return mat

        if not torch.is_tensor(pose):
            return None
        if pose.ndim != 3 or pose.shape[-2:] != (4, 4):
            return None
        return pose

    def _pred_pose_for_detection(self, pred: Dict[str, Any]) -> torch.Tensor | None:
        """Build a (B, 4, 4) cam2world pose from model prediction dict.

        NOTE: `MapAnything.forward()` (used by this Det wrapper) returns raw pose outputs as
        `cam_quats` / `cam_trans` (quat order XYZW). The higher-level `infer()` postprocess
        later adds `camera_poses`, but detection runs inside `forward()`, so we must not
        depend on postprocessed keys here.
        """
        pose = pred.get("camera_poses")
        if torch.is_tensor(pose):
            if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
                return pose
            return None

        quats = pred.get("cam_quats")
        trans = pred.get("cam_trans")
        if quats is None or trans is None:
            return None
        if not torch.is_tensor(quats):
            quats = torch.as_tensor(quats)
        if not torch.is_tensor(trans):
            trans = torch.as_tensor(trans)
        if quats.ndim != 2 or quats.shape[-1] != 4:
            return None
        if trans.ndim != 2 or trans.shape[-1] != 3:
            return None

        rot = quaternion_to_rotation_matrix(quats)
        mat = torch.eye(4, device=rot.device, dtype=rot.dtype).unsqueeze(0).repeat(rot.shape[0], 1, 1)
        mat[:, :3, :3] = rot
        mat[:, :3, 3] = trans.to(dtype=rot.dtype)
        return mat

    def forward(self, views, memory_efficient_inference: bool = False):
        preds = super().forward(views, memory_efficient_inference=memory_efficient_inference)
        if not self.det_head:
            return preds

        # Optional: align the predicted world frame to a GT reference pose for view0 (if available).
        # This removes the global gauge freedom so det boxes are evaluated in the ego/world frame,
        # without leaking cross-agent GT poses.
        pred_align = None
        if self.det_point_pose_source == "pred" and self.det_pred_pose_align_to_view0_gt:
            try:
                if views and preds and isinstance(preds[0], dict):
                    gt0 = self._view_pose_for_detection(views[0])
                    pred0 = self._pred_pose_for_detection(preds[0])
                    if (
                        gt0 is not None
                        and torch.is_tensor(pred0)
                        and pred0.ndim == 3
                        and pred0.shape[-2:] == (4, 4)
                        and gt0.shape == pred0.shape
                    ):
                        pred_align = gt0 @ torch.linalg.inv(pred0)
            except Exception:
                pred_align = None

        points_per_view = []
        weights_per_view = []
        for view, pred in zip(views, preds):
            pts3d = pred.get("pts3d")
            pts_cam = pred.get("pts3d_cam")
            if pts3d is None and pts_cam is None:
                continue

            pts3d_for_det = pts3d
            if self.det_point_pose_source == "gt":
                pose = self._view_pose_for_detection(view)
                if pose is not None and pts_cam is not None:
                    if pts_cam.ndim == 4 and pts_cam.shape[-1] == 3:
                        rot = pose[:, :3, :3]
                        trans = pose[:, :3, 3]
                        pts3d_for_det = (
                            torch.einsum("bij,bhwj->bhwi", rot, pts_cam)
                            + trans[:, None, None, :]
                        )
            elif self.det_point_pose_source == "pred" and pred_align is not None:
                # Use predicted camera_poses but align them to a known GT view0 frame.
                pred_pose = self._pred_pose_for_detection(pred)
                if torch.is_tensor(pred_pose):
                    if pts_cam is not None and pts_cam.ndim == 4 and pts_cam.shape[-1] == 3:
                        if pred_pose.ndim == 3 and pred_pose.shape[-2:] == (4, 4):
                            pose = pred_align @ pred_pose
                            rot = pose[:, :3, :3]
                            trans = pose[:, :3, 3]
                            pts3d_for_det = (
                                torch.einsum("bij,bhwj->bhwi", rot, pts_cam)
                                + trans[:, None, None, :]
                            )
                    elif pts3d is not None and torch.is_tensor(pts3d) and pts3d.ndim == 4 and pts3d.shape[-1] == 3:
                        # Align already-world-frame pointmaps in-place when camera-frame points are unavailable.
                        rot = pred_align[:, :3, :3]
                        trans = pred_align[:, :3, 3]
                        pts3d_for_det = (
                            torch.einsum("bij,bhwj->bhwi", rot, pts3d)
                            + trans[:, None, None, :]
                        )

            pts3d = pts3d_for_det
            if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
                continue
            batch_size = pts3d.shape[0]
            pts_flat = pts3d.reshape(batch_size, -1, 3)
            points_per_view.append(pts_flat)

            if self.det_use_pred_mask and "non_ambiguous_mask" in pred:
                mask = pred["non_ambiguous_mask"]
                weights = mask.reshape(batch_size, -1).float()
            else:
                weights = torch.ones(
                    (batch_size, pts_flat.shape[1]),
                    device=pts3d.device,
                    dtype=torch.float32,
                )

            if self.det_use_pred_conf and "conf" in pred:
                conf = pred["conf"]
                if torch.is_tensor(conf) and conf.ndim == 3:
                    conf_flat = conf.reshape(batch_size, -1).float()
                    conf_flat = torch.nan_to_num(conf_flat, nan=0.0, posinf=0.0, neginf=0.0)
                    clip = self.det_conf_clip
                    if clip is not None:
                        conf_flat = conf_flat.clamp(min=0.0, max=float(clip))
                        conf_w = conf_flat / max(float(clip), 1e-6)
                    else:
                        conf_w = conf_flat
                    conf_w = conf_w.clamp(min=0.0).pow(self.det_conf_power)
                    weights = weights * conf_w
            weights_per_view.append(weights)

        if not points_per_view:
            return preds

        pts_cv = torch.cat(points_per_view, dim=1)
        weights = torch.cat(weights_per_view, dim=1)
        if self.det_stop_gradient:
            pts_cv = pts_cv.detach()

        pts_carla = torch.matmul(pts_cv, self._CARLA_TO_CV.to(pts_cv))
        if self.det_point_z_min is not None or self.det_point_z_max is not None:
            z = pts_carla[..., 2]
            z_mask = torch.ones_like(z, dtype=torch.bool)
            if self.det_point_z_min is not None:
                z_mask &= z >= float(self.det_point_z_min)
            if self.det_point_z_max is not None:
                z_mask &= z <= float(self.det_point_z_max)
            weights = weights * z_mask.float()
        # Keep detection head in fp32 to avoid autocast dtype/bias mismatches (e.g., bf16 input + fp32 bias).
        with torch.autocast("cuda", enabled=False):
            det_out = self.det_head(pts_carla.float(), point_weights=weights.float())

        for pred in preds:
            pred["bev_det"] = det_out

        return preds
