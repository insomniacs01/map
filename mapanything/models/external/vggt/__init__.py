# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference wrapper for VGGT
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapanything.models.mapanything.detection_head import BEVCenternetHead
from mapanything.models.external.vggt.models.vggt import VGGT
from mapanything.models.external.vggt.utils.geometry import closed_form_inverse_se3
from mapanything.models.external.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from mapanything.models.external.vggt.utils.rotation import mat_to_quat
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    convert_z_depth_to_depth_along_ray,
    depthmap_to_camera_frame,
    get_rays_in_camera_frame,
    quaternion_to_rotation_matrix,
)


class VGGTWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        torch_hub_force_reload,
        pretrained_checkpoint_path: str | None = None,
        load_pretrained_weights=True,
        depth=24,
        num_heads=16,
        intermediate_layer_idx=[4, 11, 17, 23],
        load_custom_ckpt=False,
        custom_ckpt_path=None,
        autocast_dtype: str | None = None,
        enable_metric_scale_head: bool = False,
        metric_scale_head_type: str = "mlp",
        metric_scale_mode: str = "exp",
        metric_scale_clamp_min: float = 1e-6,
        metric_scale_clamp_max: float = 1e6,
        metric_scale_head_hidden_dim: int | None = None,
        causal_global_attention: bool = False,
        causal_camera_attention: bool = False,
    ):
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_custom_ckpt = load_custom_ckpt
        self.custom_ckpt_path = custom_ckpt_path
        self.autocast_dtype = autocast_dtype
        self.enable_metric_scale_head = bool(enable_metric_scale_head)
        self.metric_scale_head_type = str(metric_scale_head_type)
        self.metric_scale_mode = str(metric_scale_mode)
        self.metric_scale_clamp_min = float(metric_scale_clamp_min)
        self.metric_scale_clamp_max = float(metric_scale_clamp_max)
        self.causal_global_attention = bool(causal_global_attention)
        self.causal_camera_attention = bool(causal_camera_attention)

        # VGGT's aggregator uses `torch.hub.load("facebookresearch/dinov2", ...)`.
        # Prefer the locally cached repo (if present) to avoid unnecessary GitHub access.
        try:
            from mapanything.utils.hf_utils.hf_helpers import (
                _patch_torch_hub_load_for_offline_dinov2,
            )

            _patch_torch_hub_load_for_offline_dinov2()
        except Exception:
            pass

        if load_pretrained_weights:
            # Load pre-trained weights
            if not torch_hub_force_reload:
                # Initialize the 1B VGGT model from huggingface hub cache
                print("Loading facebook/VGGT-1B from huggingface cache ...")
                self.model = VGGT.from_pretrained("facebook/VGGT-1B")
            else:
                # Initialize the 1B VGGT model
                print("Re-downloading facebook/VGGT-1B ...")
                self.model = VGGT.from_pretrained(
                    "facebook/VGGT-1B", force_download=True
                )
        else:
            # Load the VGGT class
            self.model = VGGT(
                depth=depth,
                num_heads=num_heads,
                intermediate_layer_idx=intermediate_layer_idx,
                causal_global_attention=self.causal_global_attention,
                causal_camera_attention=self.causal_camera_attention,
            )
        if hasattr(self.model, "aggregator"):
            self.model.aggregator.use_causal_global = self.causal_global_attention
        if hasattr(self.model, "camera_head"):
            self.model.camera_head.causal_attention = self.causal_camera_attention

        # Get the dtype for VGGT inference.
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+).
        self.dtype = torch.float32
        if torch.cuda.is_available():
            major_cc = torch.cuda.get_device_capability(0)[0]
            default_dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
            if self.autocast_dtype is None or str(self.autocast_dtype).lower() in (
                "",
                "auto",
                "default",
            ):
                self.dtype = default_dtype
            else:
                requested = str(self.autocast_dtype).lower()
                if requested in ("bf16", "bfloat16"):
                    self.dtype = torch.bfloat16
                elif requested in ("fp16", "float16", "half"):
                    self.dtype = torch.float16
                elif requested in ("fp32", "float32"):
                    self.dtype = torch.float32
                else:
                    raise ValueError(
                        "autocast_dtype must be one of {auto,bf16,fp16,fp32}, "
                        f"got {self.autocast_dtype!r}"
                    )

        # Load custom checkpoint if requested
        if self.load_custom_ckpt:
            print(f"Loading checkpoint from {self.custom_ckpt_path} ...")
            assert self.custom_ckpt_path is not None, (
                "custom_ckpt_path must be provided if load_custom_ckpt is set to True"
            )
            custom_ckpt = torch.load(self.custom_ckpt_path, weights_only=False)
            print(self.model.load_state_dict(custom_ckpt, strict=True))
            del custom_ckpt  # in case it occupies memory

        self.metric_scale_head: nn.Module | None = None
        if self.enable_metric_scale_head:
            if self.metric_scale_head_type not in ("mlp", "constant"):
                raise ValueError(
                    "metric_scale_head_type must be one of {'mlp','constant'}, "
                    f"got {self.metric_scale_head_type!r}"
                )

            if self.metric_scale_head_type == "mlp":
                token_dim = int(self.model.camera_head.token_norm.normalized_shape[0])
                hidden_dim = int(
                    metric_scale_head_hidden_dim or max(128, token_dim // 4)
                )
                self.metric_scale_head = nn.Sequential(
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.zeros_(self.metric_scale_head[-1].weight)
                if self.metric_scale_mode == "exp":
                    nn.init.zeros_(self.metric_scale_head[-1].bias)
                elif self.metric_scale_mode == "softplus":
                    nn.init.constant_(self.metric_scale_head[-1].bias, math.log(1.0))
                elif self.metric_scale_mode == "linear":
                    nn.init.zeros_(self.metric_scale_head[-1].bias)
                else:
                    raise ValueError(
                        "metric_scale_mode must be one of {'exp','softplus','linear'}, "
                        f"got {self.metric_scale_mode!r}"
                    )
            else:
                class _ConstantScaleHead(nn.Module):
                    def __init__(self, init_value: float):
                        super().__init__()
                        self.raw = nn.Parameter(
                            torch.tensor([[init_value]], dtype=torch.float32)
                        )

                    def forward(self, features: torch.Tensor) -> torch.Tensor:
                        batch = int(features.shape[0])
                        return self.raw.expand(batch, 1)

                init_value = 0.0 if self.metric_scale_mode != "linear" else 1.0
                self.metric_scale_head = _ConstantScaleHead(init_value=init_value)

        # Optionally warm-start from a MapAnything training checkpoint (or a raw state_dict).
        # This loads *only model weights* (strict=False) and does not affect optimizer state
        # unless you also use train_params.resume in the trainer.
        if self.pretrained_checkpoint_path:
            ckpt_path = str(self.pretrained_checkpoint_path)
            print(f"Loading pretrained checkpoint from {ckpt_path} ...")
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[VGGTWrapper] Missing keys: {len(missing)}")
            if unexpected:
                print(f"[VGGTWrapper] Unexpected keys: {len(unexpected)}")
            del checkpoint  # free CPU RAM

    @torch.inference_mode()
    def infer_stream(
        self,
        views: list[dict],
        past_key_values=None,
        past_key_values_camera=None,
        return_cache: bool = False,
    ):
        """
        Streaming inference over an ordered list of views (frames).

        Each view should follow the same format as `forward()`, with "img" and
        "data_norm_type" fields. This method processes frames sequentially and
        caches KV tensors in the global attention blocks + camera head trunk.
        """
        if not views:
            return [] if not return_cache else ([], past_key_values, past_key_values_camera)

        # Initialize caches if needed.
        if past_key_values is None:
            past_key_values = [None] * self.model.aggregator.depth
        if past_key_values_camera is None:
            past_key_values_camera = [None] * self.model.camera_head.trunk_depth

        preds = []
        for frame_idx, view in enumerate(views):
            data_norm_type = view["data_norm_type"][0]
            if data_norm_type != "identity":
                raise ValueError(
                    "VGGT expects a normalized image but without the DINOv2 mean and std applied"
                )

            images = view["img"].unsqueeze(1)

            autocast_ctx = (
                torch.autocast("cuda", dtype=self.dtype)
                if images.is_cuda
                else nullcontext()
            )
            with autocast_ctx:
                aggregated_tokens_list, ps_idx, past_key_values = self.model.aggregator(
                    images,
                    past_key_values=past_key_values,
                    use_cache=True,
                    past_frame_idx=frame_idx,
                )

            metric_scaling_factor = None
            if self.enable_metric_scale_head:
                if self.metric_scale_head is None:
                    raise RuntimeError(
                        "enable_metric_scale_head=True but metric_scale_head is None."
                    )
                tokens = aggregated_tokens_list[-1]
                scale_features = tokens[:, :, 0].float().mean(dim=1)  # (B, C)
                scale_raw = self.metric_scale_head(scale_features)  # (B, 1)
                if self.metric_scale_mode == "exp":
                    metric_scaling_factor = torch.exp(scale_raw)
                elif self.metric_scale_mode == "softplus":
                    metric_scaling_factor = F.softplus(scale_raw)
                elif self.metric_scale_mode == "linear":
                    metric_scaling_factor = scale_raw
                else:
                    raise RuntimeError(
                        f"Unexpected metric_scale_mode={self.metric_scale_mode!r}"
                    )
                metric_scaling_factor = metric_scaling_factor.clamp(
                    min=self.metric_scale_clamp_min, max=self.metric_scale_clamp_max
                )

            amp_off_ctx = (
                torch.autocast("cuda", enabled=False)
                if images.is_cuda
                else nullcontext()
            )
            with amp_off_ctx:
                pose_enc_out = self.model.camera_head(
                    aggregated_tokens_list,
                    past_key_values_camera=past_key_values_camera,
                    use_cache=True,
                )
                if isinstance(pose_enc_out, tuple):
                    pose_enc_list, past_key_values_camera = pose_enc_out
                else:
                    pose_enc_list = pose_enc_out
                pose_enc = pose_enc_list[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images.shape[-2:]
                )

                depth_map, depth_conf = self.model.depth_head(
                    aggregated_tokens_list, images, ps_idx
                )

                curr_view_extrinsic = closed_form_inverse_se3(extrinsic[:, 0, ...])
                curr_view_intrinsic = intrinsic[:, 0, ...]
                curr_view_depth_z = depth_map[:, 0, ...].squeeze(-1)
                curr_view_confidence = depth_conf[:, 0, ...]

                if metric_scaling_factor is not None:
                    curr_view_depth_z = curr_view_depth_z * metric_scaling_factor.view(
                        curr_view_depth_z.shape[0], 1, 1
                    )

                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])
                if metric_scaling_factor is not None:
                    curr_view_cam_translations = (
                        curr_view_cam_translations * metric_scaling_factor
                    )

                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                ).unsqueeze(-1)

                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic,
                    curr_view_depth_z.shape[1],
                    curr_view_depth_z.shape[2],
                    normalize_to_unit_sphere=True,
                )

                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                pred = {
                    "pts3d": curr_view_pts3d,
                    "pts3d_cam": curr_view_pts3d_cam,
                    "ray_directions": curr_view_ray_dirs,
                    "depth_along_ray": curr_view_depth_along_ray,
                    "cam_trans": curr_view_cam_translations,
                    "cam_quats": curr_view_cam_quats,
                    "conf": curr_view_confidence,
                }
                if metric_scaling_factor is not None:
                    pred["metric_scaling_factor"] = metric_scaling_factor

            preds.append(pred)

        if return_cache:
            return preds, past_key_values, past_key_values_camera
        return preds

    def forward(self, views):
        """
        Forward pass wrapper for VGGT

        Assumption:
        - All the input views have the same image shape.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
                                Each dictionary should contain the following keys:
                                    "img" (tensor): Image tensor of shape (B, C, H, W).
                                    "data_norm_type" (list): ["identity"]

        Returns:
            List[dict]: A list containing the final outputs for all N views.
        """
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, _, height, width = views[0]["img"].shape
        num_views = len(views)

        # Check the data norm type
        # VGGT expects a normalized image but without the DINOv2 mean and std applied ("identity")
        data_norm_type = views[0]["data_norm_type"][0]
        assert data_norm_type == "identity", (
            "VGGT expects a normalized image but without the DINOv2 mean and std applied"
        )

        # Concatenate the images to create a single (B, V, C, H, W) tensor
        img_list = [view["img"] for view in views]
        images = torch.stack(img_list, dim=1)

        autocast_ctx = (
            torch.autocast("cuda", dtype=self.dtype) if images.is_cuda else nullcontext()
        )
        # Run the VGGT aggregator
        with autocast_ctx:
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)

        metric_scaling_factor = None
        if self.enable_metric_scale_head:
            if self.metric_scale_head is None:
                raise RuntimeError("enable_metric_scale_head=True but metric_scale_head is None.")
            tokens = aggregated_tokens_list[-1]
            scale_features = tokens[:, :, 0].float().mean(dim=1)  # (B, C)
            scale_raw = self.metric_scale_head(scale_features)  # (B, 1)
            if self.metric_scale_mode == "exp":
                metric_scaling_factor = torch.exp(scale_raw)
            elif self.metric_scale_mode == "softplus":
                metric_scaling_factor = F.softplus(scale_raw)
            elif self.metric_scale_mode == "linear":
                metric_scaling_factor = scale_raw
            else:
                raise RuntimeError(f"Unexpected metric_scale_mode={self.metric_scale_mode!r}")
            metric_scaling_factor = metric_scaling_factor.clamp(
                min=self.metric_scale_clamp_min, max=self.metric_scale_clamp_max
            )

        # Run the Camera + Pose Branch of VGGT
        amp_off_ctx = (
            torch.autocast("cuda", enabled=False) if images.is_cuda else nullcontext()
        )
        with amp_off_ctx:
            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            # Extrinsics Shape: (B, V, 3, 4)
            # Intrinsics Shape: (B, V, 3, 3)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images.shape[-2:]
            )

            # Predict Depth Maps
            # Depth Shape: (B, V, H, W, 1)
            # Depth Confidence Shape: (B, V, H, W)
            depth_map, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images, ps_idx
            )

            # Convert the output to MapAnything format
            res = []
            for view_idx in range(num_views):
                # Get the extrinsics, intrinsics, depth map for the current view
                curr_view_extrinsic = extrinsic[:, view_idx, ...]
                curr_view_extrinsic = closed_form_inverse_se3(
                    curr_view_extrinsic
                )  # Convert to cam2world
                curr_view_intrinsic = intrinsic[:, view_idx, ...]
                curr_view_depth_z = depth_map[:, view_idx, ...]
                curr_view_depth_z = curr_view_depth_z.squeeze(-1)
                curr_view_confidence = depth_conf[:, view_idx, ...]

                if metric_scaling_factor is not None:
                    curr_view_depth_z = curr_view_depth_z * metric_scaling_factor.view(
                        batch_size_per_view, 1, 1
                    )

                # Get the camera frame pointmaps
                curr_view_pts3d_cam, _ = depthmap_to_camera_frame(
                    curr_view_depth_z, curr_view_intrinsic
                )

                # Convert the extrinsics to quaternions and translations
                curr_view_cam_translations = curr_view_extrinsic[..., :3, 3]
                curr_view_cam_quats = mat_to_quat(curr_view_extrinsic[..., :3, :3])
                if metric_scaling_factor is not None:
                    curr_view_cam_translations = (
                        curr_view_cam_translations * metric_scaling_factor
                    )

                # Convert the z depth to depth along ray
                curr_view_depth_along_ray = convert_z_depth_to_depth_along_ray(
                    curr_view_depth_z, curr_view_intrinsic
                )
                curr_view_depth_along_ray = curr_view_depth_along_ray.unsqueeze(-1)

                # Get the ray directions on the unit sphere in the camera frame
                _, curr_view_ray_dirs = get_rays_in_camera_frame(
                    curr_view_intrinsic, height, width, normalize_to_unit_sphere=True
                )

                # Get the pointmaps
                curr_view_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        curr_view_ray_dirs,
                        curr_view_depth_along_ray,
                        curr_view_cam_translations,
                        curr_view_cam_quats,
                    )
                )

                # Append the outputs to the result list
                res.append(
                    {
                        "pts3d": curr_view_pts3d,
                        "pts3d_cam": curr_view_pts3d_cam,
                        "ray_directions": curr_view_ray_dirs,
                        "depth_along_ray": curr_view_depth_along_ray,
                        "cam_trans": curr_view_cam_translations,
                        "cam_quats": curr_view_cam_quats,
                        "conf": curr_view_confidence,
                        **(
                            {"metric_scaling_factor": metric_scaling_factor}
                            if metric_scaling_factor is not None
                            else {}
                        ),
                    }
                )

        return res


class VGGTDetWrapper(VGGTWrapper):
    """VGGT wrapper with a BEV detection head, mirroring MapAnythingDet semantics.

    This enables using the same det/e2e evaluation pipeline (`scripts/batch_eval.py --det_metrics`)
    with `--model_arch vggt`, while keeping protocol knobs (`point_pose_source`, align-to-view0)
    consistent with MapAnythingDet.
    """

    def __init__(self, det_head_config: dict | None = None, **kwargs):
        self.det_head_config = det_head_config or {}
        super().__init__(**kwargs)

        cfg: dict = dict(self.det_head_config) if self.det_head_config else {}
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

    def _view_pose_for_detection(self, view: dict) -> torch.Tensor | None:
        """Return a (B, 4, 4) pose tensor for aligning points into ego/world."""
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

    def _pred_pose_for_detection(self, pred: dict) -> torch.Tensor | None:
        """Build a (B, 4, 4) cam2world pose from VGGT prediction dict."""
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

    def forward(self, views):
        preds = super().forward(views)
        if not self.det_head:
            return preds

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
                    if torch.is_tensor(pts_cam) and pts_cam.ndim == 4 and pts_cam.shape[-1] == 3:
                        rot = pose[:, :3, :3]
                        trans = pose[:, :3, 3]
                        pts3d_for_det = torch.einsum("bij,bhwj->bhwi", rot, pts_cam) + trans[:, None, None, :]
            elif self.det_point_pose_source == "pred" and pred_align is not None:
                pred_pose = self._pred_pose_for_detection(pred)
                if torch.is_tensor(pred_pose):
                    if pts_cam is not None and torch.is_tensor(pts_cam) and pts_cam.ndim == 4 and pts_cam.shape[-1] == 3:
                        if pred_pose.ndim == 3 and pred_pose.shape[-2:] == (4, 4):
                            pose = pred_align @ pred_pose
                            rot = pose[:, :3, :3]
                            trans = pose[:, :3, 3]
                            pts3d_for_det = torch.einsum("bij,bhwj->bhwi", rot, pts_cam) + trans[:, None, None, :]
                    elif pts3d is not None and torch.is_tensor(pts3d) and pts3d.ndim == 4 and pts3d.shape[-1] == 3:
                        rot = pred_align[:, :3, :3]
                        trans = pred_align[:, :3, 3]
                        pts3d_for_det = torch.einsum("bij,bhwj->bhwi", rot, pts3d) + trans[:, None, None, :]

            pts3d = pts3d_for_det
            if not torch.is_tensor(pts3d) or pts3d.ndim != 4 or pts3d.shape[-1] != 3:
                continue
            batch_size = int(pts3d.shape[0])
            pts_flat = pts3d.reshape(batch_size, -1, 3)
            points_per_view.append(pts_flat)

            if self.det_use_pred_mask and "non_ambiguous_mask" in pred:
                mask = pred["non_ambiguous_mask"]
                weights = mask.reshape(batch_size, -1).float()
            else:
                weights = torch.ones((batch_size, pts_flat.shape[1]), device=pts3d.device, dtype=torch.float32)

            if self.det_use_pred_conf and "conf" in pred:
                conf = pred["conf"]
                if torch.is_tensor(conf):
                    # VGGT uses (B,H,W); MapAnything uses (B,H,W,1) or (B,H,W)
                    if conf.ndim == 3:
                        conf_flat = conf.reshape(batch_size, -1).float()
                    elif conf.ndim == 4 and conf.shape[-1] == 1:
                        conf_flat = conf.reshape(batch_size, -1).float()
                    else:
                        conf_flat = None
                    if conf_flat is not None:
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

        # Keep detection head in fp32 to avoid autocast dtype/bias mismatches.
        with torch.autocast("cuda", enabled=False):
            det_out = self.det_head(pts_carla.float(), point_weights=weights.float())

        for pred in preds:
            pred["bev_det"] = det_out
        return preds
