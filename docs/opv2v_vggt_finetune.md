# OPV2V → VGGT Fine-tuning (Metric Scale)

## Dataset

This repo expects OPV2V to be reachable via:

- `map-anything/data/opv2v_images` → OPV2V raw RGB root (with `train/validate/test`)
- `map-anything/data/opv2v_depth` → precomputed sparse depth `.npy` files in the same split layout

These are already set in `configs/machine/local_a800.yaml` via `machine.opv2v_images_root` / `machine.opv2v_depth_root`.

## Scale / Pose Normalization

VGGT itself is scale-ambiguous, but OPV2V provides metric GT depth + poses.

We keep metric scale by using the OPV2V VGGT metric loss config:

- `configs/loss/opv2v_vggt_pose_metric_loss.yaml` uses `norm_mode='?avg_dis'` so metric samples skip pose/scene normalization.

Optionally, you can also enable the metric scale head in `VGGTWrapper` (`model.model_config.enable_metric_scale_head=true`) to
learn a scalar `metric_scaling_factor` applied to both depth and pose translations.

### Scale metric interpretation rule

When reading logs/summaries, separate these two concepts:

- `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale`:
  - this is a **loss-space** training metric (after log transform + robust criterion + weight),
  - target is **near 0** (smaller is better),
  - it is **not** a direct scale ratio.
- `metric_scaling_factor` / `scale_err_mean`:
  - this is **ratio-space** scale behavior.
  - For OPV2V + `norm_mode=avg_dis`, the correct GT target is per-frame normalization factor `g` (not constant 1).
  - Preferred metric is `scale_to_gt_err_mean = mean(|metric_scaling_factor / g - 1|)`, target is **near 0**.

For cross-run comparisons with different loss configs (`scale_loss_weight`, `direct_scale_loss`, `loss_in_log`, `norm_mode`),
prefer ratio-to-GT metrics (`scale_to_gt_*`) over raw scale loss.

## Recommended Local 1-GPU Runs (>=30 epochs)

These scripts run on a single GPU (no `torchrun`) and use moderate per-epoch dataset sizes:

```bash
cd map-anything

# (1) Metric supervision, no scale head
bash bash_scripts/train/finetuning/opv2v_vggt_suite_local_1gpu_metric.sh

# (2) Metric supervision + metric scale head
bash bash_scripts/train/finetuning/opv2v_vggt_suite_local_1gpu_scalehead.sh

# Override epochs (e.g., 50/100) or any Hydra option:
bash bash_scripts/train/finetuning/opv2v_vggt_suite_local_1gpu_metric.sh train_params.epochs=50
```

Key configs used:

- single-agent: `dataset=opv2v_ft_vggt_1k` + `train_params=opv2v_vggt_single_4v_long`
- coop (2 agents / 8 views): `dataset=opv2v_coop_ft_2a8v_vggt_500` + `train_params=opv2v_vggt_coop_8v_long`

## Monitoring

Each run directory contains `train.log` and TensorBoard events:

```bash
tail -f <RUN_DIR>/train.log
tensorboard --logdir <RUN_DIR>
```

Metric sanity checks to watch during eval:

- `pose_trans_l2_m` (relative translation error, meters; target < 1)
- `pose_rot_deg` (relative rotation error, degrees)
- `depth_z_mae_m` (depth Z MAE in meters; target < 1)
- `FactoredGeometryScaleRegr3DPlusNormalGMLoss_scale` (scale loss; target near 0; not ratio)
- `scale_to_gt_err_mean` from eval summary (ratio-space scale drift to GT factor; target near 0)

## Visual Sanity Checks (Boxes / Point Cloud)

Project OPV2V GT 3D boxes onto images using GT poses (green) and VGGT predicted poses (magenta, aligned to GT view0):

```bash
cd map-anything
PYTHONPATH=$(pwd) python scripts/viz_opv2v_boxes_on_images_pred_pose.py \
  --checkpoint <RUN_DIR>/checkpoint-best.pth \
  --mode coop --split validate --index 0 --num_views 8 \
  --out_dir eval_runs/opv2v_boxproj_predpose/coop_idx0
```

Overlay VGGT predicted point cloud (camera-frame pointmaps placed into ego frame with GT poses) vs OPV2V GT LiDAR + boxes:

```bash
cd map-anything
PYTHONPATH=$(pwd) python scripts/viz_opv2v_vggt_pred_vs_gt_boxes_html.py \
  --checkpoint <RUN_DIR>/checkpoint-best.pth \
  --split validate --index 0 --num_views 8 \
  --out_html eval_runs/vggt_pred_viz/coop_idx0.html
```

Notes:

- OPV2V is CARLA/UE coords (X forward, Y right, Z up).
- VGGT/MapAnything geometry tensors are OpenCV coords (X right, Y down, Z forward).
- The dataset loader converts CARLA poses into OpenCV convention; visualization scripts convert back as needed.
