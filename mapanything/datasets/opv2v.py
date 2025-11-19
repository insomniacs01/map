import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from data_processing.opv2v_pose_utils import (
    CARLA_TO_CAMERA_CV,
    cords_to_pose,
    get_camera_poses_in_ego,
    load_frame_metadata,
)
from mapanything.datasets.base.base_dataset import BaseDataset


def _convert_pose_to_opencv(pose: np.ndarray) -> np.ndarray:
    """
    Convert a camera pose expressed in CARLA coordinates to OpenCV convention.
    """
    pose_cv = pose.copy()
    basis = CARLA_TO_CAMERA_CV[:3, :3]
    pose_cv[:3, :3] = basis @ pose[:3, :3] @ basis.T
    pose_cv[:3, 3] = basis @ pose[:3, 3]
    return pose_cv


class OPV2VDataset(BaseDataset):
    """
    Dataset loader that reads OPV2V frames directly from the raw CARLA dumps.

    Each sample corresponds to one timestamp for a specific agent, providing the
    requested number of camera views (default: 4 cameras). The world reference
    frame is the ego vehicle frame of that agent.
    """

    def __init__(
        self,
        *args,
        ROOT: str,
        depth_root: str,
        split: str,
        camera_ids: Sequence[int] = (0, 1, 2, 3),
        include_agents: Sequence[str] | None = None,
        max_scenes: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, split=split, **kwargs)
        self.root = Path(ROOT)
        self.depth_root = Path(depth_root)
        self.split = split
        self.camera_ids = [f"camera{cid}" for cid in camera_ids]
        self.include_agents = set(include_agents) if include_agents else None
        self.max_scenes = max_scenes
        self.dataset_name = "OPV2V"

        self._load_data()

        # OPV2V is rendered in CARLA, so we mark it as metric + synthetic
        self.is_metric_scale = True
        self.is_synthetic = True

    def _load_data(self):
        split_root = self.root / self.split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        scenes: List[Dict] = []
        for sequence_dir in sorted(d for d in split_root.iterdir() if d.is_dir()):
            for agent_dir in sorted(d for d in sequence_dir.iterdir() if d.is_dir()):
                agent_name = agent_dir.name
                if self.include_agents and agent_name not in self.include_agents:
                    continue

                yaml_files = sorted(agent_dir.glob("*.yaml"))
                for yaml_path in yaml_files:
                    frame_id = yaml_path.stem
                    if not frame_id.isdigit():
                        continue  # Skip auxiliary YAMLs like data_protocol
                    scenes.append(
                        dict(
                            sequence=sequence_dir.name,
                            agent=agent_name,
                            frame=frame_id,
                            yaml_path=yaml_path,
                            image_dir=agent_dir,
                        )
                    )
                    if self.max_scenes and len(scenes) >= self.max_scenes:
                        break
                if self.max_scenes and len(scenes) >= self.max_scenes:
                    break
            if self.max_scenes and len(scenes) >= self.max_scenes:
                break

        if not scenes:
            raise RuntimeError(
                f"No OPV2V scenes found in split {self.split} under {split_root}"
            )

        self.scenes = scenes
        self.num_of_scenes = len(scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_info = self.scenes[sampled_idx]
        frame_meta = load_frame_metadata(scene_info["yaml_path"])
        camera_poses = get_camera_poses_in_ego(frame_meta)

        available_cams = [
            cam_key for cam_key in self.camera_ids if cam_key in frame_meta
        ]
        if len(available_cams) < num_views_to_sample:
            raise ValueError(
                f"Requested {num_views_to_sample} views but only "
                f"{len(available_cams)} are available for frame {scene_info}"
            )

        idx_perm = self._rng.permutation(len(available_cams))
        selected_cam_keys = [
            available_cams[i] for i in idx_perm[:num_views_to_sample]
        ]

        views = []
        for cam_key in selected_cam_keys:
            cam_idx = cam_key.replace("camera", "")
            img_path = scene_info["image_dir"] / f"{scene_info['frame']}_{cam_key}.png"
            depth_path = (
                self.depth_root
                / self.split
                / scene_info["sequence"]
                / scene_info["agent"]
                / f"{scene_info['frame']}_{cam_key}_depth.npy"
            )

            if not img_path.exists():
                raise FileNotFoundError(f"Image missing: {img_path}")
            if not depth_path.exists():
                raise FileNotFoundError(f"Depth map missing: {depth_path}")

            image = Image.open(img_path).convert("RGB")
            depthmap = np.load(depth_path).astype(np.float32)
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

            intrinsics = np.array(frame_meta[cam_key]["intrinsic"], dtype=np.float32)
            camera_pose = camera_poses[cam_key].astype(np.float32)
            camera_pose = _convert_pose_to_opencv(camera_pose)

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=intrinsics,
                additional_quantities=None,
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset=self.dataset_name,
                    label=os.path.join(scene_info["sequence"], scene_info["agent"]),
                    instance=os.path.join(scene_info["frame"], cam_key),
                )
            )

        return views


class OPV2VCoopDataset(BaseDataset):
    """
    Multi-agent variant that loads the same timestamp across several vehicles and
    expresses every camera in the main agent's ego (LiDAR) frame.
    """

    def __init__(
        self,
        *args,
        ROOT: str,
        depth_root: str,
        split: str,
        camera_ids: Sequence[int] = (0, 1, 2, 3),
        include_agents: Sequence[str] | None = None,
        main_agent: str | None = None,
        main_agent_policy: str = "first",
        min_agents: int = 2,
        max_scenes: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, split=split, **kwargs)
        self.root = Path(ROOT)
        self.depth_root = Path(depth_root)
        self.split = split
        self.camera_ids = [f"camera{cid}" for cid in camera_ids]
        self.include_agents = set(include_agents) if include_agents else None
        self.main_agent = main_agent
        self.main_agent_policy = main_agent_policy
        self.min_agents = max(2, int(min_agents))
        self.max_scenes = max_scenes
        self.dataset_name = "OPV2VCoop"

        self._load_data()

        self.is_metric_scale = True
        self.is_synthetic = True

    def _load_data(self):
        split_root = self.root / self.split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        scenes: List[Dict] = []
        for sequence_dir in sorted(d for d in split_root.iterdir() if d.is_dir()):
            agent_dirs = [d for d in sequence_dir.iterdir() if d.is_dir()]
            if self.include_agents:
                agent_dirs = [d for d in agent_dirs if d.name in self.include_agents]
            if len(agent_dirs) < self.min_agents:
                continue

            frame_to_agents: Dict[str, List[str]] = {}
            for agent_dir in agent_dirs:
                for yaml_path in agent_dir.glob("*.yaml"):
                    frame_id = yaml_path.stem
                    if not frame_id.isdigit():
                        continue
                    frame_to_agents.setdefault(frame_id, []).append(agent_dir.name)

            for frame_id, agents in sorted(frame_to_agents.items()):
                if len(agents) < self.min_agents:
                    continue
                agents_sorted = sorted(agents)
                agent_dir_map = {agent_id: sequence_dir / agent_id for agent_id in agents_sorted}
                scenes.append(
                    dict(
                        sequence=sequence_dir.name,
                        frame=frame_id,
                        agents=agents_sorted,
                        agent_dirs=agent_dir_map,
                    )
                )
                if self.max_scenes and len(scenes) >= self.max_scenes:
                    break
            if self.max_scenes and len(scenes) >= self.max_scenes:
                break

        if not scenes:
            raise RuntimeError(
                f"No cooperative OPV2V scenes found in split {self.split} under {split_root}"
            )

        self.scenes = scenes
        self.num_of_scenes = len(scenes)

    def _choose_main_agent(self, agents: List[str]) -> str:
        if self.main_agent and self.main_agent in agents:
            return self.main_agent
        if self.main_agent_policy == "random":
            return agents[int(self._rng.integers(0, len(agents)))]
        # Default: deterministic first (sorted)
        return sorted(agents)[0]

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_info = self.scenes[sampled_idx]
        agents = sorted(scene_info["agents"])
        frame_id = scene_info["frame"]
        sequence = scene_info["sequence"]
        agent_dirs = scene_info["agent_dirs"]

        frame_meta_by_agent = {}
        for agent_id in agents:
            yaml_path = agent_dirs[agent_id] / f"{frame_id}.yaml"
            if not yaml_path.exists():
                raise FileNotFoundError(f"YAML missing: {yaml_path}")
            frame_meta_by_agent[agent_id] = load_frame_metadata(yaml_path)

        main_agent = self._choose_main_agent(agents)
        T_world_main = cords_to_pose(frame_meta_by_agent[main_agent]["lidar_pose"])
        T_main_world = np.linalg.inv(T_world_main)

        available_cams = []
        for agent_id in agents:
            meta = frame_meta_by_agent[agent_id]
            for cam_key in self.camera_ids:
                if cam_key not in meta:
                    continue
                img_path = agent_dirs[agent_id] / f"{frame_id}_{cam_key}.png"
                depth_path = (
                    self.depth_root
                    / self.split
                    / sequence
                    / agent_id
                    / f"{frame_id}_{cam_key}_depth.npy"
                )
                if not img_path.exists() or not depth_path.exists():
                    continue

                cam_pose_world = cords_to_pose(meta[cam_key]["cords"])
                cam_pose_main = T_main_world @ cam_pose_world
                available_cams.append(
                    dict(
                        agent_id=agent_id,
                        cam_key=cam_key,
                        img_path=img_path,
                        depth_path=depth_path,
                        intrinsics=np.array(meta[cam_key]["intrinsic"], dtype=np.float32),
                        camera_pose=cam_pose_main.astype(np.float32),
                    )
                )

        if len(available_cams) < num_views_to_sample:
            raise ValueError(
                f"Requested {num_views_to_sample} views but only "
                f"{len(available_cams)} available for frame {scene_info}"
            )

        idx_perm = self._rng.permutation(len(available_cams))
        selected_entries = [available_cams[i] for i in idx_perm[:num_views_to_sample]]

        views = []
        for entry in selected_entries:
            image = Image.open(entry["img_path"]).convert("RGB")
            depthmap = np.load(entry["depth_path"]).astype(np.float32)
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

            camera_pose = _convert_pose_to_opencv(entry["camera_pose"])
            intrinsics = entry["intrinsics"]

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=intrinsics,
                additional_quantities=None,
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset=self.dataset_name,
                    label=os.path.join(sequence, main_agent),
                    instance=os.path.join(
                        frame_id, f"{entry['cam_key']}_{entry['agent_id']}"
                    ),
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--depth_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--resolution", type=int, nargs=2, default=(518, 392))
    parser.add_argument("--max_scenes", type=int, default=10)
    parser.add_argument("--viz", action="store_true")
    return parser


if __name__ == "__main__":
    import rerun as rr
    from tqdm import trange

    from mapanything.datasets.base.base_dataset import view_name
    from mapanything.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(parser)
    args = parser.parse_args()

    dataset = OPV2VDataset(
        num_views=args.num_views,
        split=args.split,
        covisibility_thres=None,
        resolution=tuple(args.resolution),
        principal_point_centered=False,
        transform="imgnorm",
        data_norm_type="dinov2",
        ROOT=args.root,
        depth_root=args.depth_root,
        max_scenes=args.max_scenes,
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "OPV2V_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        for idx in trange(min(len(dataset), 5)):
            views = dataset[idx]
            for view in views:
                view_id = view_name(view)
                rr.log(
                    f"{view_id}/points",
                    rr.Points3D(view["pts3d"].reshape(-1, 3), colors=[255, 255, 255]),
                )
