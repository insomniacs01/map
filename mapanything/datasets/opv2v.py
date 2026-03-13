import hashlib
import json
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
    get_vehicle_bboxes_in_ego,
    load_frame_metadata,
)
from mapanything.datasets.base.base_dataset import BaseDataset


def _convert_pose_to_opencv(pose: np.ndarray) -> np.ndarray:
    pose_cv = pose.copy()
    basis = CARLA_TO_CAMERA_CV[:3, :3]
    pose_cv[:3, :3] = basis @ pose[:3, :3] @ basis.T
    pose_cv[:3, 3] = basis @ pose[:3, 3]
    return pose_cv


def _convert_points_to_opencv(points: np.ndarray) -> np.ndarray:
    basis = CARLA_TO_CAMERA_CV[:3, :3]
    return np.asarray(points) @ basis.T


def _convert_rotation_to_opencv(rotation: np.ndarray) -> np.ndarray:
    basis = CARLA_TO_CAMERA_CV[:3, :3]
    return basis @ rotation @ basis.T


def _get_vehicle_boxes_in_opencv(frame_meta: Dict) -> List[Dict]:
    vehicle_bboxes = get_vehicle_bboxes_in_ego(frame_meta, max_range=None)
    vehicle_boxes = []
    for bbox in vehicle_bboxes.values():
        vehicle_boxes.append(
            {
                "center": _convert_points_to_opencv(
                    np.asarray(bbox["center"], dtype=np.float32)
                ).astype(np.float32),
                "rotation": _convert_rotation_to_opencv(
                    np.asarray(bbox["rotation"], dtype=np.float32)
                ).astype(np.float32),
                "extent": np.asarray(bbox["extent"], dtype=np.float32),
            }
        )
    return vehicle_boxes


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
                        continue
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
    expresses every camera in the main agent's ego frame.
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
        min_agents: int = 1,
        min_num_views: int = 4,
        max_num_views: int | None = None,
        structured_sampling: bool = False,
        agents_per_sample: int | None = None,
        views_per_agent: int | None = None,
        agent_selection_policy: str = "random",
        max_agent_distance: float | None = None,
        require_complete_rig: bool = False,
        shuffle_views: bool = True,
        emit_identity_metadata: bool = False,
        max_scenes: int | None = None,
        metadata_cache_dir: str | None = None,
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
        self.structured_sampling = bool(structured_sampling)
        self.views_per_agent = (
            int(views_per_agent) if views_per_agent is not None else len(self.camera_ids)
        )
        self.agents_per_sample = (
            int(agents_per_sample) if agents_per_sample is not None else None
        )
        self.agent_selection_policy = agent_selection_policy
        self.max_agent_distance = (
            float(max_agent_distance) if max_agent_distance is not None else None
        )
        self.require_complete_rig = bool(require_complete_rig)
        self.shuffle_views = bool(shuffle_views)
        self.emit_identity_metadata = bool(emit_identity_metadata)
        self.min_dynamic_views = max(1, int(min_num_views))
        self.requested_max_num_views = max_num_views
        self.max_scenes = max_scenes
        self.dataset_name = "OPV2VCoop"
        self.metadata_cache_dir = (
            Path(metadata_cache_dir) if metadata_cache_dir is not None else None
        )

        self.min_agents = max(1, int(min_agents))
        if self.structured_sampling and self.agents_per_sample is not None:
            self.min_agents = max(self.min_agents, self.agents_per_sample)

        self.allow_variable_view_count = False
        self.min_num_views_allowed = self.min_dynamic_views

        self._load_data()

        self.is_metric_scale = True
        self.is_synthetic = True
        self.max_views_per_scene = max(len(scene["agents"]) for scene in self.scenes) * len(
            self.camera_ids
        )

        if self.structured_sampling:
            if self.agents_per_sample is None:
                if isinstance(self.num_views, int):
                    self.agents_per_sample = max(1, self.num_views // self.views_per_agent)
                else:
                    self.agents_per_sample = self.min_agents
            if isinstance(self.num_views, int):
                self.num_views = self.agents_per_sample * self.views_per_agent
            self.min_num_views_allowed = self.num_views
        else:
            requested_num_views = max(self.num_views) if isinstance(self.num_views, list) else int(self.num_views)
            max_dynamic_views = requested_num_views
            if self.requested_max_num_views is not None:
                max_dynamic_views = min(max_dynamic_views, int(self.requested_max_num_views))
            max_dynamic_views = min(max_dynamic_views, self.max_views_per_scene)
            self.max_dynamic_views = max(1, max_dynamic_views)
            self.min_dynamic_views = min(self.min_dynamic_views, self.max_dynamic_views)
            self.allow_variable_view_count = bool(self.variable_num_views) or (
                self.min_dynamic_views != self.max_dynamic_views
            )
            if self.allow_variable_view_count:
                self.num_views = list(
                    range(self.min_dynamic_views, self.max_dynamic_views + 1)
                )
                self.min_num_views_allowed = self.min_dynamic_views
            else:
                self.num_views = self.max_dynamic_views
                self.min_num_views_allowed = self.max_dynamic_views

    def _scene_cache_path(self) -> Path | None:
        if self.metadata_cache_dir is None:
            return None

        cache_dir = self.metadata_cache_dir / "opv2v_coop_scene_cache"
        cache_key_payload = {
            "version": 1,
            "split": self.split,
            "root": str(self.root),
            "depth_root": str(self.depth_root),
            "camera_ids": list(self.camera_ids),
            "include_agents": sorted(self.include_agents) if self.include_agents else None,
            "main_agent": self.main_agent,
            "main_agent_policy": self.main_agent_policy,
            "min_agents": self.min_agents,
            "structured_sampling": self.structured_sampling,
            "agents_per_sample": self.agents_per_sample,
            "views_per_agent": self.views_per_agent,
            "max_agent_distance": self.max_agent_distance,
            "require_complete_rig": self.require_complete_rig,
            "max_scenes": self.max_scenes,
        }
        cache_key = hashlib.sha1(
            json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return cache_dir / f"{self.split}_{cache_key}.json"

    def _rebuild_scene_records(self, raw_scenes: List[Dict]) -> List[Dict]:
        scenes: List[Dict] = []
        for item in raw_scenes:
            sequence = item["sequence"]
            agents = list(item["agents"])
            scenes.append(
                dict(
                    sequence=sequence,
                    frame=item["frame"],
                    agents=agents,
                    agent_dirs={agent_id: self.root / self.split / sequence / agent_id for agent_id in agents},
                )
            )
        return scenes

    def _load_cached_scenes(self) -> List[Dict] | None:
        cache_path = self._scene_cache_path()
        if cache_path is None or not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text())
            raw_scenes = payload.get("scenes", [])
            if not raw_scenes:
                return None
            scenes = self._rebuild_scene_records(raw_scenes)
            print(
                f"OPV2VCoopDataset[{self.split}] loaded {len(scenes)} cached scenes from {cache_path}"
            )
            return scenes
        except Exception as exc:
            print(
                f"OPV2VCoopDataset[{self.split}] failed to load cache {cache_path}: {exc}"
            )
            return None

    def _save_cached_scenes(
        self, scenes: List[Dict], raw_scene_count: int, dropped_scene_count: int
    ) -> None:
        cache_path = self._scene_cache_path()
        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_scenes": len(scenes),
            "raw_scene_count": raw_scene_count,
            "dropped_scene_count": dropped_scene_count,
            "scenes": [
                {
                    "sequence": scene["sequence"],
                    "frame": scene["frame"],
                    "agents": list(scene["agents"]),
                }
                for scene in scenes
            ],
        }
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(tmp_path, cache_path)
            print(
                f"OPV2VCoopDataset[{self.split}] wrote {len(scenes)} cached scenes to {cache_path}"
            )
        except Exception as exc:
            print(
                f"OPV2VCoopDataset[{self.split}] failed to write cache {cache_path}: {exc}"
            )

    def _load_data(self):
        split_root = self.root / self.split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        cached_scenes = self._load_cached_scenes()
        if cached_scenes is not None:
            self.scenes = cached_scenes
            self.num_of_scenes = len(cached_scenes)
            return

        scenes: List[Dict] = []
        raw_scene_count = 0
        dropped_scene_count = 0
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
                raw_scene_count += 1
                agents_sorted = sorted(agents)
                agent_dir_map = {agent_id: sequence_dir / agent_id for agent_id in agents_sorted}
                if not self._scene_is_usable_after_filter(
                    sequence=sequence_dir.name,
                    frame_id=frame_id,
                    agents=agents_sorted,
                    agent_dirs=agent_dir_map,
                ):
                    dropped_scene_count += 1
                    continue
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

        if raw_scene_count > 0 and dropped_scene_count > 0:
            print(
                f"OPV2VCoopDataset[{self.split}] kept {len(scenes)}/{raw_scene_count} "
                f"scenes after coop filtering (dropped {dropped_scene_count})."
            )

        self.scenes = scenes
        self.num_of_scenes = len(scenes)
        self._save_cached_scenes(
            scenes=scenes,
            raw_scene_count=raw_scene_count,
            dropped_scene_count=dropped_scene_count,
        )

    def _target_agents_per_scene(self) -> int:
        if self.structured_sampling:
            return self.agents_per_sample or self.min_agents
        return self.min_agents

    def _required_views_per_agent(self) -> int:
        return self.views_per_agent if self.structured_sampling else 1

    def _prefilter_main_agent(self, agents: List[str]) -> str | None:
        if self.main_agent and self.main_agent in agents:
            return self.main_agent
        if self.main_agent_policy == "random":
            return None
        return sorted(agents)[0]

    def _scene_is_usable_after_filter(
        self,
        sequence: str,
        frame_id: str,
        agents: List[str],
        agent_dirs: Dict[str, Path],
    ) -> bool:
        main_agent = self._prefilter_main_agent(agents)
        if main_agent is None:
            return True

        target_agents = self._target_agents_per_scene()
        required_views_per_agent = self._required_views_per_agent()

        if (
            self.max_agent_distance is None
            and not self.require_complete_rig
            and target_agents <= self.min_agents
            and required_views_per_agent <= 1
        ):
            return True

        frame_meta_by_agent = {}
        for agent_id in agents:
            yaml_path = agent_dirs[agent_id] / f"{frame_id}.yaml"
            if not yaml_path.exists():
                return False
            frame_meta_by_agent[agent_id] = load_frame_metadata(yaml_path)

        T_world_main = cords_to_pose(frame_meta_by_agent[main_agent]["lidar_pose"])
        T_main_world = np.linalg.inv(T_world_main)
        main_position = T_world_main[:3, 3]
        valid_agent_count = 0

        for agent_id in agents:
            meta = frame_meta_by_agent[agent_id]
            agent_pose_world = cords_to_pose(meta["lidar_pose"])
            agent_distance = float(np.linalg.norm(agent_pose_world[:3, 3] - main_position))
            if (
                self.max_agent_distance is not None
                and agent_id != main_agent
                and agent_distance > self.max_agent_distance
            ):
                continue

            entries = self._build_agent_entries(
                sequence=sequence,
                frame_id=frame_id,
                agent_id=agent_id,
                agent_dir=agent_dirs[agent_id],
                meta=meta,
                T_main_world=T_main_world,
                agent_distance=agent_distance,
            )
            if len(entries) < required_views_per_agent:
                continue
            if self.require_complete_rig and len(entries) < self.views_per_agent:
                continue

            valid_agent_count += 1
            if valid_agent_count >= target_agents:
                return True

        return False

    def _choose_main_agent(self, agents: List[str]) -> str:
        if self.main_agent and self.main_agent in agents:
            return self.main_agent
        if self.main_agent_policy == "random":
            return agents[int(self._rng.integers(0, len(agents)))]
        return sorted(agents)[0]

    def _build_agent_entries(
        self,
        sequence: str,
        frame_id: str,
        agent_id: str,
        agent_dir: Path,
        meta: Dict,
        T_main_world: np.ndarray,
        agent_distance: float,
    ) -> List[Dict]:
        entries: List[Dict] = []
        for camera_index, cam_key in enumerate(self.camera_ids):
            if cam_key not in meta:
                continue
            img_path = agent_dir / f"{frame_id}_{cam_key}.png"
            depth_path = (
                self.depth_root / self.split / sequence / agent_id / f"{frame_id}_{cam_key}_depth.npy"
            )
            if not img_path.exists() or not depth_path.exists():
                continue

            cam_pose_world = cords_to_pose(meta[cam_key]["cords"])
            cam_pose_main = T_main_world @ cam_pose_world
            entries.append(
                dict(
                    agent_id=agent_id,
                    agent_distance=float(agent_distance),
                    cam_key=cam_key,
                    camera_index=int(camera_index),
                    img_path=img_path,
                    depth_path=depth_path,
                    intrinsics=np.array(meta[cam_key]["intrinsic"], dtype=np.float32),
                    camera_pose=cam_pose_main.astype(np.float32),
                )
            )
        return entries

    def _order_candidates(self, candidates: List[Dict], main_agent: str) -> List[Dict]:
        if self.agent_selection_policy == "nearest":
            return sorted(
                candidates,
                key=lambda item: (
                    item["agent_id"] != main_agent,
                    item["agent_distance"],
                    item["agent_id"],
                ),
            )
        if self.agent_selection_policy == "farthest":
            return sorted(
                candidates,
                key=lambda item: (
                    item["agent_id"] != main_agent,
                    -item["agent_distance"],
                    item["agent_id"],
                ),
            )
        if self.agent_selection_policy == "first":
            return sorted(
                candidates,
                key=lambda item: (
                    item["agent_id"] != main_agent,
                    item["agent_id"],
                ),
            )
        ordered = list(candidates)
        if len(ordered) > 1:
            perm = self._rng.permutation(len(ordered))
            ordered = [ordered[idx] for idx in perm]
            ordered.sort(key=lambda item: item["agent_id"] != main_agent)
        return ordered

    def _select_structured_entries(
        self,
        candidates: List[Dict],
        main_agent: str,
    ) -> List[Dict]:
        ordered = self._order_candidates(candidates, main_agent)
        target_agents = self.agents_per_sample or self.min_agents
        selected_agents = ordered[:target_agents]
        if len(selected_agents) < target_agents:
            raise ValueError(
                f"Need {target_agents} agents after filtering but got {len(selected_agents)}"
            )

        selected_entries: List[Dict] = []
        for agent_index, agent_bundle in enumerate(selected_agents):
            entries = sorted(agent_bundle["entries"], key=lambda item: item["camera_index"])
            if len(entries) < self.views_per_agent:
                raise ValueError(
                    f"Agent {agent_bundle['agent_id']} has only {len(entries)} views, "
                    f"need {self.views_per_agent}"
                )
            entries = entries[: self.views_per_agent]
            for entry in entries:
                enriched = dict(entry)
                enriched["agent_index"] = int(agent_index)
                selected_entries.append(enriched)

        if self.shuffle_views and len(selected_entries) > 1:
            perm = self._rng.permutation(len(selected_entries))
            selected_entries = [selected_entries[idx] for idx in perm]

        return selected_entries

    def _select_unstructured_entries(
        self,
        available_entries: List[Dict],
        num_views_to_sample: int,
        main_agent: str,
    ) -> List[Dict]:
        total_available = len(available_entries)
        if total_available < self.min_dynamic_views:
            raise ValueError(
                f"Need at least {self.min_dynamic_views} views but got {total_available}"
            )

        if self.allow_variable_view_count:
            max_allowed = min(self.max_dynamic_views, total_available)
            min_allowed = min(self.min_dynamic_views, max_allowed)
            actual_num_views = int(self._rng.integers(min_allowed, max_allowed + 1))
        else:
            if num_views_to_sample > total_available:
                raise ValueError(
                    f"Requested {num_views_to_sample} views but only {total_available} available"
                )
            actual_num_views = num_views_to_sample

        idx_perm = self._rng.permutation(total_available)
        selected_entries = [available_entries[i] for i in idx_perm[:actual_num_views]]

        if self.emit_identity_metadata:
            ordered_agent_ids = sorted({entry["agent_id"] for entry in selected_entries})
            if main_agent in ordered_agent_ids:
                ordered_agent_ids = [main_agent] + [aid for aid in ordered_agent_ids if aid != main_agent]
            agent_to_index = {agent_id: idx for idx, agent_id in enumerate(ordered_agent_ids)}
            for entry in selected_entries:
                entry["agent_index"] = int(agent_to_index[entry["agent_id"]])

        return selected_entries

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
        vehicle_boxes = _get_vehicle_boxes_in_opencv(frame_meta_by_agent[main_agent])

        candidate_agents: List[Dict] = []
        available_entries: List[Dict] = []
        main_position = T_world_main[:3, 3]
        required_views_per_agent = self.views_per_agent if self.structured_sampling else 1

        for agent_id in agents:
            meta = frame_meta_by_agent[agent_id]
            agent_pose_world = cords_to_pose(meta["lidar_pose"])
            agent_distance = float(np.linalg.norm(agent_pose_world[:3, 3] - main_position))
            if (
                self.max_agent_distance is not None
                and agent_id != main_agent
                and agent_distance > self.max_agent_distance
            ):
                continue

            entries = self._build_agent_entries(
                sequence=sequence,
                frame_id=frame_id,
                agent_id=agent_id,
                agent_dir=agent_dirs[agent_id],
                meta=meta,
                T_main_world=T_main_world,
                agent_distance=agent_distance,
            )
            if len(entries) < required_views_per_agent:
                continue
            if self.require_complete_rig and len(entries) < self.views_per_agent:
                continue

            candidate_agents.append(
                {
                    "agent_id": agent_id,
                    "agent_distance": agent_distance,
                    "entries": entries,
                }
            )
            available_entries.extend(entries)

        if self.structured_sampling:
            selected_entries = self._select_structured_entries(candidate_agents, main_agent)
        else:
            selected_entries = self._select_unstructured_entries(
                available_entries,
                num_views_to_sample=num_views_to_sample,
                main_agent=main_agent,
            )

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

            view = dict(
                img=image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_name,
                label=os.path.join(sequence, main_agent),
                instance=os.path.join(frame_id, f"{entry['cam_key']}_{entry['agent_id']}"),
                vehicle_boxes=vehicle_boxes,
            )
            if self.emit_identity_metadata or self.structured_sampling:
                view["agent_index"] = int(entry.get("agent_index", 0))
                view["camera_index"] = int(entry["camera_index"])
            views.append(view)

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
