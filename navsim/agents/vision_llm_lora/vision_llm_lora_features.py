from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from PIL import Image

from navsim.agents.vision_llm.vision_llm_features import (
    BoundingBox2DIndex,
    VisionLLMFeatureBuilder,
    VisionLLMTargetBuilder,
)
from navsim.common.dataclasses import AgentInput, Scene
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder

IMAGE_NORMALIZATION = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "qwen": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Qwen/Qwen3.5-4B preprocessor_config.json
}


class VisionLLMLoRAFeatureBuilder(VisionLLMFeatureBuilder):
    """Thin wrapper to keep LoRA feature builder customization isolated."""

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Two rows of 6 tiles plus an optional 13th image. Camera layout, thumbnail, tile size, channel order and
        normalization come from the config (docs/vision_llm_lora_findings.md F9, F10, F11). The config defaults
        reproduce the parent's output.
        :param agent_input: input dataclass
        :return: [13, 3, tile, tile], or [12, 3, tile, tile] with thumbnail="none"
        """
        tile = self._config.camera_width
        assert tile == self._config.camera_height, "camera tiles must be square"
        mean, std = IMAGE_NORMALIZATION[self._config.image_normalization]

        cameras = agent_input.cameras[-1]

        if self._config.camera_layout == "original":

            def crop(img):
                return img[316:-316, 512:-512]

            top = np.concatenate(
                [crop(cameras.cam_l0.image), crop(cameras.cam_f0.image), crop(cameras.cam_r0.image)], 1
            )
            bottom = np.concatenate(
                [crop(cameras.cam_l2.image), crop(cameras.cam_b0.image), crop(cameras.cam_r2.image)], 1
            )
            stitched_image = np.concatenate([top, bottom], axis=0)
            resized_image = cv2.resize(stitched_image, (6 * tile, 2 * tile), interpolation=cv2.INTER_LANCZOS4)

        elif self._config.camera_layout == "panorama":

            def row(images):
                # Centre crop to the 6:1 row aspect, then resize to 6 tiles x 1 tile.
                strip = np.concatenate(images, axis=1)
                height = round(strip.shape[1] / 6)
                y0 = (strip.shape[0] - height) // 2
                return cv2.resize(strip[y0 : y0 + height], (6 * tile, tile), interpolation=cv2.INTER_LANCZOS4)

            # Yaw decreases left to right through both rows: L0, F0, R0, then R1, B0, L1.
            front = row([cameras.cam_l0.image[:, 416:-416], cameras.cam_f0.image, cameras.cam_r0.image[:, 416:-416]])
            rear = row([cameras.cam_r1.image, cameras.cam_b0.image, cameras.cam_l1.image])
            resized_image = np.concatenate([front, rear], axis=0)

        else:
            raise ValueError(f"Unknown camera_layout: {self._config.camera_layout!r}")
        if self._config.swap_red_blue:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        thumbnail = self._config.thumbnail
        if thumbnail not in ("canvas", "front_zoom", "none"):
            raise ValueError(f"Unknown thumbnail: {thumbnail!r}")
        images = self.dynamic_preprocess(
            Image.fromarray(resized_image), image_size=tile, use_thumbnail=thumbnail == "canvas"
        )
        if thumbnail == "front_zoom":
            # F0 (fx 1545, cy 560, ~1.5 deg downward pitch): 14.2 deg wide, about -5.6 to +8.5 deg around the horizon,
            # i.e. lead vehicles and traffic lights straight ahead at ~9 px/deg.
            zoom = cv2.resize(cameras.cam_f0.image[290:674, 768:1152], (tile, tile), interpolation=cv2.INTER_LANCZOS4)
            if self._config.swap_red_blue:
                zoom = cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(zoom))

        transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        return torch.stack([transform(image) for image in images])


class VisionLLMLoRATargetBuilder(VisionLLMTargetBuilder):
    """Thin wrapper to keep LoRA target builder customization isolated."""


# Order of the sub-score rows in "pdm_scores" (as score_codebook_trajectories.py names the metrics).
PDM_SCORE_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
)


class PDMScoreTargetBuilder(AbstractTargetBuilder):
    """
    Per-candidate PDM sub-scores of a scene, {"pdm_scores": (len(PDM_SCORE_KEYS), K) float16}, for the sub-score heads
    (config.pdm_score_loss_weight > 0). Too expensive to compute per sample (~2 min per scene): they are scored offline
    (scripts/evaluation/score_codebook_trajectories.py) and written into the training cache under this builder's name
    (scripts/training/cache_pdm_score_targets.py), so training reads them like any cached target.
    """

    def get_unique_name(self) -> str:
        return "vision_llm_pdm_scores"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "PDM score targets are precomputed: run scripts/training/cache_pdm_score_targets.py on the training cache."
        )


# Dynamic agent classes kept as agent targets (annotation name -> class id); other boxes are static objects.
AUX_AGENT_CLASSES = {"vehicle": 0, "pedestrian": 1, "bicycle": 2}
# Map layers PDM's drivable-area check uses (PDMDrivableMap.from_simulation): roadblock polygons, the lane connectors
# inside roadblock connectors, intersections and car parks.
_DRIVABLE_POLYGON_LAYERS = (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA)
NO_LANE_HEADING = 255  # lane_heading_map value where no lane is drawn
_SHIFT = 4  # fractional bits of the OpenCV drawing coordinates (1/16 pixel)
_SUPERSAMPLE = 4  # filled shapes are drawn this many times finer, then averaged


class VisionLLMLoRAAuxTargetBuilder(AbstractTargetBuilder):
    """
    Perception targets of the current frame for auxiliary losses (no loss uses them yet). Computed from annotations and
    the map, no sensors; cached once by scripts/training/cache_aux_targets.py into a squashfs image with the training
    cache layout (<log>/<token>/vision_llm_aux_target.gz).

    BEV grid in the ego frame of the current frame: row i covers x = x_min + (i + 0.5) * resolution (forward), column j
    covers y = y_max - (j + 0.5) * resolution (y points left, so column 0 is the left edge). Defaults: x in [-16, 64) m,
    y in [-32, 32) m at 0.5 m -> 160 x 128; 4 s trajectories at speed reach beyond the 32 m of TransFuser's BEV.

    Targets:
    - bev_semantic_map (H, W) uint8: classes of `semantic_classes` (TransFuser's, from VisionLLMLoRAConfig), later
      classes drawn over earlier ones, 0 = none
    - drivable_area_map (H, W) uint8: 1 on the surface PDM's drivable-area compliance accepts
    - lane_heading_map (H, W) uint8: travel direction of lanes and lane connectors relative to ego heading, in
      `heading_bins` bins over [-pi, pi), NO_LANE_HEADING off lanes; lines `lane_width` m wide
    - agent_states (max_agents, 7) float32: x, y, heading, length, width, vx, vy of the nearest dynamic agents
    - agent_classes (max_agents,) int8: AUX_AGENT_CLASSES id, -1 for empty rows
    """

    def __init__(
        self,
        semantic_classes: Dict[int, Tuple[str, list]],
        x_range: Tuple[float, float] = (-16.0, 64.0),
        y_range: Tuple[float, float] = (-32.0, 32.0),
        resolution: float = 0.5,
        max_agents: int = 40,
        heading_bins: int = 36,
        lane_width: float = 3.0,
    ):
        self._semantic_classes = semantic_classes
        self._x_min, self._x_max = x_range
        self._y_min, self._y_max = y_range
        self._resolution = resolution
        self._shape = (round((self._x_max - self._x_min) / resolution), round((self._y_max - self._y_min) / resolution))
        self._max_agents = max_agents
        self._heading_bins = heading_bins
        self._lane_thickness = max(1, round(lane_width / resolution))
        # Map objects reaching any grid cell: distance from ego to the farthest grid corner.
        self._radius = float(np.hypot(max(abs(self._x_min), abs(self._x_max)), max(abs(self._y_min), abs(self._y_max))))

    def get_unique_name(self) -> str:
        return "vision_llm_aux_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        ego_pose = StateSE2(*frame.ego_status.ego_pose)
        annotations = frame.annotations
        map_api = scene.map_api

        semantic = np.zeros(self._shape, dtype=np.uint8)
        for label, (entity_type, layers) in self._semantic_classes.items():
            if entity_type == "polygon":
                mask = self._polygon_mask(self._map_polygons(map_api, ego_pose, layers), ego_pose)
            elif entity_type == "linestring":
                mask = self._centerline_mask(map_api, ego_pose, layers)
            else:
                mask = self._box_mask(annotations, layers)
            semantic[mask] = label

        return {
            "bev_semantic_map": torch.from_numpy(semantic),
            "drivable_area_map": torch.from_numpy(self._drivable_area_mask(map_api, ego_pose).astype(np.uint8)),
            "lane_heading_map": torch.from_numpy(self._lane_heading_map(map_api, ego_pose)),
            **self._agent_targets(annotations),
        }

    # ── geometry ──────────────────────────────────────────────────────────────────────────────────────────────────

    def _to_pixels(self, xy: np.ndarray, scale: int = 1) -> np.ndarray:
        """
        (N, 2) ego-frame x, y in metres -> (N, 2) int32 OpenCV points (column, row) on a grid `scale` times finer, in
        fixed point with _SHIFT fractional bits. OpenCV puts integer coordinates at pixel centres while cell k spans
        [k, k + 1) here, hence -0.5.
        """
        col = (self._y_max - xy[:, 1]) / self._resolution * scale - 0.5
        row = (xy[:, 0] - self._x_min) / self._resolution * scale - 0.5
        return (np.stack([col, row], axis=-1) * (1 << _SHIFT)).round().astype(np.int32)

    def _fill(self, outlines: list) -> np.ndarray:
        """
        Boolean (H, W) mask of the cells at least half covered by the (N, 2) ego-frame outlines. Filled at _SUPERSAMPLE
        times the resolution and averaged: cv2.fillPoly also fills every pixel an edge touches, which at grid resolution
        grows each shape by up to one cell.
        """
        fine = np.zeros((self._shape[0] * _SUPERSAMPLE, self._shape[1] * _SUPERSAMPLE), dtype=np.uint8)
        for outline in outlines:  # one call per shape: several contours in one call follow the even-odd rule
            cv2.fillPoly(fine, [self._to_pixels(outline, _SUPERSAMPLE)], color=1, shift=_SHIFT)
        blocks = fine.reshape(self._shape[0], _SUPERSAMPLE, self._shape[1], _SUPERSAMPLE)
        return blocks.mean(axis=(1, 3)) >= 0.5

    @staticmethod
    def _to_local(xy: np.ndarray, origin: StateSE2) -> np.ndarray:
        """(N, 2) global x, y -> ego frame of origin."""
        c, s = np.cos(origin.heading), np.sin(origin.heading)
        d = xy - np.array([origin.x, origin.y])
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=-1)

    def _map_polygons(self, map_api, ego_pose: StateSE2, layers: Sequence[SemanticMapLayer]) -> list:
        objects = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self._radius, layers=list(layers))
        return [obj.polygon for layer in layers for obj in objects[layer]]

    def _polygon_mask(self, polygons: list, origin: StateSE2) -> np.ndarray:
        """Filled mask of global-frame polygons, drawn in the ego frame of origin."""
        outlines = []
        for polygon in polygons:
            for part in getattr(polygon, "geoms", [polygon]):  # MultiPolygon -> its polygons
                outlines.append(self._to_local(np.asarray(part.exterior.coords)[:, :2], origin))
        return self._fill(outlines)

    def _drivable_area_mask(self, map_api, ego_pose: StateSE2) -> np.ndarray:
        polygons = self._map_polygons(map_api, ego_pose, _DRIVABLE_POLYGON_LAYERS)
        connectors = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._radius, layers=[SemanticMapLayer.ROADBLOCK_CONNECTOR]
        )[SemanticMapLayer.ROADBLOCK_CONNECTOR]
        polygons += [lane.polygon for connector in connectors for lane in connector.interior_edges]
        return self._polygon_mask(polygons, ego_pose)

    def _centerline_mask(self, map_api, ego_pose: StateSE2, layers: Sequence[SemanticMapLayer]) -> np.ndarray:
        mask = np.zeros(self._shape, dtype=np.uint8)
        objects = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self._radius, layers=list(layers))
        for layer in layers:
            for obj in objects[layer]:
                points = self._to_local(np.asarray(obj.baseline_path.linestring.coords)[:, :2], ego_pose)
                cv2.polylines(mask, [self._to_pixels(points)], isClosed=False, color=1, thickness=1, shift=_SHIFT)
        return mask > 0

    def _lane_heading_map(self, map_api, ego_pose: StateSE2) -> np.ndarray:
        heading_map = np.full(self._shape, NO_LANE_HEADING, dtype=np.uint8)
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        objects = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self._radius, layers=layers)
        for layer in layers:
            for obj in objects[layer]:
                path = obj.baseline_path.discrete_path
                xy = self._to_local(np.array([[p.x, p.y] for p in path]), ego_pose)
                headings = np.array([p.heading for p in path]) - ego_pose.heading
                bins = (((headings + np.pi) % (2 * np.pi)) / (2 * np.pi) * self._heading_bins).astype(int)
                bins = np.clip(bins, 0, self._heading_bins - 1)
                pixels = self._to_pixels(xy)
                for k in range(len(pixels) - 1):
                    cv2.line(heading_map, tuple(map(int, pixels[k])), tuple(map(int, pixels[k + 1])),
                             color=int(bins[k]), thickness=self._lane_thickness, shift=_SHIFT)
        return heading_map

    def _box_mask(self, annotations, layers) -> np.ndarray:
        outlines = []
        for name, box in zip(annotations.names, annotations.boxes):
            if tracked_object_types[name] in layers:
                oriented = OrientedBox(
                    StateSE2(box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y], box[BoundingBoxIndex.HEADING]),
                    box[BoundingBoxIndex.LENGTH], box[BoundingBoxIndex.WIDTH], box[BoundingBoxIndex.HEIGHT],
                )
                outlines.append(np.asarray(oriented.geometry.exterior.coords)[:, :2])  # annotations: ego frame
        return self._fill(outlines)

    def _agent_targets(self, annotations) -> Dict[str, torch.Tensor]:
        rows, classes = [], []
        for name, box, velocity in zip(annotations.names, annotations.boxes, annotations.velocity_3d):
            if name in AUX_AGENT_CLASSES:
                rows.append([box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y], box[BoundingBoxIndex.HEADING],
                             box[BoundingBoxIndex.LENGTH], box[BoundingBoxIndex.WIDTH], velocity[0], velocity[1]])
                classes.append(AUX_AGENT_CLASSES[name])
        states = np.zeros((self._max_agents, 7), dtype=np.float32)
        agent_classes = np.full(self._max_agents, -1, dtype=np.int8)
        if rows:
            rows_arr = np.asarray(rows, dtype=np.float32)
            nearest = np.argsort(np.linalg.norm(rows_arr[:, :2], axis=-1))[: self._max_agents]
            states[: len(nearest)] = rows_arr[nearest]
            agent_classes[: len(nearest)] = np.asarray(classes, dtype=np.int8)[nearest]
        return {"agent_states": torch.from_numpy(states), "agent_classes": torch.from_numpy(agent_classes)}
