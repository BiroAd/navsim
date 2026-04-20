from enum import IntEnum
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, MapObject, SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import affinity
from shapely.geometry import LineString, Polygon
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from navsim.agents.vision_llm.vision_llm_config import VisionLLMConfig
from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class VisionLLMFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for VisionLLM."""

    def __init__(self, config: VisionLLMConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of VisionLLM
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "vision_llm_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features['velocity'] = torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32)
        features['acceleration'] = torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        features['driving_command'] = torch.argmax(torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.int16))
        #print(f"Driving command: {features['driving_command']}")

        '''if features['driving_command'] == 3:
            image_file_name = "camera_image.png"
    
            # Convert from (C, H, W) to (H, W, C) format
            camera_array = camera_tensor.permute(1, 2, 0).numpy()
            
            # Scale back from [0, 1] float to [0, 255] uint8
            camera_array = (camera_array * 255).astype(np.uint8)
            
            # Now create the image
            image = Image.fromarray(camera_array)
            image.save(image_file_name)
            print(f"Image saved as {image_file_name}")'''

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        def crop(img):
            return img[316:-316, 512:-512]

        # Crop to ensure 4:1 aspect ratio

        l0 = crop(cameras.cam_l0.image)
        f0 = crop(cameras.cam_f0.image)
        r0 = crop(cameras.cam_r0.image)

        l2 = crop(cameras.cam_l2.image)
        b0 = crop(cameras.cam_b0.image)
        r2 = crop(cameras.cam_r2.image)

        # stitch l0, f0, r0 images
        stitched_image1 = np.concatenate([l0, f0, r0], axis=1)
        stitched_image2 = np.concatenate([l2, b0, r2], axis=1)
        stitched_image = np.concatenate([stitched_image1, stitched_image2], axis=0)


        #stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)

        #stitched_tensor = torch.from_numpy(stitched_image).permute(2, 0, 1).float() / 255.0
        #print(tensor_image.shape)

        resized_image = cv2.resize(stitched_image, (672, 224), interpolation=cv2.INTER_LANCZOS4)
        resized_image_pil = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)) 
        #tensor_image = transforms.ToTensor()(resized_image_pil)

        transform = self.build_transform(input_size=112)
        #images = self.fast_dynamic_preprocess(stitched_tensor, image_size=448, use_thumbnail=True)
        images = self.dynamic_preprocess(resized_image_pil, image_size=112, use_thumbnail=True)
        #pixel_values = transform(images)'''
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        print(f"Pixel values shape: {pixel_values.shape}")

        return pixel_values
        #print(f"Stitched image shape: {stitched_image.shape}")
        #return stitched_image
    
    def fast_dynamic_preprocess(self, stitched_tensor, image_size=448, use_thumbnail=False):
        """
        Fast tensor slicing into blocks without any Python for-loop.
        """
        # stitched_tensor: (C, H, W)
        C, H, W = stitched_tensor.shape
        num_blocks_w, num_blocks_h = 6, 2  # target aspect ratio width:height

        block_width = image_size
        block_height = image_size


        # First, reshape into blocks
        stitched_tensor = stitched_tensor[:, :block_height * num_blocks_h, :block_width * num_blocks_w]
        blocks = stitched_tensor.view(
            C,
            num_blocks_h, block_height,
            num_blocks_w, block_width
        )  # Shape: (C, num_blocks_h, block_height, num_blocks_w, block_width)

        # Now permute to group the blocks nicely: (num_blocks_h, num_blocks_w, C, block_height, block_width)
        blocks = blocks.permute(1, 3, 0, 2, 4)

        # Merge first two dimensions (flatten block grid)
        blocks = blocks.reshape(-1, C, block_height, block_width)  # (12, 3, 448, 448)

        # Optional thumbnail
        if use_thumbnail:
            thumbnail_img = F.interpolate(
                stitched_tensor.unsqueeze(0),  # (1, 3, H, W)
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)  # (3, 448, 448)
            blocks = torch.cat([blocks, thumbnail_img.unsqueeze(0)], dim=0)  # (13, 3, 448, 448)

        return blocks  # shape (12, 3, 448, 448) or (13, 3, 448, 448)
    

    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            #T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        target_aspect_ratio = (6, 2)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        #target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        #resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = image.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images


class VisionLLMTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: VisionLLMConfig,
    ):
        """
        Initializes target builder.
        :param trajectory_sampling: trajectory sampling specification
        :param config: global config dataclass of VisionLLM
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "vision_llm_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses).poses
        )
        '''frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)'''

        return {
            "trajectory": trajectory,
            #"agent_states": agent_states,
            #"agent_labels": agent_labels,
            #"bev_semantic_map": bev_semantic_map,
        }

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle":
                agent_states_list.append(
                    np.array(
                        [box_x, box_y, box_heading, box_length, box_width],
                        dtype=np.float32,
                    )
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(
                    map_linestring_mask,
                    [points],
                    isClosed=False,
                    color=255,
                    thickness=2,
                )
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = (
                    box_value[3],
                    box_value[4],
                    box_value[5],
                )
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self, layers=layers)
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in TransFuser."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
