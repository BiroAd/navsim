"""
CPU-only checks of VisionLLMLoRAAuxTargetBuilder's coordinate conventions with a fake map and scene (no data).
Run: pytest tests/test_vision_llm_lora_aux_targets.py
"""

from types import SimpleNamespace

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from shapely.geometry import LineString, Polygon

from navsim.agents.vision_llm_lora.vision_llm_lora_config import VisionLLMLoRAConfig
from navsim.agents.vision_llm_lora.vision_llm_lora_features import NO_LANE_HEADING, VisionLLMLoRAAuxTargetBuilder

# Ego at (100, 200) facing north (+y global): ego-forward = global +y, ego-left = global -x.
EGO = (100.0, 200.0, np.pi / 2)


def _lane(points):
    headings = np.arctan2(np.diff([p[1] for p in points]), np.diff([p[0] for p in points]))
    headings = np.append(headings, headings[-1])
    path = SimpleNamespace(
        linestring=LineString(points), discrete_path=[StateSE2(x, y, h) for (x, y), h in zip(points, headings)]
    )
    return SimpleNamespace(baseline_path=path, polygon=LineString(points).buffer(1.5))


class _Map:
    def __init__(self):
        # Road patch 10-20 m ahead of ego, 2 m to each side (global: x 98..102, y 210..220).
        self.road = SimpleNamespace(polygon=Polygon([(98, 210), (102, 210), (102, 220), (98, 220)]))
        # Lane driving north (= ego heading) through ego's position.
        self.lane = _lane([(100.0, 190.0), (100.0, 240.0)])

    def get_proximal_map_objects(self, point, radius, layers):
        objects = {layer: [] for layer in layers}
        if SemanticMapLayer.ROADBLOCK in objects:
            objects[SemanticMapLayer.ROADBLOCK] = [self.road]
        if SemanticMapLayer.LANE in objects:
            objects[SemanticMapLayer.LANE] = [self.lane]
        return objects


def _scene():
    # Annotations are in the ego frame: a car 10 m ahead, 5 m left, driving forward at 3 m/s; a cone (static).
    boxes = np.array([[10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0], [30.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0]], dtype=np.float32)
    annotations = SimpleNamespace(
        names=["vehicle", "traffic_cone"], boxes=boxes, velocity_3d=np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    frame = SimpleNamespace(ego_status=SimpleNamespace(ego_pose=np.array(EGO)), annotations=annotations)
    return SimpleNamespace(frames=[frame], scene_metadata=SimpleNamespace(num_history_frames=1), map_api=_Map())


def _pixel(x, y, res=0.5, x_min=-16.0, y_max=32.0):
    return int((x - x_min) / res), int((y_max - y) / res)  # row, column


def _targets():
    builder = VisionLLMLoRAAuxTargetBuilder(VisionLLMLoRAConfig().bev_semantic_classes)
    return builder.compute_targets(_scene())


def test_grid_shape_and_dtypes():
    t = _targets()
    for key in ("bev_semantic_map", "drivable_area_map", "lane_heading_map"):
        assert tuple(t[key].shape) == (160, 128) and str(t[key].dtype) == "torch.uint8"
    assert tuple(t["agent_states"].shape) == (40, 7) and tuple(t["agent_classes"].shape) == (40,)


def test_map_polygons_are_rotated_into_the_ego_frame():
    drivable = _targets()["drivable_area_map"].numpy()
    assert drivable[_pixel(15.0, 0.0)] == 1            # 15 m ahead, centre: on the road patch
    assert drivable[_pixel(15.0, 5.0)] == 0            # 5 m to the left: off the patch
    assert drivable[_pixel(-5.0, 0.0)] == 0            # behind ego: off the patch
    assert drivable.sum() == 160                       # 10 m x 4 m at 0.5 m: exactly 20 x 8 cells


def test_boxes_and_agents_use_ego_coordinates():
    t = _targets()
    semantic = t["bev_semantic_map"].numpy()
    assert semantic[_pixel(10.0, 5.0)] == 5            # vehicle class at 10 m ahead, 5 m left
    assert semantic[_pixel(30.0, 0.0)] == 4            # traffic cone: static-object class
    assert t["agent_classes"][0] == 0 and t["agent_classes"][1] == -1   # one dynamic agent (the cone is not)
    np.testing.assert_allclose(t["agent_states"][0].numpy(), [10.0, 5.0, 0.0, 4.0, 2.0, 3.0, 0.0])


def test_lane_heading_is_relative_to_ego():
    heading = _targets()["lane_heading_map"].numpy()
    assert heading[_pixel(20.0, 0.0)] == 18            # lane along ego heading: relative angle 0 -> bin 18 of 36
    assert heading[_pixel(20.0, 10.0)] == NO_LANE_HEADING


def test_overlapping_polygons_do_not_cancel_out():
    # A second road patch overlapping the first (as lane connectors overlap intersections): the overlap stays drivable.
    scene = _scene()
    road = scene.map_api.road
    overlap = SimpleNamespace(polygon=Polygon([(98, 215), (102, 215), (102, 225), (98, 225)]))
    scene.map_api.get_proximal_map_objects = lambda point, radius, layers: {
        layer: ([road, overlap] if layer == SemanticMapLayer.ROADBLOCK else []) for layer in layers
    }
    drivable = VisionLLMLoRAAuxTargetBuilder(VisionLLMLoRAConfig().bev_semantic_classes).compute_targets(scene)
    drivable = drivable["drivable_area_map"].numpy()
    assert drivable[_pixel(17.5, 0.0)] == 1            # inside both patches
    assert drivable.sum() == 30 * 8                    # union: 15 m x 4 m
