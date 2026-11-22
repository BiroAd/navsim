"""Explain a driving-direction-compliance (DDC) penalty in one navtest scene: BEV with the route's roadblocks shaded,
the logged human and the planner's final plan, and the parts of each trajectory that count as "oncoming traffic"
for DDC highlighted.

NAVSIM's rule (pdm_scorer.py, _calculate_driving_direction_compliance): a pose is in oncoming traffic when the ego
centre is in no drivable polygon on the route (here: the route roadblocks of the current frame) and not in an
intersection; the distance travelled in oncoming traffic is summed over a sliding 1 s window; DDC = 1 if its maximum
is < 2 m, 0.5 if < 6 m, else 0. Reconstructed here on the planned / logged poses (10 Hz codebook entry for the plan,
linear interpolation for the human); NAVSIM evaluates the LQR-simulated poses, so the metres are approximate.
  python scripts/visualization/plot_ddc_example.py --token 349606982fb252ae --sensor-path <mount> --out <png>
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pq", HERE / "plot_qualitative.py")
pq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq)
plt = pq.plt
from matplotlib.patches import Patch  # noqa: E402
from nuplan.common.maps.abstract_map import SemanticMapLayer  # noqa: E402


def to_global(xy, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack([pose[0] + c * xy[:, 0] - s * xy[:, 1], pose[1] + s * xy[:, 0] + c * xy[:, 1]], 1)


def to_local(xy, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = xy - pose[:2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def ddc(xy_local, pose, route, inter, dt):
    """oncoming flags per pose, max distance in oncoming traffic within 1 s, DDC value"""
    g = to_global(np.concatenate([[[0.0, 0.0]], xy_local]), pose)
    onc = np.array([not route.contains(Point(p)) and not inter.contains(Point(p)) for p in g])
    step = np.r_[0.0, np.linalg.norm(np.diff(g, axis=0), axis=1)] * onc
    w = int(round(1.0 / dt))
    worst = max(step[max(0, i - w): i + 1].sum() for i in range(len(step)))
    return onc, worst, 1.0 if worst < 2.0 else 0.5 if worst < 6.0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--token", required=True)
    p.add_argument("--sensor-path", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="rl3_b0.005_ep01")
    p.add_argument("--scorer", default="p3_aux_policy_ep03")
    p.add_argument("--ec-folder", default="rl3_b0.005_ep01_own_ecseq_mix10_n64")
    args = p.parse_args()
    rules = pq.Rules(args.model, args.scorer, args.ec_folder)
    loader = pq.scene_loader([args.token], Path(args.sensor_path))
    scene = loader.get_scene_from_token(args.token)
    frame = scene.frames[3]
    pose = np.array(frame.ego_status.ego_pose, dtype=float)
    m = scene.map_api
    route_polys = []
    for rid in frame.roadblock_ids:
        for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            try:
                obj = m.get_map_object(str(rid), layer)
            except Exception:  # noqa: BLE001
                obj = None
            if obj is not None:
                route_polys.append(obj.polygon)
    route = unary_union(route_polys)
    from nuplan.common.actor_state.state_representation import Point2D
    inter = unary_union([o.polygon for o in m.get_proximal_map_objects(Point2D(pose[0], pose[1]), 80,
                                                                        [SemanticMapLayer.INTERSECTION])[
        SemanticMapLayer.INTERSECTION]] or [Point(0, 0).buffer(0)])
    tr, _ = rules(args.token)
    plan = tr["mixture + EC n64"]
    hum8 = scene.get_future_trajectory().poses
    hum = pq.densify(hum8, 5)[1:]  # 0.1 s steps through the logged poses
    res = {"human (logged)": (hum, "black", ddc(hum, pose, route, inter, 0.1)),
           "planner (mixture + consistency)": (plan[:, :2], "#d62728", ddc(plan[:, :2], pose, route, inter, 0.1))}
    trajs = {"human": hum8, "mixture + EC n64": plan}
    fig, ax = plt.subplots(figsize=(4.8, 5.1))  # print: 0.8 x text width (150 mm)
    lim = pq.bev_extent(trajs, margin=10)
    pq.draw_bev(ax, scene, trajs, None, lim=lim)
    for poly in getattr(route, "geoms", [route]):
        xy = to_local(np.asarray(poly.exterior.coords), pose)
        ax.fill(xy[:, 1], xy[:, 0], color="#4c9be8", alpha=0.18, zorder=3, lw=0)
    for name, (xy, col, (onc, worst, val)) in res.items():
        pts = np.concatenate([[[0.0, 0.0]], xy])
        ax.scatter(pts[onc, 1], pts[onc, 0], s=12, c="#e7298a", zorder=40, edgecolors="white", linewidths=0.5)
    off = {"human (logged)": pq.per_scene("human_agent_non_reactive", suffix="").loc[args.token],
           "planner (mixture + consistency)": pq.per_scene(f"{args.model}_own_ecseq_mix10_n64_argmax").loc[args.token]}
    lines = [f"{n.split(' (')[0]}: official DDC {off[n]['driving_direction_compliance']:.1f}, EPDMS {off[n]['score']:.2f}"
             f" (reconstructed: {w:.1f} m off-route in 1 s)" for n, (_, _, (_, w, v)) in res.items()]
    ax.set_title("\n".join(lines), fontsize=6.8)
    ax.legend(handles=[Patch(color="#4c9be8", alpha=0.3, label="route lanes (on-route drivable area)"),
                       plt.Line2D([], [], color="black", linestyle="--", label="human (logged)"),
                       plt.Line2D([], [], color="#d62728", label="planner (mixture + consistency)"),
                       plt.Line2D([], [], color="#e7298a", marker="o", linestyle="", label="pose outside the route lanes (counts for DDC)")],
              loc="lower left", fontsize=6.5, framealpha=0.9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(args.out, lines)


if __name__ == "__main__":
    main()
