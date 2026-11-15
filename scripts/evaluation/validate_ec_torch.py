"""
End-to-end check of the GPU/torch EC (ec_torch.py) on real navtest scenes: for every navtest scene whose previous frame
(0.5 s earlier) is also a navtest scene, EC of the trajectory a scored run output at t against the one it output at
t - 0.5 s, each simulated from that frame's logged velocity / acceleration (the scene logs' ego_dynamic_state, as the
metric cache), compared with the official per-scene EC of that run's CSV.
    python scripts/evaluation/validate_ec_torch.py --tracking $NAVSIM_EXP_ROOT/codebook/16384_8step_ec_tracking.npz \
        --predictions $NAVSIM_EXP_ROOT/predictions/rl3_b0.005_ep01/navtest \
        --csv "$NAVSIM_EXP_ROOT/pdm_navtest/rl3_b0.005_ep01_argmax_non_reactive_split/*/2*[0-9].csv"
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.vision_llm_lora.ec_torch import TorchExtendedComfort
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracking", required=True)
    p.add_argument("--predictions", required=True, help="prediction folder of the scored run (argmax trajectories)")
    p.add_argument("--csv", required=True, help="glob of the run's navtest CSV")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    scene_filter = instantiate(OmegaConf.load(root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
    loader = SceneLoader(data_path=Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / "test",
                         original_sensor_path=None, scene_filter=scene_filter,
                         sensor_config=SensorConfig.build_no_sensors())
    tokens = json.loads((Path(args.predictions) / "tokens.json").read_text())
    row = {t: i for i, t in enumerate(tokens)}
    codebook = np.load(Path(args.predictions) / "codebook.npy")
    traj = np.load(Path(args.predictions) / "trajectory.npy")
    flat = codebook.reshape(len(codebook), -1)
    entry = np.array([np.abs(flat - t.reshape(1, -1)).max(1).argmin() for t in traj])  # codebook id of each output

    cur, prev, v, a, pv, pa = [], [], [], [], [], []
    for token, frames in loader.scene_frames_dicts.items():
        f_now, f_prev = frames[3], frames[2]
        if token not in row or f_prev["token"] not in row:
            continue
        cur.append(token)
        prev.append(f_prev["token"])
        v.append(f_now["ego_dynamic_state"][:2])
        a.append(f_now["ego_dynamic_state"][2:4])
        pv.append(f_prev["ego_dynamic_state"][:2])
        pa.append(f_prev["ego_dynamic_state"][2:4])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ec = TorchExtendedComfort(dict(np.load(args.tracking)), device)
    t = lambda x: torch.tensor(np.asarray(x, np.float64), device=device)
    got = []
    for s in range(0, len(cur), 512):
        sl = slice(s, s + 512)
        got.append(ec.against_previous(t(v[sl]), t(a[sl]), t(pv[sl]), t(pa[sl]),
                                       torch.tensor([entry[row[x]] for x in prev[sl]], device=device),
                                       entries=torch.tensor([[entry[row[x]]] for x in cur[sl]], device=device))[:, 0].cpu())
    got = torch.cat(got).numpy()
    df = pd.read_csv(sorted(glob.glob(args.csv))[-1], index_col=0).set_index("token")
    official = df.loc[cur, "two_frame_extended_comfort"].to_numpy()
    valid = ~np.isnan(official)
    agree = (got[valid] == (official[valid] > 0.5)).mean()
    print(f"{len(cur)} pairs ({valid.sum()} with official EC); torch EC mean {got[valid].mean():.4f}, official "
          f"{official[valid].mean():.4f}; agreement {agree * 100:.2f}% on {device}")


if __name__ == "__main__":
    main()
