#!/usr/bin/env python3
"""
Check that a (mounted) sensor-blob tree contains the camera frames a NAVSIM scene filter needs.

Reads the log pickles without numpy/nuplan installed, so it runs with the system python (python3.11 -I).
Example:
  python3.11 -I verify_sensor_image.py --mount mnt --logs trainval_navsim_logs/trainval \
      --scene-filter ../navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml
"""
import argparse
import os
import pickle
import re
import sys

# Cameras of vision_llm_lora's camera_layout=panorama (L1/R1 replace the original layout's L2/R2).
USED_CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_B0"]


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("numpy", "nuplan", "navsim"):
            return _Stub
        return super().find_class(module, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mount", required=True, help="sensor blob root, e.g. the squashfuse mountpoint")
    parser.add_argument("--logs", required=True, help="directory with the split's log pickles")
    parser.add_argument("--scene-filter", required=True, help="scene_filter/<split>.yaml")
    parser.add_argument("--cams", nargs="+", default=USED_CAMS, help="cameras to check")
    parser.add_argument("--history", type=int, default=0, help="also check CAM_F0 of this many history frames")
    args = parser.parse_args()

    text = open(args.scene_filter).read()
    log_names = re.findall(r"^\s*- '?(\d{4}\.\d\d\.\d\d[^'\s]+)'?$", text, re.M)
    tokens = set(re.findall(r"^\s*- '?([0-9a-f]{16})'?$", text, re.M))
    print(f"filter: {len(log_names)} logs, {len(tokens)} tokens", flush=True)

    found = missing = hist_checked = hist_missing = 0
    examples = []
    for log_name in log_names:
        log_path = os.path.join(args.logs, f"{log_name}.pkl")
        if not os.path.exists(log_path):
            print(f"missing log pickle: {log_path}")
            missing += 1
            continue
        frames = _StubUnpickler(open(log_path, "rb")).load()
        for idx, frame in enumerate(frames):
            if frame["token"] not in tokens:
                continue
            found += 1
            for cam in args.cams:
                path = os.path.join(args.mount, frame["cams"][cam]["data_path"])
                if not os.path.exists(path):
                    missing += 1
                    if len(examples) < 5:
                        examples.append(path)
            for hist_idx in range(max(0, idx - args.history), idx):
                hist_checked += 1
                hist_path = os.path.join(args.mount, frames[hist_idx]["cams"]["CAM_F0"]["data_path"])
                hist_missing += not os.path.exists(hist_path)

    print(f"tokens found in logs: {found} / {len(tokens)}")
    print(f"current-frame images missing: {missing} of {found * len(args.cams)} ({', '.join(args.cams)})")
    if args.history:
        print(f"history frames (CAM_F0) missing: {hist_missing} of {hist_checked}")
    for path in examples:
        print(f"  e.g. {path}")
    sys.exit(1 if missing or hist_missing or found < len(tokens) else 0)


if __name__ == "__main__":
    main()
