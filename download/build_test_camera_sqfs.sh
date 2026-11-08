#!/usr/bin/env bash
# Download the OpenScene test camera archives (32 parts, ~127 GB) and pack them into one squashfs image.
# Lidar is skipped (camera-only agents). Run from the download directory, inside tmux, on the login node.
# Mount the result with:
#   squashfuse test_sensor_blobs.sqfs <dataset>/sensor_blobs/test
# No checksums are published for these archives; pigz verifies each gzip stream and pipefail stops the build.
set -euo pipefail

URL=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
mkdir -p test_camera

echo "Downloading 32 camera archives (8 in parallel)..."
seq 0 31 | xargs -P 8 -I{} wget -c -nv -P test_camera "$URL/openscene_sensor_test_camera_{}.tgz"

echo "Building test_sensor_blobs.sqfs..."
archives=()
for i in $(seq 0 31); do
    archives+=("test_camera/openscene_sensor_test_camera_$i.tgz")
done
# Members are openscene-v1.1/sensor_blobs/test/<log>/..., so drop three leading components.
python3.11 -I merge_sensor_tars.py --strip-components 3 "${archives[@]}" \
    | tar2sqfs -q -c lz4 -j 8 test_sensor_blobs.sqfs
echo "BUILD OK"
