#!/usr/bin/env bash
# Verify the navtrain archives (MD5 sums from docs/splits.md), then pack them into one squashfs image.
# Run from the download directory, inside tmux. Mount the result with:
#   squashfuse navtrain_sensor_blobs.sqfs <dataset>/sensor_blobs/trainval
set -euo pipefail

cat > navtrain.md5 <<'EOF'
6f92f38d5f03ed852da7872a7122bdd2  navtrain_current_1.tgz
7a72f0a758b5df6cbe4c565920a4869f  navtrain_current_2.tgz
b083fce1428308abb5682a1a150cc1af  navtrain_current_3.tgz
68354ac2c993aa1ebbfac59547fdb840  navtrain_current_4.tgz
dc46ed34d92d5ab9cc1464d67b72fbf6  navtrain_history_1.tgz
fab177bdb79c0c9536da1566d13e5995  navtrain_history_2.tgz
71ed9a2387edc3849921861d7873c7f0  navtrain_history_3.tgz
2cc13aced2f458e50fe4cc2f26d18e07  navtrain_history_4.tgz
EOF

echo "Checking MD5 sums (parallel)..."
pids=()
while read -r _ name; do
    grep " $name\$" navtrain.md5 | md5sum -c - &
    pids+=($!)
done < navtrain.md5
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "MD5 mismatch: re-download the failed archive before building." >&2; exit 1; }
done

echo "Building navtrain_sensor_blobs.sqfs..."
python3.11 -I merge_sensor_tars.py navtrain_current_{1..4}.tgz navtrain_history_{1..4}.tgz \
    | tar2sqfs -q -c lz4 -j 8 navtrain_sensor_blobs.sqfs
echo "BUILD OK"
