#!/usr/bin/env bash
# Parallel, restartable variant of download_navtrain.sh. Run it from the download directory.
# Extracting many small files to Lustre is limited by per-file latency, not bandwidth, so archives
# are extracted concurrently and straight into trainval_sensor_blobs/trainval (no split dir + rsync copy).
#   JOBS=8          archives fetched/extracted concurrently
#   HISTORY=1       0 = skip navtrain_history_* (past frames)
#   SENSOR_FILTER=0 1 = skip MergedPointCloud, CAM_L1, CAM_R1
# Progress: wc -l .navtrain_*.extracted
set -euo pipefail

JOBS=${JOBS:-8}
HISTORY=${HISTORY:-1}
export SENSOR_FILTER=${SENSOR_FILTER:-0}
export URL=https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim
export TARGET=trainval_sensor_blobs/trainval

if pgrep -u "$USER" -f "tar -xzf navtrain_" > /dev/null; then
    echo "download_navtrain.sh is still extracting; stop it first (Ctrl-C in its terminal)." >&2
    exit 1
fi
mkdir -p "$TARGET"

if [ ! -d trainval_navsim_logs ]; then
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
    tar -xzf openscene_metadata_trainval.tgz
    rm openscene_metadata_trainval.tgz
    mv openscene-v1.1/meta_datas trainval_navsim_logs
    rm -r openscene-v1.1
fi

# Adopt split dirs left by an interrupted download_navtrain.sh: drop the file that was being written
# (newest ctime) and move log dirs into place. --skip-old-files below re-extracts only what is missing.
for dir in current_split_* history_split_*; do
    [ -d "$dir" ] || continue
    echo "Adopting partially extracted $dir"
    last=$(find "$dir" -type f -printf '%C@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$last" ]; then rm -f "$last"; fi
    for log in "$dir"/*/; do
        if [ ! -e "$TARGET/$(basename "$log")" ]; then mv "$log" "$TARGET/"; fi
    done
    rm -rf "$dir"
done

fetch_extract() {
    local name=$1
    local list=".$name.extracted"
    local excludes=()
    if [ -e ".$name.done" ]; then
        echo "$name: already done"
        return
    fi
    if [ "$SENSOR_FILTER" = 1 ]; then
        excludes=(--exclude=MergedPointCloud --exclude=CAM_L1 --exclude=CAM_R1)
    fi
    wget -c -nv "$URL/$name.tgz"
    # The last entry listed by an interrupted run may be truncated.
    if [ -s "$list" ]; then
        local last="$TARGET/$(tail -1 "$list" | cut -d/ -f2-)"
        if [ -f "$last" ]; then rm -f "$last"; fi
    fi
    echo "$name: extracting"
    tar -I pigz -xvf "$name.tgz" -C "$TARGET" --strip-components=1 --skip-old-files --warning=no-existing-file "${excludes[@]}" > "$list"
    rm "$name.tgz" "$list"
    touch ".$name.done"
    echo "$name: done"
}
export -f fetch_extract

names="navtrain_current_1 navtrain_current_2 navtrain_current_3 navtrain_current_4"
if [ "$HISTORY" = 1 ]; then
    names="$names navtrain_history_1 navtrain_history_2 navtrain_history_3 navtrain_history_4"
fi
echo $names | tr ' ' '\n' | xargs -P "$JOBS" -I{} bash -euo pipefail -c 'fetch_extract {}'
