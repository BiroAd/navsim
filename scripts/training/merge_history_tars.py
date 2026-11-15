#!/usr/bin/env python3
"""
Stream the history-cache shard tars (scripts/training/cache_history_features.py) to stdout as one tar, for tar2sqfs.
With --delete each shard tar is removed as soon as it has been streamed, so the shards and the image never need
twice the space (the project storage quota); a failed pack then needs the removed shards recomputed (resumable per
shard). The per-shard info files are merged into one history_info.json member.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time

BUFSIZE = 1 << 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", help="history_shard<i>of<n>.tar files")
    parser.add_argument("--delete", action="store_true", help="remove each shard tar after streaming it")
    args = parser.parse_args()
    infos = []
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.GNU_FORMAT, bufsize=BUFSIZE) as out:
        out.copybufsize = BUFSIZE
        for path in args.shards:
            print(f"{path}: streaming", file=sys.stderr, flush=True)
            with tarfile.open(path, mode="r|", bufsize=BUFSIZE) as shard:
                for member in shard:
                    if member.name.startswith("history_info_shard"):
                        infos.append(json.load(shard.extractfile(member)))
                        continue
                    out.addfile(member, shard.extractfile(member) if member.isfile() else None)
            if args.delete:
                os.remove(path)
        summary = {
            "shards": len(infos), "logs": sum(i["logs"] for i in infos),
            "scenes_written": sum(i["scenes_written"] for i in infos),
            "missing": [m for i in infos for m in i["missing"]],
            "source_cache": infos[0]["source_cache"] if infos else None,
            "run_config": infos[0]["run_config"] if infos else None,
            "files": ["vision_llm_history_t1.gz", "vision_llm_history_t2.gz", "vision_llm_history_t3.gz",
                      "vision_llm_ego_history.gz"],
            "packed": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        data = json.dumps(summary, indent=1).encode()
        info = tarfile.TarInfo("history_info.json")
        info.size, info.mtime = len(data), int(time.time())
        out.addfile(info, io.BytesIO(data))
    print(f"merged {len(infos)} shards, {summary['scenes_written']} scenes", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
