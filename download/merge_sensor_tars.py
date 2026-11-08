#!/usr/bin/env python3
"""
Stream several NAVSIM sensor-blob .tgz archives to stdout as one merged tar.

Leading directories (by default the split directory current_split_N/ or history_split_N/) are stripped so all
archives land in one <log>/<sensor>/<file> tree, and duplicate entries are dropped. Pipe the output into tar2sqfs to get a single
squashfs image, which costs one inode on Lustre instead of one per file.
"""
import argparse
import subprocess
import sys
import tarfile

BUFSIZE = 1 << 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", help=".tgz archives, merged in the given order")
    parser.add_argument("--exclude", action="append", default=[], help="skip entries with this path component")
    parser.add_argument("--strip-components", type=int, default=1, help="leading path components to drop")
    args = parser.parse_args()

    excluded = set(args.exclude)
    seen = set()
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.GNU_FORMAT, bufsize=BUFSIZE) as out:
        out.copybufsize = BUFSIZE
        for path in args.archives:
            print(f"{path}: streaming", file=sys.stderr, flush=True)
            unpack = subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=unpack.stdout, mode="r|", bufsize=BUFSIZE) as src:
                src.copybufsize = BUFSIZE
                for member in src:
                    parts = member.name.strip("/").split("/")[args.strip_components :]
                    if not parts or excluded.intersection(parts):
                        continue
                    member.name = "/".join(parts)
                    if member.name in seen:
                        continue
                    seen.add(member.name)
                    out.addfile(member, src.extractfile(member) if member.isfile() else None)
            if unpack.wait() != 0:
                sys.exit(f"{path}: pigz failed")
            print(f"{path}: done, {len(seen)} entries so far", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
