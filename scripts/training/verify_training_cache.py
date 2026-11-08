"""
Verifies a dataset cache written by run_dataset_caching.py: every token dir must hold all expected .gz files and each
must decompress and unpickle. A job killed mid-write (TIMEOUT/scancel) can leave a truncated .gz behind, and re-runs
skip tokens whose files exist, so such files are never repaired. Bad token dirs are listed and optionally deleted so
the next caching run recomputes them.
Usage:
    python verify_training_cache.py $NAVSIM_EXP_ROOT/training_cache --workers 32 [--delete]
"""

import argparse
import gzip
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

def check_token(token_path: Path, expected_files: List[str]) -> Optional[Tuple[Path, str]]:
    """Returns (token_path, reason) if the token dir is incomplete or corrupt, else None."""
    for name in expected_files:
        path = token_path / name
        if not path.is_file():
            return token_path, f"missing {name}"
        try:
            with gzip.open(path, "rb") as f:
                pickle.load(f)
                # pickle.load can stop before the gzip trailer; reading to EOF also checks the CRC and length.
                f.read()
        except Exception as e:
            return token_path, f"{name}: {type(e).__name__}: {e}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_path", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    # Builder unique names + ".gz"; the defaults are the vision_llm_lora feature/target builders.
    parser.add_argument("--files", nargs="+", default=["vision_llm_feature.gz", "vision_llm_target.gz"])
    parser.add_argument("--delete", action="store_true", help="remove bad token dirs so caching recomputes them")
    args = parser.parse_args()

    token_paths: List[Path] = [t for log in sorted(args.cache_path.iterdir()) for t in sorted(log.iterdir())]
    with ProcessPoolExecutor(args.workers) as pool:
        check = partial(check_token, expected_files=args.files)
        results = list(tqdm(pool.map(check, token_paths, chunksize=64), total=len(token_paths)))
    bad = [r for r in results if r is not None]

    for token_path, reason in bad:
        print(f"BAD {token_path}: {reason}")
    print(f"{len(token_paths)} tokens checked, {len(bad)} bad")

    if args.delete:
        for token_path, _ in bad:
            shutil.rmtree(token_path)
        print(f"Deleted {len(bad)} token dirs")


if __name__ == "__main__":
    main()
