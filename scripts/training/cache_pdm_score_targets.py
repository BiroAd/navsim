"""
Writes the offline per-candidate PDM sub-scores (scripts/evaluation/score_codebook_trajectories.py, one .npz per log)
into the training cache as the PDMScoreTargetBuilder target: <cache>/<log>/<token>/vision_llm_pdm_scores.gz with
{"pdm_scores": (len(PDM_SCORE_KEYS), K) float16}, so training with pdm_score_loss_weight > 0 reads them like any
cached target. Adds one file per scene (~103k for navtrain; /project has a quota of 1M files). Existing files are
skipped; scenes without a token folder in the cache are counted and listed. Usage:
  python scripts/training/cache_pdm_score_targets.py --scores $NAVSIM_EXP_ROOT/codebook_scores_navtrain \
      --cache $NAVSIM_EXP_ROOT/training_cache --workers 16
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from navsim.agents.vision_llm_lora.vision_llm_lora_features import PDM_SCORE_KEYS, PDMScoreTargetBuilder
from navsim.planning.training.dataset import dump_feature_target_to_pickle

logger = logging.getLogger(__name__)
FILE_NAME = PDMScoreTargetBuilder().get_unique_name() + ".gz"


def scene_targets(scores: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """{token: {"pdm_scores": (M, K) float16}} for every scored scene of one log's .npz arrays."""
    stacked = np.stack([scores[key] for key in PDM_SCORE_KEYS], axis=1).astype(np.float16)  # (N, M, K)
    return {str(token): {"pdm_scores": torch.from_numpy(stacked[i])} for i, token in enumerate(scores["tokens"])}


def _write_log(job: Tuple[Path, Path]) -> Tuple[int, int, List[str]]:
    """Writes one log's targets. Returns (written, already present, tokens missing from the cache)."""
    npz_path, cache = job
    written, present, missing = 0, 0, []
    log_dir = cache / npz_path.stem
    for token, target in scene_targets(dict(np.load(npz_path))).items():
        token_dir = log_dir / token
        if not token_dir.is_dir():
            missing.append(token)
            continue
        path = token_dir / FILE_NAME
        if path.exists():
            present += 1
            continue
        tmp = path.with_suffix(".tmp")
        dump_feature_target_to_pickle(tmp, target)
        tmp.replace(path)  # never a half-written target file
        written += 1
    return written, present, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    jobs = [(path, args.cache) for path in sorted(args.scores.glob("*.npz"))]
    written = present = 0
    missing: List[str] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (w, p, m) in enumerate(ex.map(_write_log, jobs), 1):
            written, present, missing = written + w, present + p, missing + m
            if i % 100 == 0 or i == len(jobs):
                logger.info(f"[{i}/{len(jobs)}] logs: {written} written, {present} already present, {len(missing)} missing")
    if missing:
        logger.warning(f"{len(missing)} scored scenes have no folder in the cache, e.g. {missing[:5]}")


if __name__ == "__main__":
    main()
