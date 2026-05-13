import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm


def _iter_target_files(cache_path: Path, target_filename: str) -> Iterable[Path]:
    for log_dir in cache_path.iterdir():
        if not log_dir.is_dir():
            continue
        for token_dir in log_dir.iterdir():
            if not token_dir.is_dir():
                continue
            target_file = token_dir / target_filename
            if target_file.is_file():
                yield target_file


def _load_pickle_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _to_numpy(array_like) -> np.ndarray:
    if hasattr(array_like, "detach") and hasattr(array_like, "cpu"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _collect_trajectories(
    cache_path: Path,
    target_filename: str,
    target_key: str,
    max_trajectories: Optional[int],
) -> np.ndarray:
    trajectories: List[np.ndarray] = []

    target_files = list(_iter_target_files(cache_path, target_filename))
    for target_file in tqdm(target_files, desc="Loading trajectory targets"):
        data_dict = _load_pickle_gz(target_file)
        if target_key not in data_dict:
            continue

        trajectory = _to_numpy(data_dict[target_key]).astype(np.float32)

        if trajectory.ndim != 2:
            continue
        if trajectory.shape[-1] != 3:
            continue

        trajectories.append(trajectory)
        if max_trajectories is not None and len(trajectories) >= max_trajectories:
            break

    if not trajectories:
        raise RuntimeError(
            "No trajectory data found. "
            "Check cache path, target filename, and target key."
        )

    horizon = trajectories[0].shape[0]
    valid = [traj for traj in trajectories if traj.shape == (horizon, 3)]
    if not valid:
        raise RuntimeError("No valid trajectories with consistent shape were found.")

    return np.stack(valid, axis=0)


def _fit_kmeans(data_2d: np.ndarray, num_clusters: int, use_minibatch: bool, random_state: int):
    if use_minibatch:
        estimator = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=random_state,
            batch_size=8192,
            n_init=10,
            max_iter=300,
            reassignment_ratio=0.01,
            verbose=1,
        )
    else:
        estimator = KMeans(
            n_clusters=num_clusters,
            random_state=random_state,
            n_init="auto",
            max_iter=300,
            tol=0.0,
            verbose=1,
            init="k-means++",
        )

    estimator.fit(data_2d)
    return estimator


def build_codebook(
    trajectories: np.ndarray,
    codebook_size: int,
    level: str,
    use_minibatch: bool,
    random_state: int,
) -> np.ndarray:
    if level == "point":
        clustered = trajectories.reshape(-1, trajectories.shape[-1])
        estimator = _fit_kmeans(clustered, codebook_size, use_minibatch, random_state)
        return estimator.cluster_centers_.astype(np.float32)

    if level == "trajectory":
        clustered = trajectories.reshape(trajectories.shape[0], -1)
        estimator = _fit_kmeans(clustered, codebook_size, use_minibatch, random_state)
        centers = estimator.cluster_centers_.reshape(codebook_size, trajectories.shape[1], trajectories.shape[2])
        return centers.astype(np.float32)

    raise ValueError(f"Unsupported level: {level}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline KMeans trajectory codebook from NAVSIM cache.")
    parser.add_argument("--cache-path", type=Path, required=True, help="Path to training cache root.")
    parser.add_argument(
        "--target-filename",
        type=str,
        default="vision_llm_target.gz",
        help="Cached target filename inside each token directory.",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="trajectory",
        help="Target dictionary key storing trajectory tensor.",
    )
    parser.add_argument("--codebook-size", type=int, default=2048, help="Number of KMeans clusters.")
    parser.add_argument(
        "--level",
        choices=["point", "trajectory"],
        default="point",
        help="Cluster individual points (Kx3) or full trajectories (KxTx3).",
    )
    parser.add_argument("--output-codebook", type=Path, required=True, help="Output .npy file for cluster centers.")
    parser.add_argument(
        "--output-trajectories",
        type=Path,
        default=None,
        help="Optional output .npy for extracted trajectories before clustering.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional cap on number of trajectories loaded.",
    )
    parser.add_argument(
        "--full-kmeans",
        action="store_true",
        help="Use full KMeans instead of MiniBatchKMeans.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cache_path.is_dir():
        raise FileNotFoundError(f"Cache path does not exist: {args.cache_path}")

    trajectories = _collect_trajectories(
        cache_path=args.cache_path,
        target_filename=args.target_filename,
        target_key=args.target_key,
        max_trajectories=args.max_trajectories,
    )

    args.output_codebook.parent.mkdir(parents=True, exist_ok=True)
    if args.output_trajectories is not None:
        args.output_trajectories.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_trajectories, trajectories)

    codebook = build_codebook(
        trajectories=trajectories,
        codebook_size=args.codebook_size,
        level=args.level,
        use_minibatch=not args.full_kmeans,
        random_state=args.random_state,
    )
    np.save(args.output_codebook, codebook)

    summary = {
        "num_trajectories": int(trajectories.shape[0]),
        "horizon": int(trajectories.shape[1]),
        "dim": int(trajectories.shape[2]),
        "level": args.level,
        "codebook_size": int(codebook.shape[0]),
        "codebook_shape": list(codebook.shape),
        "cache_path": str(args.cache_path),
        "target_filename": args.target_filename,
        "target_key": args.target_key,
        "output_codebook": str(args.output_codebook),
    }

    summary_path = args.output_codebook.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Codebook saved:", args.output_codebook)
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
