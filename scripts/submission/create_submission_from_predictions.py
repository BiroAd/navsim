"""
Leaderboard submission pickle from saved split-evaluation predictions (no GPU): the token loop and output dict of
navsim/planning/script/run_create_submission_pickle.py, with trajectories from VisionLLMLoRAPrecomputedAgent (the same
select_trajectory as the local scoring jobs), so a policy and a scorer from two different models (SCORES_PRED in
scripts/evaluation/run_vision_llm_lora_score_predictions_slurm.sh) can be submitted together. The upstream script cannot
run this agent: it rejects agents with requires_scene, and the precomputed agent needs the scene only for its token.

    python scripts/submission/create_submission_from_predictions.py \
        --predictions $NAVSIM_EXP_ROOT/predictions/rl3_b0.005_ep01/navhard_two_stage \
        --scores $NAVSIM_EXP_ROOT/predictions/p3_aux_ctrl_ep02/navhard_two_stage \
        --selection shortlist --shortlist-size 256 --out $NAVSIM_EXP_ROOT/submission_navhard_two_stage/<name>
"""
import argparse
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from tqdm import tqdm

from navsim.agents.vision_llm_lora.vision_llm_lora_precomputed_agent import VisionLLMLoRAPrecomputedAgent
from navsim.common.dataloader import SceneLoader

CONFIG_DIR = Path(__file__).resolve().parents[2] / "navsim/planning/script/config/pdm_scoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="predictions/<pred>/<split> (policy logits)")
    parser.add_argument("--scores", type=Path, default=None, help="predictions/<pred>/<split> whose sub-scores to use")
    parser.add_argument("--selection", default="argmax")
    parser.add_argument("--shortlist-size", type=int, default=64)
    parser.add_argument("--mixture-lambda", type=float, default=10.0)
    parser.add_argument("--split", default="navhard_two_stage")
    parser.add_argument("--out", type=Path, required=True, help="folder for submission.pkl")
    parser.add_argument("--team-name", default=os.environ.get("TEAM_NAME", "Anonymous"))
    parser.add_argument("--authors", default=os.environ.get("AUTHORS", "Anonymous"))
    parser.add_argument("--email", default=os.environ.get("EMAIL", "biroadam02@gmail.com"))
    parser.add_argument("--institution", default=os.environ.get("INSTITUTION", "Budapest_University_of_Technology_and_Economics"))
    parser.add_argument("--country", default=os.environ.get("COUNTRY", "Hungary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="default_run_create_submission_pickle",
            overrides=[
                f"train_test_split={args.split}",
                f"synthetic_scenes_path={data_root / args.split / 'synthetic_scene_pickles'}",
                f"synthetic_sensor_path={data_root / args.split / 'sensor_blobs'}",
                f"original_sensor_path={data_root}",  # never read: the agent asks for no sensors
                "agent=vision_llm_lora_precomputed_agent",  # as in run_vision_llm_lora_score_predictions_slurm.sh
                f"agent.predictions_path='{args.predictions}'",
                f"agent.trajectory_selection={args.selection}",
                f"agent.selection_shortlist_size={args.shortlist_size}",
                f"agent.selection_mixture_lambda={args.mixture_lambda}",
            ]
            + ([f"agent.scores_path='{args.scores}'"] if args.scores else []),
        )
    agent = instantiate(cfg.agent)
    assert isinstance(agent, VisionLLMLoRAPrecomputedAgent)
    agent.initialize()
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        sensor_config=agent.get_sensor_config(),
    )

    def run(tokens, stage):
        out = {}
        for token in tqdm(tokens, desc=stage):
            scene = SimpleNamespace(scene_metadata=SimpleNamespace(initial_token=token))  # the agent reads only the token
            out[token] = agent.compute_trajectory(None, scene)
        return out

    first = run(loader.tokens_stage_one, "first stage")
    second = run(loader.reactive_tokens_stage_two, "second stage")
    submission = {
        "team_name": args.team_name,
        "authors": args.authors,
        "email": args.email,
        "institution": args.institution,
        "country / region": args.country,
        "first_stage_predictions": [first],
        "second_stage_predictions": [second],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "submission.pkl", "wb") as f:
        pickle.dump(submission, f)
    print(f"{args.out / 'submission.pkl'}: stage one {len(first)}, stage two {len(second)} trajectories; "
          f"selection {args.selection} (k {args.shortlist_size}, lambda {args.mixture_lambda}); "
          f"policy {args.predictions}; scores {args.scores or args.predictions}")


if __name__ == "__main__":
    main()
