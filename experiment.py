"""
Run a single evaluation: load reward_config.yaml, play games with a frozen policy, return aggregate stats.

PyTorch / sb3_contrib are imported only inside functions so tools like analyze.py can run without loading torch.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from sapai_gym import SuperAutoPetsEnv

from reward_config import apply_reward_config, load_reward_config, make_env


def _get_maskable_ppo():
    from sb3_contrib import MaskablePPO

    return MaskablePPO


def _get_action_masks(env):
    from sb3_contrib.common.maskable.utils import get_action_masks

    return get_action_masks(env)


def rollout_episode(model: Any, env: SuperAutoPetsEnv) -> Dict[str, Any]:
    obs = env.reset()
    steps = 0
    while True:
        masks = _get_action_masks(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _reward, done, _info = env.step(int(action))
        steps += 1
        if done:
            break
    p = env.player
    won = p.wins >= 10
    return {
        "trophies": int(p.wins),
        "lives": int(p.lives),
        "turn": int(p.turn),
        "won": won,
        "steps": steps,
    }


def run_evaluation(
    model_path: str,
    reward_config_path: str,
    n_rows: int = 10,
    n_cols: int = 10,
    seed: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Play n_rows * n_cols full games (episodes) with the same policy checkpoint.
    Returns (summary, per_episode_records).
    """
    MaskablePPO = _get_maskable_ppo()

    cfg = load_reward_config(reward_config_path)
    weights = cfg["weights"]

    env = make_env()
    apply_reward_config(env, weights)

    model = MaskablePPO.load(model_path)

    rng = np.random.RandomState(seed)
    episodes: List[Dict[str, Any]] = []
    n_games = n_rows * n_cols

    for k in range(n_games):
        env.reset()
        if seed is not None:
            env.action_space.seed(int(rng.randint(0, 2**31 - 1)))
        ep = rollout_episode(model, env)
        ep["episode_index"] = k
        episodes.append(ep)

    env.close()

    wins = sum(1 for e in episodes if e["won"])
    trophies = [e["trophies"] for e in episodes]
    summary = {
        "n_games": n_games,
        "grid": [n_rows, n_cols],
        "win_rate": wins / max(n_games, 1),
        "mean_trophies": float(np.mean(trophies)),
        "std_trophies": float(np.std(trophies)),
        "mean_final_stats": {
            "mean_lives": float(np.mean([e["lives"] for e in episodes])),
            "mean_turn": float(np.mean([e["turn"] for e in episodes])),
            "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        },
        "weights": weights,
    }
    return summary, episodes


def append_log(log_path: str, record: Dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with reward_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to MaskablePPO .zip without extension")
    parser.add_argument("--reward-config", type=str, default="reward_config.yaml")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--log", type=str, default=None, help="Append JSON summary as one line")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    zip_path = args.model + ".zip" if not args.model.endswith(".zip") else args.model
    base = args.model[:-4] if args.model.endswith(".zip") else args.model
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Model not found: {zip_path}")

    summary, _ = run_evaluation(base, args.reward_config, args.rows, args.cols, args.seed)
    print(json.dumps(summary, indent=2))

    if args.log:
        append_log(args.log, {"kind": "eval_only", **summary})


if __name__ == "__main__":
    main()
