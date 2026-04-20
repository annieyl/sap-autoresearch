"""
Orchestrate autoresearch: propose patch -> write YAML + snapshot -> finetune+eval -> log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from typing import Any, Dict

from analyze import apply_patch_to_yaml, propose_next, snapshot_config
from train_run import train_run


def _warn_if_conda_and_venv() -> None:
    """Mixing conda (base) with a venv often breaks PyTorch DLLs on Windows (c10.dll WinError 1114)."""
    if os.environ.get("CONDA_PREFIX") and "venv" in sys.prefix.replace("\\", "/").lower():
        print(
            "WARNING: Conda is active (e.g. (base)) while sys.prefix is a venv. "
            "Run `conda deactivate` until only (.venv) remains, or recreate the venv with "
            "https://www.python.org/downloads/ Python (not conda's interpreter).",
            file=sys.stderr,
        )


def run_loop(
    n_iters: int,
    checkpoint_stem: str,
    log_path: str = "log.txt",
    reward_config_path: str = "reward_config.yaml",
    idea_path: str = "idea.md",
    results_root: str = "results",
    finetune_steps: int = 2048,
    eval_rows: int = 10,
    eval_cols: int = 10,
    seed: int = 0,
) -> Dict[str, Any]:
    os.makedirs(results_root, exist_ok=True)
    last: Dict[str, Any] = {}
    run_started = time.perf_counter()

    for i in range(n_iters):
        iter_started = time.perf_counter()
        run_dir = os.path.join(results_root, f"exp_{i:04d}")
        os.makedirs(run_dir, exist_ok=True)

        proposal_path = os.path.join(run_dir, "proposal.json")
        t0 = time.perf_counter()
        proposal = propose_next(log_path, reward_config_path, idea_path, seed=seed + i)
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)
        propose_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        apply_patch_to_yaml(reward_config_path, proposal.get("patch") or {})
        snapshot_config(reward_config_path, run_dir)
        patch_seconds = time.perf_counter() - t1

        t2 = time.perf_counter()
        out = train_run(
            checkpoint_stem,
            reward_config_path,
            log_path,
            results_dir=run_dir,
            finetune_steps=finetune_steps,
            eval_rows=eval_rows,
            eval_cols=eval_cols,
            iter_index=i,
            proposal=proposal,
            seed=seed + i * 9973,
        )
        train_eval_seconds = time.perf_counter() - t2
        last = out

        if out.get("is_new_best"):
            best_dir = os.path.join(results_root, "best_checkpoint")
            os.makedirs(best_dir, exist_ok=True)
            src = os.path.join(run_dir, "best_model.zip")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(best_dir, "best_model.zip"))

        iter_seconds = time.perf_counter() - iter_started
        stats = out.get("summary", {})
        print(
            (
                f"[iter {i + 1}/{n_iters}] "
                f"elapsed={iter_seconds:.2f}s "
                f"(propose={propose_seconds:.2f}s, patch={patch_seconds:.2f}s, train_eval={train_eval_seconds:.2f}s) "
                f"verdict={out.get('verdict')} "
                f"win_rate={float(stats.get('win_rate', 0.0)):.3f} "
                f"mean_trophies={float(stats.get('mean_trophies', 0.0)):.3f}"
            )
        )

    return {"last": last, "total_runtime_seconds": round(time.perf_counter() - run_started, 3)}


def main() -> None:
    # Keep terminal output clean during long loops.
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3, help="Number of autoresearch iterations")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "base_model"),
        help="Pretrained MaskablePPO stem (no .zip)",
    )
    parser.add_argument("--log", type=str, default="log.txt")
    parser.add_argument("--reward-config", type=str, default="reward_config.yaml")
    parser.add_argument("--idea", type=str, default="idea.md")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--finetune-steps", type=int, default=2048)
    parser.add_argument("--eval-rows", type=int, default=10)
    parser.add_argument("--eval-cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _warn_if_conda_and_venv()

    summary = run_loop(
        n_iters=args.iters,
        checkpoint_stem=args.checkpoint,
        log_path=args.log,
        reward_config_path=args.reward_config,
        idea_path=args.idea,
        results_root=args.results_root,
        finetune_steps=args.finetune_steps,
        eval_rows=args.eval_rows,
        eval_cols=args.eval_cols,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
