# Program Specification: super-ml-pets

## 1) Purpose

`super-ml-pets` is a reinforcement learning project for Super Auto Pets that iteratively improves reward shaping through short train-and-evaluate cycles. The system supports both offline heuristic proposals and optional LLM-generated reward patches.

## 2) Primary Objectives

- Train and evaluate a MaskablePPO agent from an existing checkpoint.
- Improve validation performance (`mean_trophies`, `win_rate`) through repeated reward-weight tuning.
- Keep experiments reproducible with append-only logging and per-run result folders.
- Track and preserve the best-performing checkpoint over time.

## 3) In Scope

- Running iterative loops: propose -> patch -> finetune -> evaluate.
- Editing scalar reward weights in `reward_config.yaml`.
- Logging each experiment as one JSONL record in `log.txt`.
- Saving run artifacts under `results/` including best checkpoints.

## 4) Out of Scope

- Building a new game environment.
- Full hyperparameter sweeps beyond the loop controls already exposed.
- Production deployment infrastructure.

## 5) Inputs and Dependencies

### Required Inputs

- A pretrained checkpoint in `checkpoints/` (e.g., `checkpoints/base_model.zip`).
- Reward configuration file: `reward_config.yaml`.
- Python environment with dependencies from `requirements.txt`.

### Optional Inputs

- `GEMINI_API_KEY` to enable LLM-based patch proposals in `analyze.py`.
- Prompt guidance in `idea.md`.

## 6) Core Workflow

1. Read current reward config and recent experiment history.
2. Generate a reward patch proposal (`analyze.py`).
3. Optionally apply patch to `reward_config.yaml`.
4. Finetune from checkpoint for configurable timesteps (`train_run.py`).
5. Evaluate on a fixed game grid and compute metrics (`experiment.py`).
6. Append metrics and metadata to `log.txt`.
7. If performance improves, update `results/<run>/best_model.zip` and `results/best_checkpoint/best_model.zip`.

## 7) Success Metrics

- **Primary:** `mean_trophies` (higher is better).
- **Tie-breaker:** `win_rate` (higher is better).
- **Secondary:** stability of performance over multiple loop iterations.

## 8) Repository Components

- `run_loop.py`: Orchestrates repeated autoresearch iterations.
- `train_run.py`: Finetune + evaluate + log + best-model tracking.
- `experiment.py`: Standalone evaluation entrypoint.
- `analyze.py`: Proposes reward-weight updates from historical results.
- `reward_config.py`: Reward shaping logic and YAML handling.
- `reward_config.yaml`: Active reward-weight configuration.
- `idea.md`: Proposal-agent instructions.
- `log.txt`: JSONL experiment history.

## 9) Deliverables

- A reproducible command-line loop for iterative RL reward tuning.
- Logged evidence of experimental outcomes in JSONL format.
- Snapshot and best-checkpoint artifacts in `results/`.
- Project documentation in `README.md` and this specification file.

## 10) Operational Notes

- Default iteration count and finetune steps should remain configurable via CLI flags.
- Reward updates should use conservative deltas to reduce instability.
- The project should remain runnable without API keys by using offline fallback proposal logic.
