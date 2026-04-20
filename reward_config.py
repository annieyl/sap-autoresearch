"""
Reward YAML + sapai-gym env shaping (no PyTorch / SB3 imports).
"""

from __future__ import annotations

import types
from typing import Any, Dict

import yaml
from sapai_gym import SuperAutoPetsEnv
from sapai_gym.opponent_gen.opponent_generators import biggest_numbers_horizontal_opp_generator

DEFAULT_WEIGHTS = {
    "wins": 1.0,
    "bad_action": 1.0,
    "lives": 0.0,
    "gold": 0.0,
    "turn": 0.0,
    "team_power": 0.0,
}


def opponent_generator(num_turns: int):
    return biggest_numbers_horizontal_opp_generator(num_turns)


def load_reward_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    weights = dict(DEFAULT_WEIGHTS)
    weights.update((data.get("weights") or {}))
    return {"weights": weights}


def _team_power_norm(player) -> float:
    total = 0
    for slot in player.team:
        if slot.empty:
            continue
        total += int(slot.pet.attack) + int(slot.pet.health)
    return float(min(total / 200.0, 1.0))


def _shaped_get_reward(self):
    w = getattr(self, "_reward_cfg", DEFAULT_WEIGHTS)
    p = self.player
    bad = float(self.bad_action_reward_sum)
    r = w.get("bad_action", 1.0) * bad
    r += w.get("wins", 1.0) * (p.wins / 10.0)
    r += w.get("lives", 0.0) * (p.lives / 10.0)
    r += w.get("gold", 0.0) * (min(p.gold, 20) / 20.0) * 0.1
    r += w.get("turn", 0.0) * (1.0 - min(p.turn, 25) / 25.0) * 0.01
    r += w.get("team_power", 0.0) * _team_power_norm(p)
    return float(r)


def apply_reward_config(env: SuperAutoPetsEnv, weights: Dict[str, float]) -> None:
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(weights)
    env._reward_cfg = merged
    env.get_reward = types.MethodType(_shaped_get_reward, env)


def make_env() -> SuperAutoPetsEnv:
    return SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
