from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RLConfig:
    episodes: int = 100
    max_turns: int = 100
    replay_capacity: int = 20_000
    batch_size: int = 64
    self_play_workers: int = 1
    self_play_temperature: float = 1.0
    self_play_exploration: float = 0.10
    self_play_repetition_avoidance: float = 0.35
    self_play_repetition_threshold: float = 0.20
    run_name: str | None = None
    mcts_simulations: int = 32
    mcts_c_puct: float = 1.5
    mcts_root_exploration: float = 0.25
    learning_rate: float = 1e-3
    policy_hidden_dim: int = 128
    value_hidden_dim: int = 64
    eval_games: int = 10
    seed: int | None = None
    start_fen: str | None = None
    save_every: int = 10
