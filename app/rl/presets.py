from __future__ import annotations

PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "episodes": 5,
        "max_turns": 20,
        "batch_size": 4,
        "replay_capacity": 100,
        "learning_rate": 1e-3,
        "temperature": 1.0,
        "exploration": 0.10,
        "eval_games": 3,
    },
    "quick": {
        "episodes": 25,
        "max_turns": 40,
        "batch_size": 16,
        "replay_capacity": 1_000,
        "learning_rate": 1e-3,
        "temperature": 1.0,
        "exploration": 0.10,
        "eval_games": 5,
    },
    "standard": {
        "episodes": 100,
        "max_turns": 80,
        "batch_size": 32,
        "replay_capacity": 10_000,
        "learning_rate": 1e-3,
        "temperature": 1.0,
        "exploration": 0.10,
        "eval_games": 10,
    },
    "long": {
        "episodes": 500,
        "max_turns": 100,
        "batch_size": 64,
        "replay_capacity": 20_000,
        "learning_rate": 5e-4,
        "temperature": 1.0,
        "exploration": 0.10,
        "eval_games": 20,
    },
}
