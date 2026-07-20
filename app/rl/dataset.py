from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingSample:
    fen: str
    legal_moves: tuple[str, ...]
    chosen_move: str
    side_to_move: str
    result: str
    value_target: float
    policy_target: dict[str, float] | None = None
    game_id: str = ""
    ply: int = 0


@dataclass(slots=True)
class SelfPlayEpisode:
    game_id: str
    result: str
    termination: str
    samples: list[TrainingSample] = field(default_factory=list)
    pgn: str = ""
