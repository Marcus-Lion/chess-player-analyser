from __future__ import annotations

from dataclasses import dataclass
import random

import chess

from app.games import choose_engine_move
from app.rl.config import RLConfig
from app.rl.model import ChessRLModel


@dataclass(slots=True)
class MatchSummary:
    games: int
    white_wins: int
    black_wins: int
    draws: int

    @property
    def white_score(self) -> float:
        return (self.white_wins + 0.5 * self.draws) / max(1, self.games)


def _play_model_vs_heuristic_game(
    model: ChessRLModel,
    *,
    max_turns: int,
    seed: int | None = None,
) -> str:
    rng = random.Random(seed)
    board = chess.Board()
    turn = 0
    while turn < max_turns and not board.is_game_over(claim_draw=False):
        if board.turn == chess.WHITE:
            move = model.choose_move(board, temperature=0.0, rng=rng)
        else:
            move, _score = choose_engine_move(board, rng=rng, top_k=1)
        board.push(move)
        turn += 1
    return board.result(claim_draw=False)


def evaluate_matchup(
    model: ChessRLModel,
    config: RLConfig,
    *,
    games: int | None = None,
    seed: int | None = None,
) -> MatchSummary:
    total_games = max(1, games or config.eval_games)
    rng = random.Random(seed if seed is not None else config.seed)
    white_wins = black_wins = draws = 0
    for _ in range(total_games):
        result = _play_model_vs_heuristic_game(model, max_turns=config.max_turns, seed=rng.randint(0, 2**31 - 1))
        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1
        else:
            draws += 1
    return MatchSummary(
        games=total_games,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
    )
