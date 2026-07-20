from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
import random

import chess

from app.rl.config import RLConfig
from app.rl.dataset import SelfPlayEpisode, TrainingSample
from app.rl.model import ChessRLModel


def _terminal_result(board: chess.Board) -> tuple[str, str]:
    if board.is_checkmate():
        return ("1-0" if board.turn == chess.BLACK else "0-1", "checkmate")
    if board.is_stalemate():
        return ("1/2-1/2", "stalemate")
    if board.is_insufficient_material():
        return ("1/2-1/2", "insufficient material")
    if board.is_fivefold_repetition():
        return ("1/2-1/2", "fivefold repetition")
    if board.is_seventyfive_moves():
        return ("1/2-1/2", "75-move rule")
    if board.can_claim_threefold_repetition():
        return ("1/2-1/2", "3-fold repetition")
    if board.can_claim_fifty_moves():
        return ("1/2-1/2", "fifty-move rule")
    return ("", "")


def _value_for_side(result: str, side_to_move: str) -> float:
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if side_to_move == "White" else -1.0
    if result == "0-1":
        return 1.0 if side_to_move == "Black" else -1.0
    return 0.0


def _sample_from_policy(policy: dict[str, float], rng: random.Random) -> str:
    moves = list(policy.keys())
    weights = [max(0.0, float(policy[move])) for move in moves]
    total = sum(weights)
    if total <= 0:
        return rng.choice(moves)
    threshold = rng.random() * total
    cumulative = 0.0
    for move, weight in zip(moves, weights, strict=False):
        cumulative += weight
        if cumulative >= threshold:
            return move
    return moves[-1]


def play_self_play_game(
    model: ChessRLModel,
    config: RLConfig,
    *,
    seed: int | None = None,
    start_fen: str | None = None,
    game_id: str | None = None,
) -> SelfPlayEpisode:
    rng = random.Random(seed)
    board = chess.Board(start_fen or config.start_fen) if (start_fen or config.start_fen) else chess.Board()
    samples: list[TrainingSample] = []
    game_id = game_id or uuid4().hex
    turn = 0
    result = ""
    termination = ""

    while turn < config.max_turns:
        result, termination = _terminal_result(board)
        if result:
            break

        legal_moves = tuple(move.uci() for move in board.legal_moves)
        if not legal_moves:
            result = board.result(claim_draw=False)
            termination = termination or "no legal moves"
            break

        policy, _value = model.predict(board, list(legal_moves), temperature=config.self_play_temperature)
        if rng.random() < config.self_play_exploration:
            chosen_move = rng.choice(legal_moves)
        else:
            chosen_move = _sample_from_policy(policy, rng)

        samples.append(
            TrainingSample(
                fen=board.fen(),
                legal_moves=legal_moves,
                chosen_move=chosen_move,
                side_to_move="White" if board.turn == chess.WHITE else "Black",
                result="",
                value_target=0.0,
                game_id=game_id,
                ply=turn,
            )
        )
        board.push(chess.Move.from_uci(chosen_move))
        turn += 1

    if not result:
        result, termination = ("1/2-1/2", "max turns reached")

    for sample in samples:
        sample.result = result
        sample.value_target = _value_for_side(result, sample.side_to_move)

    return SelfPlayEpisode(
        game_id=game_id,
        result=result,
        termination=termination,
        samples=samples,
    )


def generate_self_play_batch(
    model: ChessRLModel,
    config: RLConfig,
    *,
    episodes: int | None = None,
    seed: int | None = None,
) -> list[TrainingSample]:
    rng = random.Random(seed)
    total_episodes = max(1, episodes or config.episodes)
    samples: list[TrainingSample] = []
    for episode_index in range(total_episodes):
        episode = play_self_play_game(
            model,
            config,
            seed=rng.randint(0, 2**31 - 1),
            game_id=f"{seed or 'rl'}-{episode_index + 1}",
        )
        samples.extend(episode.samples)
    return samples
