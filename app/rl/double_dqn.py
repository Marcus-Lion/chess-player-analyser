from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import math
import random

import chess
import numpy as np

from app.games import choose_engine_move
from app.rl.encoding import BOARD_FEATURE_DIM, MOVE_FEATURE_DIM, encode_board, encode_move


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(logits, 1.0 / logits.size)
    return exp / total


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


def _reward_for_result(result: str, side_to_move: str) -> float:
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if side_to_move == "White" else -1.0
    if result == "0-1":
        return 1.0 if side_to_move == "Black" else -1.0
    return 0.0


@dataclass(slots=True)
class DQNTransition:
    fen: str
    legal_moves: tuple[str, ...]
    action: str
    reward: float
    next_fen: str
    next_legal_moves: tuple[str, ...]
    done: bool
    side_to_move: str
    result: str = ""
    game_id: str = ""
    ply: int = 0


@dataclass(slots=True)
class DQNTrainingStep:
    episode: int
    transitions: int
    loss: float
    epsilon: float
    buffer_size: int
    result: str


@dataclass(slots=True)
class DQNTrainingRun:
    steps: list[DQNTrainingStep] = field(default_factory=list)

    @property
    def latest(self) -> DQNTrainingStep | None:
        return self.steps[-1] if self.steps else None


@dataclass(slots=True)
class DoubleDQNConfig:
    episodes: int = 200
    max_turns: int = 100
    replay_capacity: int = 50_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 0.25
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    updates_per_episode: int = 4
    target_update_every: int = 10
    train_warmup: int = 256
    step_penalty: float = -0.01
    hidden_dim: int = 128
    seed: int | None = None
    save_every: int = 10


class TransitionReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: deque[DQNTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add_many(self, samples: list[DQNTransition]) -> None:
        self._items.extend(samples)

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[DQNTransition]:
        if not self._items:
            return []
        rng = rng or random.Random()
        batch_size = max(1, min(int(batch_size), len(self._items)))
        return rng.sample(list(self._items), batch_size)


class DoubleDQNModel:
    """Small state-action network for a Double DQN experiment branch."""

    def __init__(
        self,
        *,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: float,
    ) -> None:
        self.w1 = w1.astype(np.float32)
        self.b1 = b1.astype(np.float32)
        self.w2 = w2.astype(np.float32)
        self.b2 = float(b2)

    @classmethod
    def initialize(cls, *, seed: int | None = None, hidden_dim: int = 128) -> "DoubleDQNModel":
        rng = np.random.default_rng(seed)
        input_dim = BOARD_FEATURE_DIM + MOVE_FEATURE_DIM
        w1 = rng.normal(0.0, 0.02, size=(input_dim, hidden_dim)).astype(np.float32)
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = rng.normal(0.0, 0.02, size=(hidden_dim,)).astype(np.float32)
        b2 = 0.0
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    def clone(self) -> "DoubleDQNModel":
        return DoubleDQNModel(w1=self.w1.copy(), b1=self.b1.copy(), w2=self.w2.copy(), b2=self.b2)

    def copy_from(self, other: "DoubleDQNModel") -> None:
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = float(other.b2)

    def _forward_move(
        self,
        board_feat: np.ndarray,
        legal_moves: list[chess.Move],
        board: chess.Board,
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        q_values: list[float] = []
        caches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for move in legal_moves:
            move_feat = encode_move(board, move)
            x = np.concatenate((board_feat, move_feat)).astype(np.float32)
            z1 = x @ self.w1 + self.b1
            h = _relu(z1)
            q = float(h @ self.w2 + self.b2)
            q_values.append(q)
            caches.append((x, z1, h))
        return np.asarray(q_values, dtype=np.float32), caches

    def predict_qs(self, board: chess.Board, legal_moves: list[str]) -> dict[str, float]:
        if not legal_moves:
            return {}
        board_feat = encode_board(board)
        move_objs = [chess.Move.from_uci(move) for move in legal_moves]
        q_values, _ = self._forward_move(board_feat, move_objs, board)
        return {move: float(q) for move, q in zip(legal_moves, q_values, strict=False)}

    def choose_move(
        self,
        board: chess.Board,
        legal_moves: list[str] | None = None,
        *,
        epsilon: float = 0.05,
        rng: random.Random | None = None,
    ) -> chess.Move:
        rng = rng or random.Random()
        legal_moves = legal_moves or [move.uci() for move in board.legal_moves]
        if not legal_moves:
            raise ValueError("No legal moves available")
        if epsilon > 0.0 and rng.random() < epsilon:
            return chess.Move.from_uci(rng.choice(legal_moves))
        q_values = self.predict_qs(board, legal_moves)
        best_move = max(q_values.items(), key=lambda item: item[1])[0]
        return chess.Move.from_uci(best_move)

    def _predict_q_value(self, board: chess.Board, move_uci: str) -> float:
        move = chess.Move.from_uci(move_uci)
        board_feat = encode_board(board)
        q_values, _ = self._forward_move(board_feat, [move], board)
        return float(q_values[0]) if q_values.size else 0.0

    def train_batch(
        self,
        samples: list[DQNTransition],
        *,
        target_model: "DoubleDQNModel",
        learning_rate: float,
        gamma: float,
    ) -> float:
        if not samples:
            return 0.0

        w1_grad = np.zeros_like(self.w1)
        b1_grad = np.zeros_like(self.b1)
        w2_grad = np.zeros_like(self.w2)
        b2_grad = 0.0
        loss = 0.0

        for sample in samples:
            board = chess.Board(sample.fen)
            board_feat = encode_board(board)
            action_move = chess.Move.from_uci(sample.action)
            q_pred, caches = self._forward_move(board_feat, [action_move], board)
            pred = float(q_pred[0]) if q_pred.size else 0.0

            if sample.done or not sample.next_legal_moves:
                target = float(sample.reward)
            else:
                next_board = chess.Board(sample.next_fen)
                next_legal_moves = list(sample.next_legal_moves)
                next_online_qs = self.predict_qs(next_board, next_legal_moves)
                next_best_move = max(next_online_qs.items(), key=lambda item: item[1])[0]
                target = float(sample.reward) + gamma * target_model._predict_q_value(next_board, next_best_move)

            td_error = pred - target
            loss += 0.5 * td_error * td_error

            x, z1, h = caches[0]
            d_out = td_error
            w2_grad += h * d_out
            b2_grad += float(d_out)
            dh = d_out * self.w2
            dz1 = dh * _relu_grad(z1)
            w1_grad += np.outer(x, dz1)
            b1_grad += dz1

        scale = learning_rate / float(len(samples))
        self.w1 -= scale * w1_grad
        self.b1 -= scale * b1_grad
        self.w2 -= scale * w2_grad
        self.b2 -= scale * b2_grad

        return loss / len(samples)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=np.asarray(self.b2, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DoubleDQNModel":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=float(data["b2"]),
        )


def play_double_dqn_game(
    model: DoubleDQNModel,
    config: DoubleDQNConfig,
    *,
    epsilon: float | None = None,
    seed: int | None = None,
    start_fen: str | None = None,
    game_id: str | None = None,
) -> list[DQNTransition]:
    rng = random.Random(seed)
    board = chess.Board(start_fen) if start_fen else chess.Board()
    game_id = game_id or uuid4().hex
    transitions: list[DQNTransition] = []
    turn = 0
    rollout_epsilon = config.epsilon_start if epsilon is None else max(0.0, float(epsilon))

    while turn < config.max_turns:
        result, _termination = _terminal_result(board)
        if result:
            break

        legal_moves = [move.uci() for move in board.legal_moves]
        if not legal_moves:
            break

        pre_fen = board.fen()
        action = model.choose_move(board, legal_moves, epsilon=rollout_epsilon, rng=rng)
        action_uci = action.uci()
        side_to_move = "White" if board.turn == chess.WHITE else "Black"

        board.push(action)
        turn += 1
        next_legal_moves = [move.uci() for move in board.legal_moves]
        next_result, _ = _terminal_result(board)
        done = bool(next_result) or board.is_game_over(claim_draw=False) or turn >= config.max_turns
        if done and not next_result:
            next_result = "1/2-1/2"

        reward = config.step_penalty
        if done:
            reward += _reward_for_result(next_result, side_to_move)

        transitions.append(
            DQNTransition(
                fen=pre_fen,
                legal_moves=tuple(legal_moves),
                action=action_uci,
                reward=reward,
                next_fen=board.fen(),
                next_legal_moves=tuple(next_legal_moves) if not done else tuple(),
                done=done,
                side_to_move=side_to_move,
                result=next_result,
                game_id=game_id,
                ply=turn - 1,
            )
        )

        if done:
            break

    if transitions and not transitions[-1].result:
        transitions[-1].result = "1/2-1/2"
    return transitions


def _play_model_vs_heuristic_game(
    model: DoubleDQNModel,
    *,
    epsilon: float,
    max_turns: int,
    seed: int | None = None,
) -> str:
    rng = random.Random(seed)
    board = chess.Board()
    turn = 0
    while turn < max_turns and not board.is_game_over(claim_draw=False):
        if board.turn == chess.WHITE:
            legal_moves = [move.uci() for move in board.legal_moves]
            move = model.choose_move(board, legal_moves, epsilon=0.0, rng=rng)
        else:
            move, _score = choose_engine_move(board, rng=rng, top_k=1)
        board.push(move)
        turn += 1
    return board.result(claim_draw=False)


@dataclass(slots=True)
class DoubleDQNEvalSummary:
    games: int
    white_wins: int
    black_wins: int
    draws: int
    white_score: float


def evaluate_matchup(
    model: DoubleDQNModel,
    config: DoubleDQNConfig,
    *,
    games: int = 10,
    seed: int | None = None,
) -> DoubleDQNEvalSummary:
    total_games = max(1, games)
    rng = random.Random(seed if seed is not None else config.seed)
    white_wins = black_wins = draws = 0
    for _ in range(total_games):
        result = _play_model_vs_heuristic_game(
            model,
            epsilon=0.0,
            max_turns=config.max_turns,
            seed=rng.randint(0, 2**31 - 1),
        )
        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1
        else:
            draws += 1
    return DoubleDQNEvalSummary(
        games=total_games,
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        white_score=(white_wins + 0.5 * draws) / max(1, total_games),
    )


def train_from_self_play_double_dqn(
    model: DoubleDQNModel,
    config: DoubleDQNConfig,
    *,
    save_path: str | Path | None = None,
    seed: int | None = None,
    progress_callback=None,
) -> DQNTrainingRun:
    rng = random.Random(seed if seed is not None else config.seed)
    replay = TransitionReplayBuffer(config.replay_capacity)
    target_model = model.clone()
    run = DQNTrainingRun()
    epsilon = float(config.epsilon_start)

    for episode in range(1, config.episodes + 1):
        episode_epsilon = max(config.epsilon_end, epsilon)
        episode_transitions = play_double_dqn_game(
            model,
            config,
            epsilon=episode_epsilon,
            seed=rng.randint(0, 2**31 - 1),
            game_id=f"{seed or 'dqn'}-{episode}",
        )
        replay.add_many(episode_transitions)

        loss = 0.0
        if len(replay) >= config.train_warmup:
            for _ in range(max(1, config.updates_per_episode)):
                batch = replay.sample(config.batch_size, rng=rng)
                loss = model.train_batch(
                    batch,
                    target_model=target_model,
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                )

        if episode % max(1, config.target_update_every) == 0:
            target_model.copy_from(model)

        if save_path is not None and episode % max(1, config.save_every) == 0:
            model.save(save_path)

        last_result = episode_transitions[-1].result if episode_transitions else "1/2-1/2"
        step = DQNTrainingStep(
            episode=episode,
            transitions=len(episode_transitions),
            loss=loss,
            epsilon=epsilon,
            buffer_size=len(replay),
            result=last_result,
        )
        run.steps.append(step)
        if progress_callback is not None:
            progress_callback(step)

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    if save_path is not None:
        model.save(save_path)
    return run
