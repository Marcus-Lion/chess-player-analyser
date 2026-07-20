from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random

import chess
import numpy as np

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


def _sample_index(probs: np.ndarray, rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += float(prob)
        if draw <= cumulative:
            return idx
    return max(0, len(probs) - 1)


@dataclass(slots=True)
class TrainMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    samples: int = 0

    @property
    def total_loss(self) -> float:
        return self.policy_loss + self.value_loss


class ChessRLModel:
    """Small NumPy policy/value network.

    This is intentionally simple: one hidden layer for the policy head and one
    hidden layer for the value head. It is enough to start the reinforcement-
    learning loop without adding a heavyweight ML dependency.
    """

    def __init__(
        self,
        *,
        policy_w1: np.ndarray,
        policy_b1: np.ndarray,
        policy_w2: np.ndarray,
        policy_b2: float,
        value_w1: np.ndarray,
        value_b1: np.ndarray,
        value_w2: np.ndarray,
        value_b2: float,
    ) -> None:
        self.policy_w1 = policy_w1.astype(np.float32)
        self.policy_b1 = policy_b1.astype(np.float32)
        self.policy_w2 = policy_w2.astype(np.float32)
        self.policy_b2 = float(policy_b2)
        self.value_w1 = value_w1.astype(np.float32)
        self.value_b1 = value_b1.astype(np.float32)
        self.value_w2 = value_w2.astype(np.float32)
        self.value_b2 = float(value_b2)

    @classmethod
    def initialize(
        cls,
        *,
        seed: int | None = None,
        policy_hidden_dim: int = 128,
        value_hidden_dim: int = 64,
    ) -> "ChessRLModel":
        rng = np.random.default_rng(seed)
        policy_input_dim = BOARD_FEATURE_DIM + MOVE_FEATURE_DIM
        policy_w1 = rng.normal(0.0, 0.02, size=(policy_input_dim, policy_hidden_dim)).astype(np.float32)
        policy_b1 = np.zeros(policy_hidden_dim, dtype=np.float32)
        policy_w2 = rng.normal(0.0, 0.02, size=(policy_hidden_dim,)).astype(np.float32)
        policy_b2 = 0.0
        value_w1 = rng.normal(0.0, 0.02, size=(BOARD_FEATURE_DIM, value_hidden_dim)).astype(np.float32)
        value_b1 = np.zeros(value_hidden_dim, dtype=np.float32)
        value_w2 = rng.normal(0.0, 0.02, size=(value_hidden_dim,)).astype(np.float32)
        value_b2 = 0.0
        return cls(
            policy_w1=policy_w1,
            policy_b1=policy_b1,
            policy_w2=policy_w2,
            policy_b2=policy_b2,
            value_w1=value_w1,
            value_b1=value_b1,
            value_w2=value_w2,
            value_b2=value_b2,
        )

    def _policy_forward(self, board_feat: np.ndarray, legal_moves: list[chess.Move], board: chess.Board) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        logits: list[float] = []
        caches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for move in legal_moves:
            move_feat = encode_move(board, move)
            x = np.concatenate((board_feat, move_feat)).astype(np.float32)
            z1 = x @ self.policy_w1 + self.policy_b1
            h = _relu(z1)
            logit = float(h @ self.policy_w2 + self.policy_b2)
            logits.append(logit)
            caches.append((x, z1, h))
        return np.asarray(logits, dtype=np.float32), caches

    def _value_forward(self, board_feat: np.ndarray) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
        z1 = board_feat @ self.value_w1 + self.value_b1
        h = _relu(z1)
        raw = float(h @ self.value_w2 + self.value_b2)
        value = math.tanh(raw)
        return value, (z1, h)

    def predict(
        self,
        board_or_fen: chess.Board | str,
        legal_moves: list[str],
        *,
        temperature: float = 1.0,
    ) -> tuple[dict[str, float], float]:
        board = board_or_fen if isinstance(board_or_fen, chess.Board) else chess.Board(board_or_fen)
        legal_move_objs = [chess.Move.from_uci(move) for move in legal_moves]
        board_feat = encode_board(board)
        logits, _ = self._policy_forward(board_feat, legal_move_objs, board)
        if temperature <= 0:
            scaled = logits
        else:
            scaled = logits / max(temperature, 1e-6)
        probs = _softmax(scaled)
        policy = {move: float(prob) for move, prob in zip(legal_moves, probs, strict=False)}
        value, _ = self._value_forward(board_feat)
        return policy, value

    def choose_move(
        self,
        board: chess.Board,
        *,
        temperature: float = 1.0,
        rng: random.Random | None = None,
    ) -> chess.Move:
        rng = rng or random.Random()
        legal_moves = [move.uci() for move in board.legal_moves]
        if not legal_moves:
            raise ValueError("No legal moves available")
        policy, _ = self.predict(board, legal_moves, temperature=temperature)
        probs = np.asarray([policy[move] for move in legal_moves], dtype=np.float32)
        if temperature <= 0:
            return chess.Move.from_uci(legal_moves[int(np.argmax(probs))])
        index = _sample_index(probs / max(float(np.sum(probs)), 1e-8), rng)
        return chess.Move.from_uci(legal_moves[index])

    def train_batch(self, samples: list["TrainingSample"], *, learning_rate: float = 1e-3) -> TrainMetrics:
        from app.rl.dataset import TrainingSample  # local import to avoid cycles

        if not samples:
            return TrainMetrics()

        policy_w1_grad = np.zeros_like(self.policy_w1)
        policy_b1_grad = np.zeros_like(self.policy_b1)
        policy_w2_grad = np.zeros_like(self.policy_w2)
        policy_b2_grad = 0.0
        value_w1_grad = np.zeros_like(self.value_w1)
        value_b1_grad = np.zeros_like(self.value_b1)
        value_w2_grad = np.zeros_like(self.value_w2)
        value_b2_grad = 0.0
        policy_loss = 0.0
        value_loss = 0.0

        for sample in samples:
            board = chess.Board(sample.fen)
            board_feat = encode_board(board)
            legal_move_objs = [chess.Move.from_uci(move) for move in sample.legal_moves]
            logits, caches = self._policy_forward(board_feat, legal_move_objs, board)
            probs = _softmax(logits)
            target = np.zeros_like(probs)
            if sample.policy_target:
                total = float(sum(sample.policy_target.values()))
                if total > 0:
                    for idx, move in enumerate(sample.legal_moves):
                        target[idx] = float(sample.policy_target.get(move, 0.0)) / total
                else:
                    target[int(np.argmax([move == sample.chosen_move for move in sample.legal_moves]))] = 1.0
            else:
                target[int(np.argmax([move == sample.chosen_move for move in sample.legal_moves]))] = 1.0

            policy_loss -= float(np.sum(target * np.log(np.clip(probs, 1e-8, 1.0))))
            dlogits = probs - target
            for grad, (x, z1, h) in zip(dlogits, caches, strict=False):
                policy_w2_grad += h * grad
                policy_b2_grad += float(grad)
                dh = grad * self.policy_w2
                dz1 = dh * _relu_grad(z1)
                policy_w1_grad += np.outer(x, dz1)
                policy_b1_grad += dz1

            value_pred, (z1_v, h_v) = self._value_forward(board_feat)
            dv = (value_pred - sample.value_target)
            value_loss += 0.5 * dv * dv
            d_raw = dv * (1.0 - value_pred * value_pred)
            value_w2_grad += h_v * d_raw
            value_b2_grad += float(d_raw)
            dh_v = d_raw * self.value_w2
            dz1_v = dh_v * _relu_grad(z1_v)
            value_w1_grad += np.outer(board_feat, dz1_v)
            value_b1_grad += dz1_v

        scale = learning_rate / float(len(samples))
        self.policy_w1 -= scale * policy_w1_grad
        self.policy_b1 -= scale * policy_b1_grad
        self.policy_w2 -= scale * policy_w2_grad
        self.policy_b2 -= scale * policy_b2_grad
        self.value_w1 -= scale * value_w1_grad
        self.value_b1 -= scale * value_b1_grad
        self.value_w2 -= scale * value_w2_grad
        self.value_b2 -= scale * value_b2_grad

        return TrainMetrics(
            policy_loss=policy_loss / len(samples),
            value_loss=value_loss / len(samples),
            samples=len(samples),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            policy_w1=self.policy_w1,
            policy_b1=self.policy_b1,
            policy_w2=self.policy_w2,
            policy_b2=np.asarray(self.policy_b2, dtype=np.float32),
            value_w1=self.value_w1,
            value_b1=self.value_b1,
            value_w2=self.value_w2,
            value_b2=np.asarray(self.value_b2, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ChessRLModel":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            policy_w1=data["policy_w1"],
            policy_b1=data["policy_b1"],
            policy_w2=data["policy_w2"],
            policy_b2=float(data["policy_b2"]),
            value_w1=data["value_w1"],
            value_b1=data["value_b1"],
            value_w2=data["value_w2"],
            value_b2=float(data["value_b2"]),
        )
