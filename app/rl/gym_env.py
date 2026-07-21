from __future__ import annotations

from dataclasses import dataclass
import random

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.games import _evaluate_position, choose_engine_move
from app.rl.encoding import BOARD_FEATURE_DIM, encode_board

PROMOTION_PIECES: tuple[chess.PieceType | None, ...] = (
    None,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)
ACTION_SPACE_SIZE = 64 * 64 * len(PROMOTION_PIECES)
OBSERVATION_DIM = BOARD_FEATURE_DIM + 1


def move_to_action(move: chess.Move) -> int:
    promotion_index = PROMOTION_PIECES.index(move.promotion if move.promotion in PROMOTION_PIECES else None)
    return ((move.from_square * 64) + move.to_square) * len(PROMOTION_PIECES) + promotion_index


def action_to_move(action: int) -> chess.Move:
    if action < 0 or action >= ACTION_SPACE_SIZE:
        raise ValueError(f"Action out of range: {action}")
    move_slot, promotion_index = divmod(int(action), len(PROMOTION_PIECES))
    from_square, to_square = divmod(move_slot, 64)
    promotion = PROMOTION_PIECES[promotion_index]
    return chess.Move(from_square=from_square, to_square=to_square, promotion=promotion)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[move_to_action(move)] = True
    return mask


@dataclass(slots=True)
class ChessGymConfig:
    max_agent_turns: int = 100
    agent_color: str = "random"  # "white", "black", or "random"
    start_fen: str | None = None
    opponent_depth: int = 3
    opponent_top_k: int = 1
    illegal_move_penalty: float = -1.0
    step_penalty: float = -0.005
    shaping_scale: float = 0.01
    terminal_win_reward: float = 1.0
    terminal_loss_reward: float = -1.0
    terminal_draw_reward: float = 0.0


class ChessMatchEnv(gym.Env):
    """Single-agent chess environment against the heuristic baseline.

    The agent acts on one color and the environment auto-plays the baseline
    engine for the opponent. Rewards combine a small shaping term based on the
    existing static evaluator with terminal win/loss outcomes.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: ChessGymConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ChessGymConfig()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.board = chess.Board(self.config.start_fen) if self.config.start_fen else chess.Board()
        self.agent_color = chess.WHITE
        self.agent_turns = 0
        self._rng = random.Random()

    def _choose_agent_color(self, options: dict | None = None) -> chess.Color:
        choice = (options or {}).get("agent_color", self.config.agent_color)
        if choice == "white":
            return chess.WHITE
        if choice == "black":
            return chess.BLACK
        if choice == "random":
            return self._rng.choice((chess.WHITE, chess.BLACK))
        raise ValueError(f"Unsupported agent_color: {choice!r}")

    def _make_board(self, options: dict | None = None) -> chess.Board:
        start_fen = (options or {}).get("start_fen", self.config.start_fen)
        return chess.Board(start_fen) if start_fen else chess.Board()

    def _agent_score(self) -> float:
        score = float(_evaluate_position(self.board))
        return score if self.agent_color == chess.WHITE else -score

    def _result_to_reward(self, result: str) -> float:
        if result == "1/2-1/2":
            return self.config.terminal_draw_reward
        if result == "1-0":
            return self.config.terminal_win_reward if self.agent_color == chess.WHITE else self.config.terminal_loss_reward
        if result == "0-1":
            return self.config.terminal_win_reward if self.agent_color == chess.BLACK else self.config.terminal_loss_reward
        return 0.0

    def _advance_opponent(self) -> None:
        while not self.board.is_game_over(claim_draw=False) and self.board.turn != self.agent_color:
            move, _score = choose_engine_move(
                self.board,
                depth=self.config.opponent_depth,
                top_k=self.config.opponent_top_k,
            )
            self.board.push(move)

    def _observation(self) -> np.ndarray:
        obs = np.empty(OBSERVATION_DIM, dtype=np.float32)
        obs[:BOARD_FEATURE_DIM] = encode_board(self.board)
        obs[BOARD_FEATURE_DIM] = 1.0 if self.agent_color == chess.WHITE else 0.0
        return obs

    def action_masks(self) -> np.ndarray:
        if self.board.is_game_over(claim_draw=False):
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        return legal_action_mask(self.board)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = random.Random(seed)
        self.board = self._make_board(options)
        self.agent_color = self._choose_agent_color(options)
        self.agent_turns = 0
        self._advance_opponent()
        info = {
            "agent_color": "white" if self.agent_color == chess.WHITE else "black",
            "legal_actions": int(self.action_masks().sum()),
        }
        return self._observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.board.is_game_over(claim_draw=False):
            result = self.board.result(claim_draw=False)
            info = {
                "result": result,
                "termination": "game_over",
                "agent_color": "white" if self.agent_color == chess.WHITE else "black",
                "legal_actions": 0,
            }
            return self._observation(), 0.0, True, False, info

        before_score = self._agent_score()
        legal_moves = {move_to_action(move): move for move in self.board.legal_moves}
        move = legal_moves.get(int(action))
        if move is None:
            info = {
                "illegal_move": True,
                "result": "",
                "termination": "illegal_move",
                "agent_color": "white" if self.agent_color == chess.WHITE else "black",
                "legal_actions": len(legal_moves),
            }
            return self._observation(), self.config.illegal_move_penalty, True, False, info

        self.board.push(move)
        self.agent_turns += 1

        terminated = False
        truncated = self.agent_turns >= self.config.max_agent_turns
        if not self.board.is_game_over(claim_draw=False):
            self._advance_opponent()

        after_score = self._agent_score()
        reward = self.config.step_penalty + self.config.shaping_scale * (after_score - before_score)

        result = self.board.result(claim_draw=False) if self.board.is_game_over(claim_draw=False) else ""
        termination = ""
        if result:
            terminated = True
            termination = "terminal"
            reward += self._result_to_reward(result)
        elif truncated:
            termination = "max_agent_turns"

        info = {
            "result": result,
            "termination": termination,
            "agent_color": "white" if self.agent_color == chess.WHITE else "black",
            "legal_actions": int(self.action_masks().sum()),
            "agent_turns": self.agent_turns,
            "board_fen": self.board.fen(),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        board_text = str(self.board)
        if self.render_mode == "human":
            print(board_text)
            return None
        return board_text

    def close(self) -> None:
        return None
