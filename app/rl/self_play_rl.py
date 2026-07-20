from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from uuid import uuid4
import random

import chess
import numpy as np

from app.rl.config import RLConfig
from app.rl.dataset import SelfPlayEpisode, TrainingSample
from app.rl.model import ChessRLModel


_DIRICHLET_ALPHA = 0.3


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


def _terminal_value(board: chess.Board) -> float | None:
    if board.is_checkmate():
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fivefold_repetition()
        or board.is_seventyfive_moves()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
    ):
        return 0.0
    return None


def _visit_distribution(visits: dict[str, int]) -> dict[str, float]:
    total = float(sum(max(0, count) for count in visits.values()))
    if total <= 0.0:
        return {}
    return {move: max(0.0, float(count)) / total for move, count in visits.items() if count > 0}


@dataclass(slots=True)
class _MCTSNode:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[str, "_MCTSNode"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _apply_root_noise(policy: dict[str, float], *, rng: random.Random, epsilon: float) -> dict[str, float]:
    if epsilon <= 0.0 or len(policy) <= 1:
        return policy
    moves = list(policy.keys())
    priors = np.asarray([max(0.0, float(policy[move])) for move in moves], dtype=np.float32)
    total = float(np.sum(priors))
    if total <= 0.0:
        priors = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)
    else:
        priors /= total
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).dirichlet([_DIRICHLET_ALPHA] * len(moves))
    blended = {
        move: (1.0 - epsilon) * float(prior) + epsilon * float(noisy)
        for move, prior, noisy in zip(moves, priors, noise, strict=False)
    }
    total_blended = float(sum(blended.values()))
    if total_blended <= 0.0:
        return {move: 1.0 / len(blended) for move in blended}
    return {move: value / total_blended for move, value in blended.items()}


def _expand_node(
    node: _MCTSNode,
    board: chess.Board,
    model: ChessRLModel,
    *,
    rng: random.Random,
    is_root: bool,
    exploration: float,
) -> float:
    legal_moves = [move.uci() for move in board.legal_moves]
    if not legal_moves:
        return 0.0
    policy, value = model.predict(board, legal_moves, temperature=1.0)
    if is_root:
        policy = _apply_root_noise(policy, rng=rng, epsilon=exploration)
    node.children = {
        move: _MCTSNode(prior=max(0.0, float(policy.get(move, 0.0))))
        for move in legal_moves
    }
    node.expanded = True
    return value


def _select_child(node: _MCTSNode, *, c_puct: float) -> tuple[str, _MCTSNode]:
    best_move = ""
    best_child = None
    best_score = float("-inf")
    sqrt_visits = math.sqrt(max(1, node.visits))
    for move, child in node.children.items():
        q = child.value
        u = c_puct * child.prior * sqrt_visits / (1 + child.visits)
        score = q + u
        if score > best_score:
            best_move = move
            best_child = child
            best_score = score
    if best_child is None:
        raise RuntimeError("MCTS selection failed to choose a child")
    return best_move, best_child


def _run_mcts(
    board: chess.Board,
    model: ChessRLModel,
    *,
    simulations: int,
    c_puct: float,
    root_exploration: float,
    rng: random.Random,
) -> _MCTSNode:
    root = _MCTSNode()

    for _ in range(max(1, simulations)):
        search_board = board.copy(stack=False)
        node = root
        path = [node]

        while True:
            terminal_value = _terminal_value(search_board)
            if terminal_value is not None:
                value = terminal_value
                break

            if not node.expanded:
                value = _expand_node(
                    node,
                    search_board,
                    model,
                    rng=rng,
                    is_root=(node is root),
                    exploration=root_exploration,
                )
                break

            move, child = _select_child(node, c_puct=c_puct)
            search_board.push(chess.Move.from_uci(move))
            node = child
            path.append(node)

        for current in reversed(path):
            current.visits += 1
            current.value_sum += value
            value = -value

    return root


def _choose_move_from_visits(
    visits: dict[str, int],
    *,
    temperature: float,
    rng: random.Random,
    fallback_policy: dict[str, float],
) -> str:
    if not visits:
        if fallback_policy:
            if temperature <= 0:
                return max(fallback_policy.items(), key=lambda item: item[1])[0]
            return _sample_from_policy(fallback_policy, rng)
        return ""

    moves = list(visits.keys())
    counts = np.asarray([max(0.0, float(visits[move])) for move in moves], dtype=np.float32)
    if float(np.sum(counts)) <= 0.0:
        return _sample_from_policy({move: 1.0 for move in moves}, rng)
    if temperature <= 0:
        return moves[int(np.argmax(counts))]
    scaled = counts ** (1.0 / max(temperature, 1e-6))
    total = float(np.sum(scaled))
    if total <= 0.0:
        return moves[int(np.argmax(counts))]
    probabilities = {move: float(weight) / total for move, weight in zip(moves, scaled, strict=False)}
    return _sample_from_policy(probabilities, rng)


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

        root = _run_mcts(
            board,
            model,
            simulations=config.mcts_simulations,
            c_puct=config.mcts_c_puct,
            root_exploration=config.mcts_root_exploration if config.self_play_exploration > 0 else 0.0,
            rng=rng,
        )
        visits = {move: child.visits for move, child in root.children.items() if child.visits > 0}
        policy_target = _visit_distribution(visits)
        fallback_policy = {move: float(child.prior) for move, child in root.children.items() if child.prior > 0}
        if not policy_target:
            policy_target = fallback_policy
        if not policy_target:
            policy_target = {move: 1.0 for move in legal_moves}
        chosen_move = _choose_move_from_visits(
            visits,
            temperature=config.self_play_temperature,
            rng=rng,
            fallback_policy=fallback_policy or {move: 1.0 for move in legal_moves},
        )

        samples.append(
            TrainingSample(
                fen=board.fen(),
                legal_moves=legal_moves,
                chosen_move=chosen_move,
                side_to_move="White" if board.turn == chess.WHITE else "Black",
                result="",
                value_target=0.0,
                policy_target=policy_target,
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
