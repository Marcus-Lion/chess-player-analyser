from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from collections import Counter
import math
from uuid import uuid4
import random

import chess
import numpy as np

from app.rl.config import RLConfig
from app.rl.dataset import SelfPlayEpisode, TrainingSample
from app.rl.model import ChessRLModel, _softmax
from app.games import _evaluate_position


_DIRICHLET_ALPHA = 0.3


def _is_perpetual_check(
    board: chess.Board,
    *,
    lookback_plies: int = 16,
    min_king_moves: int = 4,
) -> bool:
    """Heuristic: classify a threefold draw as "perpetual check".

    Matches a recent suffix where one side gives check every turn and the
    defending king moves back and forth between the same two squares.
    """
    needed_plies = 2 * min_king_moves
    if min_king_moves <= 0 or len(board.move_stack) < needed_plies:
        return False

    tmp = board.copy(stack=True)
    after = tmp.copy(stack=False)
    records: list[tuple[bool, bool, bool, int | None]] = []
    for _ in range(min(int(lookback_plies), len(tmp.move_stack))):
        move = tmp.pop()
        mover = not after.turn
        gave_check = after.is_check()
        piece = tmp.piece_at(move.from_square)
        moved_king = piece is not None and piece.piece_type == chess.KING
        king_to = move.to_square if moved_king else None
        records.append((mover, gave_check, moved_king, king_to))
        after = tmp.copy(stack=False)
    if len(records) < needed_plies:
        return False

    records.reverse()
    suffix = records[-needed_plies:]
    for defender in (chess.WHITE, chess.BLACK):
        attacker = not defender
        defender_moves = [r for r in suffix if r[0] == defender]
        attacker_moves = [r for r in suffix if r[0] == attacker]
        if len(defender_moves) != min_king_moves or len(attacker_moves) != min_king_moves:
            continue
        if not all(moved_king for (_, _, moved_king, _) in defender_moves):
            continue
        king_squares = [sq for (_, _, _, sq) in defender_moves]
        if any(sq is None for sq in king_squares):
            continue
        if len(set(king_squares)) != 2:
            continue
        if any(king_squares[i] != king_squares[i % 2] for i in range(len(king_squares))):
            continue
        if not all(gave_check for (_, gave_check, _, _) in attacker_moves):
            continue
        return True

    return False


def _can_claim_threefold(board: chess.Board, repetition_counts: Counter) -> bool:
    """Check threefold claims from counts instead of replaying the full stack."""
    if not any(count >= 2 for count in repetition_counts.values()):
        return False
    if repetition_counts[board._transposition_key()] >= 3:
        return True
    for move in board.generate_legal_moves():
        board.push(move)
        try:
            if repetition_counts[board._transposition_key()] >= 2:
                return True
        finally:
            board.pop()
    return False


def _terminal_result(
    board: chess.Board,
    repetition_counts: Counter | None = None,
) -> tuple[str, str]:
    if board.is_checkmate():
        return ("1-0" if board.turn == chess.BLACK else "0-1", "checkmate")
    if board.is_stalemate():
        return ("1/2-1/2", "stalemate")
    if board.is_insufficient_material():
        return ("1/2-1/2", "insufficient material")
    if (repetition_counts is None and board.is_fivefold_repetition()) or (
        repetition_counts is not None and repetition_counts[board._transposition_key()] >= 5
    ):
        return ("1/2-1/2", "5-fold-rep")
    if board.is_seventyfive_moves():
        return ("1/2-1/2", "75-moves")
    if (repetition_counts is None and board.can_claim_threefold_repetition()) or (
        repetition_counts is not None and _can_claim_threefold(board, repetition_counts)
    ):
        if _is_perpetual_check(board):
            return ("1/2-1/2", "perpetual check")
        return ("1/2-1/2", "3-fold-rep")
    if board.can_claim_fifty_moves():
        return ("1/2-1/2", "50-moves")
    return ("", "")


def _value_for_side(result: str, side_to_move: str) -> float:
    if result == "1/2-1/2":
        return 0.0
    if result == "1-0":
        return 1.0 if side_to_move == "White" else -1.0
    if result == "0-1":
        return 1.0 if side_to_move == "Black" else -1.0
    return 0.0


def _max_turn_draw_value(board: chess.Board, side_to_move: str) -> float:
    """Use the final heuristic position as a learning signal for max-turn draws."""
    score = float(_evaluate_position(board))
    value = math.tanh(score / 10.0)
    return value if side_to_move == "White" else -value


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


def _terminal_value(board: chess.Board, repetition_counts: Counter | None = None) -> float | None:
    if board.is_checkmate():
        return -1.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or (
            (repetition_counts is None and board.is_fivefold_repetition())
            or (repetition_counts is not None and repetition_counts[board._transposition_key()] >= 5)
        )
        or board.is_seventyfive_moves()
        or (
            (repetition_counts is None and board.can_claim_threefold_repetition())
            or (repetition_counts is not None and _can_claim_threefold(board, repetition_counts))
        )
        or board.can_claim_fifty_moves()
    ):
        return 0.0
    return None


def _visit_distribution(visits: dict[str, int]) -> dict[str, float]:
    total = float(sum(max(0, count) for count in visits.values()))
    if total <= 0.0:
        return {}
    return {move: max(0.0, float(count)) / total for move, count in visits.items() if count > 0}


def _ranked_root_policy(
    visits: dict[str, int],
    fallback_policy: dict[str, float],
    *,
    temperature: float,
    ply: int,
) -> dict[str, float]:
    """Return a sharper self-play policy from search visits and network priors.

    The old policy sampled directly from visit counts, which stayed close to
    uniform when the search signal was weak. This version keeps the search
    signal but sharpens it by combining:

    - visit counts
    - network priors from the root
    - a small top-k truncation
    - a lower temperature later in the game
    """

    if not visits:
        if not fallback_policy:
            return {}
        total = float(sum(max(0.0, float(v)) for v in fallback_policy.values()))
        if total <= 0.0:
            return {move: 1.0 / len(fallback_policy) for move in fallback_policy}
        return {move: max(0.0, float(v)) / total for move, v in fallback_policy.items()}

    moves = list(visits.keys())
    counts = np.asarray([max(0.0, float(visits[move])) for move in moves], dtype=np.float32)
    total_counts = float(np.sum(counts))
    prior_probs = np.asarray([max(0.0, float(fallback_policy.get(move, 0.0))) for move in moves], dtype=np.float32)
    prior_total = float(np.sum(prior_probs))
    if prior_total > 0.0:
        prior_probs /= prior_total
    elif len(moves) > 0:
        prior_probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)

    effective_temperature = max(0.15, float(temperature))
    if ply >= 24:
        effective_temperature = min(effective_temperature, 0.65)
    if ply >= 48:
        effective_temperature = min(effective_temperature, 0.35)

    # Blend visit counts with priors, then keep only the most promising few moves.
    blended_scores = np.log1p(counts) + 0.25 * np.log1p(prior_probs * max(total_counts, 1.0))
    top_k = min(6, len(moves))
    if top_k <= 0:
        return {}
    top_indices = np.argsort(blended_scores)[-top_k:]
    top_moves = [moves[idx] for idx in top_indices]
    top_logits = blended_scores[top_indices].astype(np.float32) / effective_temperature
    top_probs = _softmax(top_logits)
    return {move: float(prob) for move, prob in zip(top_moves, top_probs, strict=False)}


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
        search_board = board.copy(stack=True)
        repetition_counts = Counter({board._transposition_key(): 1})
        node = root
        path = [node]

        while True:
            terminal_value = _terminal_value(search_board, repetition_counts)
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
            repetition_counts[search_board._transposition_key()] += 1
            node = child
            path.append(node)

        for current in reversed(path):
            current.visits += 1
            current.value_sum += value
            value = -value

    return root


def _choose_move_from_visits(
    policy: dict[str, float],
    *,
    rng: random.Random,
) -> str:
    if not policy:
        return ""
    return _sample_from_policy(policy, rng)


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
    repetition_counts = Counter({board._transposition_key(): 1})

    while turn < config.max_turns:
        result, termination = _terminal_result(board, repetition_counts)
        if result:
            break

        legal_moves = tuple(move.uci() for move in board.legal_moves)
        if not legal_moves:
            result = board.result(claim_draw=False)
            termination = termination or "no legal moves"
            break

        _, root_value = model.predict(board, list(legal_moves), temperature=1.0)

        root = _run_mcts(
            board,
            model,
            simulations=config.mcts_simulations,
            c_puct=config.mcts_c_puct,
            root_exploration=config.mcts_root_exploration if config.self_play_exploration > 0 else 0.0,
            rng=rng,
        )
        visits = {move: child.visits for move, child in root.children.items() if child.visits > 0}
        fallback_policy = {move: float(child.prior) for move, child in root.children.items() if child.prior > 0}
        policy_target = _ranked_root_policy(
            visits,
            fallback_policy,
            temperature=config.self_play_temperature,
            ply=turn,
        )
        if not policy_target:
            policy_target = _visit_distribution(visits)
        if not policy_target:
            policy_target = fallback_policy
        if not policy_target:
            policy_target = {move: 1.0 for move in legal_moves}
        chosen_move = _choose_move_from_visits(
            policy_target,
            rng=rng,
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
        repetition_counts[board._transposition_key()] += 1
        turn += 1

    if not result:
        result, termination = ("1/2-1/2", "max turns")

    for sample in samples:
        sample.result = result
        if result == "1/2-1/2" and termination == "max turns":
            sample.value_target = _max_turn_draw_value(board, sample.side_to_move)
        else:
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
