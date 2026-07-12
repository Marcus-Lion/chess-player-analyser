from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import chess
import chess.pgn

from app.games import _calculate_control, _calculate_material, _calculate_total_score, _result_summary


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SELF_PLAY_RESULTS_PATH = CACHE_DIR / "self_play_results.jsonl"


@dataclass
class SelfPlayGame:
    index: int
    result: str
    termination: str
    plies: int
    pgn: str
    final_fen: str
    final_score: int
    outcome: str = ""
    winner: str = ""
    loser: str = ""
    run_id: str = ""
    played_at: str = ""
    seed: int | None = None
    top_k: int = 3
    max_plies: int = 200
    start_fen: str = "startpos"


@dataclass
class SelfPlayConfig:
    games: int = 1
    max_plies: int = 200
    top_k: int = 3
    seed: int | None = None
    fen: str | None = None


def _evaluate_board(board: chess.Board) -> int:
    legal_moves = len(list(board.legal_moves))
    c1, c2 = _calculate_control(board)
    material = _calculate_material(board)
    control_score = (c1["White"] + c2["White"]) - (c1["Black"] + c2["Black"])
    material_score = material["White"] - material["Black"]
    return _calculate_total_score(legal_moves, material_score, control_score)


def _move_utility(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    mover = board.turn
    board.push(move)
    try:
        score = _evaluate_board(board)
        utility = score if mover == chess.WHITE else -score
        return utility, score
    finally:
        board.pop()


def _choose_move(board: chess.Board, rng: random.Random, top_k: int) -> tuple[chess.Move, int]:
    scored_moves: list[tuple[int, int, chess.Move]] = []
    for move in board.legal_moves:
        utility, score = _move_utility(board, move)
        scored_moves.append((utility, score, move))

    if not scored_moves:
        raise ValueError("No legal moves available")

    scored_moves.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_n = scored_moves[: max(1, min(top_k, len(scored_moves)))]
    _, score, move = rng.choice(top_n)
    return move, score


def _terminal_reason(board: chess.Board) -> tuple[str, str]:
    if board.is_checkmate():
        return ("1-0" if board.turn == chess.BLACK else "0-1", "checkmate")
    if board.is_stalemate():
        return ("1/2-1/2", "stalemate")
    if board.is_insufficient_material():
        return ("1/2-1/2", "insufficient material")
    return ("", "")


def play_self_game(config: SelfPlayConfig, game_index: int, rng: random.Random | None = None) -> SelfPlayGame:
    rng = rng or random.Random(config.seed)
    board = chess.Board(config.fen) if config.fen else chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-play harness"
    game.headers["Site"] = "Local"
    game.headers["Round"] = str(game_index)
    game.headers["White"] = "Heuristic"
    game.headers["Black"] = "Heuristic"

    node = game
    plies = 0

    result, termination = _terminal_reason(board)
    while plies < config.max_plies and not result:
        move, _ = _choose_move(board, rng, config.top_k)
        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)
        node.comment = san
        plies += 1
        result, termination = _terminal_reason(board)

    if not result:
        result = "1/2-1/2"
        termination = "max plies reached"

    game.headers["Result"] = result
    game.headers["Termination"] = termination

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    pgn_text = game.accept(exporter)
    final_score = _evaluate_board(board)
    summary = _result_summary(result, white="Heuristic", black="Heuristic")

    return SelfPlayGame(
        index=game_index,
        result=result,
        termination=termination,
        plies=plies,
        pgn=pgn_text,
        final_fen=board.fen(),
        final_score=final_score,
        outcome=summary["status"],
        winner=summary["winner"],
        loser=summary["loser"],
    )


def run_self_play(config: SelfPlayConfig) -> list[SelfPlayGame]:
    rng = random.Random(config.seed)
    played_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    start_fen = config.fen or "startpos"
    games: list[SelfPlayGame] = []
    for i in range(config.games):
        game = play_self_game(config, i + 1, rng=rng)
        game.run_id = run_id
        game.played_at = played_at
        game.seed = config.seed
        game.top_k = config.top_k
        game.max_plies = config.max_plies
        game.start_fen = start_fen
        games.append(game)
    return games


def save_self_play_results(games: list[SelfPlayGame]) -> None:
    if not games:
        return

    with SELF_PLAY_RESULTS_PATH.open("a", encoding="utf-8") as handle:
        for game in games:
            payload = {
                "played_at": game.played_at or datetime.now(timezone.utc).isoformat(),
                "run_id": game.run_id,
                "index": game.index,
                "seed": game.seed,
                "top_k": game.top_k,
                "max_plies": game.max_plies,
                "start_fen": game.start_fen,
                "result": game.result,
                "termination": game.termination,
                "plies": game.plies,
                "final_fen": game.final_fen,
                "final_score": game.final_score,
                "outcome": game.outcome,
                "winner": game.winner,
                "loser": game.loser,
                "pgn": game.pgn,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_self_play_results(limit: int = 50) -> list[dict]:
    if not SELF_PLAY_RESULTS_PATH.exists():
        return []

    rows: list[dict] = []
    with SELF_PLAY_RESULTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows[-limit:]


def load_self_play_result(run_id: str, index: int) -> dict | None:
    if not SELF_PLAY_RESULTS_PATH.exists():
        return None

    with SELF_PLAY_RESULTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") == run_id and int(row.get("index", 0)) == index:
                return row
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the position scorer against itself.")
    parser.add_argument("--games", type=int, default=1, help="Number of self-play games to run.")
    parser.add_argument("--max-plies", type=int, default=200, help="Stop each game after this many plies.")
    parser.add_argument("--top-k", type=int, default=3, help="Randomly choose among the top K evaluated moves.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for move selection.")
    parser.add_argument("--fen", type=str, default=None, help="Optional starting FEN.")
    parser.add_argument("--output", type=Path, default=None, help="Optional file to write PGN output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = SelfPlayConfig(
        games=max(1, args.games),
        max_plies=max(1, args.max_plies),
        top_k=max(1, args.top_k),
        seed=args.seed,
        fen=args.fen,
    )
    games = run_self_play(config)
    save_self_play_results(games)

    if args.output:
        args.output.write_text("\n\n".join(game.pgn for game in games) + "\n", encoding="utf-8")

    for game in games:
        print(f"Game {game.index}: {game.result} after {game.plies} plies ({game.termination}); final score {game.final_score}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
