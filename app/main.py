from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import chess
import plotly.express as px
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.chesscom import ChessComClient
from app.games import (
    FORWARD_SCORE_WEIGHT,
    LEGAL_MOVES_WEIGHT,
    MATERIAL_SCORE_WEIGHT,
    _result_summary,
    choose_engine_move,
    load_game_summaries,
    load_game_detail,
)
from app.parser import parse_pgn_to_dataframe
from app.self_play import (
    SelfPlayConfig,
    load_self_play_job,
    load_self_play_result,
    load_self_play_results,
    prune_old_jobs,
    run_self_play,
    save_self_play_results,
    start_self_play_job,
)
from app.metrics import (
    summarize,
    monthly_performance,
    rolling_performance,
    hourly_performance,
    day_performance,
    time_day_matrix,
    prepost_breakpoint,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PIECE_UNICODE = {
    (chess.PAWN, chess.WHITE): "♙",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.ROOK, chess.WHITE): "♖",
    (chess.QUEEN, chess.WHITE): "♕",
    (chess.KING, chess.WHITE): "♔",
    (chess.PAWN, chess.BLACK): "♟",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.ROOK, chess.BLACK): "♜",
    (chess.QUEEN, chess.BLACK): "♛",
    (chess.KING, chess.BLACK): "♚",
}

app = FastAPI(title="Marcus Lion Chess Player Analyser")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def _cached_paths(username: str) -> tuple[Path, Path]:
    safe = "".join(c for c in username.lower() if c.isalnum() or c in "_-")
    return CACHE_DIR / f"{safe}.pgn", CACHE_DIR / f"{safe}.games.csv"


def _maybe_save_to_neo4j(username: str, df: pd.DataFrame) -> None:
    """Export parsed games to Neo4j when NEO4J_ENABLED is truthy.

    This is fully optional: any failure is swallowed so the analytics
    engine keeps working when Neo4j is not configured or reachable.
    """
    if os.getenv("NEO4J_ENABLED", "").lower() not in ("1", "true", "yes", "on"):
        return
    try:
        from app.neo4j_store import Neo4jStore

        with Neo4jStore() as store:
            store.save_games(username, df)
    except Exception:
        # Neo4j is an optional sink; never break the request because of it.
        pass


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})


def _make_charts(df: pd.DataFrame) -> dict[str, str]:
    charts = {}

    monthly = monthly_performance(df)
    if not monthly.empty:
        fig = px.line(monthly, x="month", y=["performance_rating", "avg_opponent", "avg_user_rating"],
                      markers=True, title="Monthly performance rating", hover_data=["games"])
        charts["monthly"] = _fig_html(fig)

    rolling = rolling_performance(df)
    if not rolling.empty:
        fig = px.line(rolling, x="mid_game", y=["performance_rating", "avg_opponent", "avg_user_rating"],
                      title="Rolling 100-game performance rating", hover_data=["games"])
        charts["rolling"] = _fig_html(fig)

    hourly = hourly_performance(df)
    if not hourly.empty:
        fig = px.line(hourly, x="local_hour", y="performance_rating", markers=True,
                      title="Performance by local start hour", hover_data=["games"])
        charts["hourly"] = _fig_html(fig)

    day = day_performance(df)
    if not day.empty:
        fig = px.bar(day, x="local_day", y="performance_rating", title="Performance by day of week",
                     hover_data=["games"], text="games")
        fig.update_traces(textposition="outside")
        charts["day"] = _fig_html(fig)

    matrix = time_day_matrix(df)
    if not matrix.empty:
        pivot = matrix.pivot(index="time_bucket", columns="day_group", values="performance_rating")
        pivot = pivot.reindex(["6–8 PM", "8–10 PM", "10 PM–Midnight", "After Midnight"])
        pivot_games = matrix.pivot(index="time_bucket", columns="day_group", values="games")
        pivot_games = pivot_games.reindex(pivot.index)
        fig = px.imshow(pivot, text_auto=".0f", aspect="auto", title="Time × day performance heatmap")
        fig.update_traces(
            customdata=pivot_games.values,
            hovertemplate="Time: %{y}<br>Day: %{x}<br>Rating: %{z:.0f}<br>Games: %{customdata}<extra></extra>"
        )
        charts["time_day"] = _fig_html(fig)

    return charts


def _ensure_pgn(username: str, force_refresh: bool = False) -> str:
    """Return the cached PGN for ``username``, fetching it if needed."""
    pgn_path, _ = _cached_paths(username)
    if force_refresh or not pgn_path.exists():
        client = ChessComClient()
        pgn_text = client.fetch_all_pgn(username)
        pgn_path.write_text(pgn_text, encoding="utf-8")
        return pgn_text
    return pgn_path.read_text(encoding="utf-8")


def _normalize_human_color(value: str | None) -> str:
    return "White" if (value or "").strip().lower().startswith("w") else "Black"


def _parse_history(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _play_labels(human_color: str) -> tuple[str, str]:
    if human_color == "White":
        return "Human (White)", "Engine (Black)"
    return "Engine (White)", "Human (Black)"


def _piece_unicode(piece: chess.Piece | None) -> str:
    if piece is None:
        return ""
    return PIECE_UNICODE[(piece.piece_type, piece.color)]


def _board_grid(board: chess.Board, can_drag: bool) -> list[list[dict]]:
    grid: list[list[dict]] = []
    for rank in range(7, -1, -1):
        row: list[dict] = []
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            row.append({
                "square": chess.square_name(square),
                "is_light": (file + rank) % 2 == 0,
                "piece": _piece_unicode(piece),
                "piece_color": "white" if piece and piece.color == chess.WHITE else "black" if piece else "",
                "draggable": bool(piece and can_drag and piece.color == board.turn),
            })
        grid.append(row)
    return grid


def _legal_move_options(board: chess.Board) -> list[dict]:
    options: list[dict] = []
    for move in board.legal_moves:
        options.append({
            "uci": move.uci(),
            "san": board.san(move),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
        })
    return options


def _append_history(history: list[dict], board: chess.Board, move: chess.Move) -> None:
    history.append({
        "ply": len(history) + 1,
        "move_number": board.fullmove_number,
        "side": "White" if board.turn == chess.WHITE else "Black",
        "san": board.san(move),
    })


def _advance_engine(board: chess.Board, history: list[dict], human_is_white: bool, rng: random.Random, top_k: int) -> chess.Move | None:
    last_move: chess.Move | None = None
    while not board.is_game_over(claim_draw=False) and board.turn != human_is_white:
        move, _ = choose_engine_move(board, rng, top_k)
        _append_history(history, board, move)
        board.push(move)
        last_move = move
    return last_move


def _play_context(
    board: chess.Board,
    history: list[dict],
    human_color: str,
    top_k: int,
    seed: int | None,
    last_move: chess.Move | None,
    message: str = "",
    error: str = "",
) -> dict:
    human_label, engine_label = _play_labels(human_color)
    human_is_white = human_color == "White"
    can_move = board.turn == human_is_white and not board.is_game_over(claim_draw=False)
    legal_move_options = _legal_move_options(board) if can_move else []
    result_summary = None
    if board.is_game_over(claim_draw=False):
        white_label = human_label if human_color == "White" else engine_label
        black_label = engine_label if human_color == "White" else human_label
        result_summary = _result_summary(board.result(claim_draw=False), white=white_label, black=black_label)

    return {
        "board_grid": _board_grid(board, can_move),
        "legal_move_options": legal_move_options,
        "history": history,
        "history_json": json.dumps(history, ensure_ascii=False),
        "current_fen": board.fen(),
        "human_color": human_color,
        "human_label": human_label,
        "engine_label": engine_label,
        "can_move": can_move,
        "side_to_move": "White" if board.turn == chess.WHITE else "Black",
        "top_k": top_k,
        "seed": seed,
        "message": message,
        "error": error,
        "result_summary": result_summary,
        "game_over": board.is_game_over(claim_draw=False),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/play", response_class=HTMLResponse)
def play(request: Request, human_color: str = "white", top_k: int = 3, seed: str | None = None):
    human_color_n = _normalize_human_color(human_color)
    seed_value = int(seed) if seed and seed.strip().isdigit() else None
    rng = random.Random(seed_value)
    board = chess.Board()
    history: list[dict] = []
    last_move = _advance_engine(board, history, human_color_n == "White", rng, max(1, top_k))
    message = "Engine moved first." if last_move is not None else "New game ready."
    context = _play_context(board, history, human_color_n, max(1, top_k), seed_value, last_move, message=message)
    return templates.TemplateResponse("play.html", {"request": request, **context})


@app.post("/play", response_class=HTMLResponse)
def play_move(
    request: Request,
    current_fen: str = Form(...),
    human_color: str = Form("white"),
    top_k: int = Form(3),
    seed: str | None = Form(None),
    history_json: str | None = Form(None),
    move_uci: str | None = Form(None),
    move_san: str | None = Form(None),
):
    human_color_n = _normalize_human_color(human_color)
    seed_value = int(seed) if seed and seed.strip().isdigit() else None
    rng = random.Random(seed_value)
    board = chess.Board(current_fen)
    history = _parse_history(history_json)
    error = ""
    message = ""
    last_move: chess.Move | None = None

    if move_san or move_uci:
        try:
            if board.turn != (human_color_n == "White"):
                raise ValueError("It is not your turn.")
            if move_uci:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    raise ValueError("Illegal move.")
            else:
                move = board.parse_san(move_san)
            san_text = board.san(move)
            _append_history(history, board, move)
            board.push(move)
            last_move = move
            message = f"You played {san_text}."
        except Exception as exc:
            error = str(exc)

    if not error:
        engine_move = _advance_engine(board, history, human_color_n == "White", rng, max(1, top_k))
        if engine_move is not None:
            last_move = engine_move
            message = f"{message} Engine replied." if message else "Engine moved."

    context = _play_context(board, history, human_color_n, max(1, top_k), seed_value, last_move, message=message, error=error)
    return templates.TemplateResponse("play.html", {"request": request, **context})


@app.get("/self-play", response_class=HTMLResponse)
def self_play_page(request: Request):
    results = load_self_play_results()
    return templates.TemplateResponse("self_play.html", {
        "request": request,
        "results": results,
        "recent_games": [],
        "config": None,
    })


@app.post("/self-play", response_class=HTMLResponse)
def self_play_run(
    request: Request,
    games: int = Form(1),
    max_plies: int = Form(55),
    top_k: int = Form(3),
    seed: str | None = Form(None),
    fen: str | None = Form(None),
):
    fen = fen.strip() if fen and fen.strip() else None
    try:
        seed_value = int(seed) if seed and seed.strip() else None
    except ValueError:
        seed_value = None
    config = SelfPlayConfig(
        games=max(1, games),
        max_plies=max(1, max_plies),
        top_k=max(1, top_k),
        seed=seed_value,
        fen=fen,
    )
    recent_games = run_self_play(config)
    save_self_play_results(recent_games)
    results = load_self_play_results()
    return templates.TemplateResponse("self_play.html", {
        "request": request,
        "results": results,
        "recent_games": recent_games,
        "config": config,
    })


@app.post("/self-play/start")
def self_play_start(
    games: int = Form(1),
    max_plies: int = Form(55),
    top_k: int = Form(3),
    seed: str | None = Form(None),
    fen: str | None = Form(None),
):
    fen = fen.strip() if fen and fen.strip() else None
    try:
        seed_value = int(seed) if seed and seed.strip() else None
    except ValueError:
        seed_value = None
    config = SelfPlayConfig(
        games=max(1, games),
        max_plies=max(1, max_plies),
        top_k=max(1, top_k),
        seed=seed_value,
        fen=fen,
    )
    return JSONResponse(start_self_play_job(config))


@app.get("/self-play/status/{job_id}")
def self_play_status(job_id: str):
    headers = {"Cache-Control": "no-store"}
    prune_old_jobs()
    job = load_self_play_job(job_id)
    if job is None:
        return JSONResponse(
            {"job_id": job_id, "state": "missing"}, status_code=404, headers=headers
        )
    return JSONResponse(job, headers=headers)


@app.get("/games", response_class=HTMLResponse)
def list_games(request: Request, username: str, force_refresh: str | None = None):
    username = username.strip()
    pgn_text = _ensure_pgn(username, bool(force_refresh))
    games = load_game_summaries(pgn_text, username=username)
    return templates.TemplateResponse("games.html", {
        "request": request,
        "username": username,
        "games": games,
    })


@app.get("/games/{username}/{index}", response_class=HTMLResponse)
def view_game(request: Request, username: str, index: int):
    username = username.strip()
    pgn_text = _ensure_pgn(username)
    detail = load_game_detail(pgn_text, index)
    summaries = load_game_summaries(pgn_text, username=username)
    total = len(summaries)
    positions_data = [asdict(pos) for pos in detail.positions] if detail else []
    return templates.TemplateResponse("game.html", {
        "request": request,
        "username": username,
        "index": index,
        "total": total,
        "detail": detail,
        "positions_data": positions_data,
        "score_weights": {
            "legal_moves": LEGAL_MOVES_WEIGHT,
            "material": MATERIAL_SCORE_WEIGHT,
            "forward": FORWARD_SCORE_WEIGHT,
        },
    })


@app.get("/self-play/view/{run_id}/{index}", response_class=HTMLResponse)
def view_self_play_game(request: Request, run_id: str, index: int):
    row = load_self_play_result(run_id, index)
    if row is None:
        return templates.TemplateResponse("game.html", {
            "request": request,
            "username": "Self-play",
            "index": index,
            "total": 0,
            "detail": None,
            "positions_data": [],
            "back_url": "/self-play",
            "back_label": "Back to self-play",
        })

    detail = load_game_detail(row["pgn"], 1)
    positions_data = [asdict(pos) for pos in detail.positions] if detail else []
    game_summary = {
        "status": row.get("outcome") or "",
        "winner": row.get("winner") or "",
        "loser": row.get("loser") or "",
        "white_weights": row.get("white_weights"),
        "black_weights": row.get("black_weights"),
    }
    return templates.TemplateResponse("game.html", {
        "request": request,
        "username": f"Self-play {run_id}",
        "index": index,
        "total": 1,
        "detail": detail,
        "positions_data": positions_data,
        "back_url": "/self-play",
        "back_label": "Back to self-play",
        "game_summary": game_summary,
        "score_weights": {
            "legal_moves": LEGAL_MOVES_WEIGHT,
            "material": MATERIAL_SCORE_WEIGHT,
            "forward": FORWARD_SCORE_WEIGHT,
        },
    })


@app.post("/analyse", response_class=HTMLResponse)
def analyse(
    request: Request,
    username: str = Form(...),
    timezone: str = Form("America/New_York"),
    breakpoint_iso: str | None = Form(None),
    force_refresh: str | None = Form(None),
):
    username = username.strip()
    pgn_path, csv_path = _cached_paths(username)

    if force_refresh or not pgn_path.exists():
        client = ChessComClient()
        pgn_text = client.fetch_all_pgn(username)
        pgn_path.write_text(pgn_text, encoding="utf-8")
    else:
        pgn_text = pgn_path.read_text(encoding="utf-8")

    if force_refresh or not csv_path.exists():
        df = parse_pgn_to_dataframe(pgn_text, username=username, tz_name=timezone)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    _maybe_save_to_neo4j(username, df)

    summary = summarize(df)
    charts = _make_charts(df)
    prepost = prepost_breakpoint(df, breakpoint_iso, label="breakpoint")
    prepost_rows = prepost.to_dict(orient="records") if not prepost.empty else []

    return templates.TemplateResponse("user.html", {
        "request": request,
        "username": username,
        "timezone": timezone,
        "breakpoint_iso": breakpoint_iso,
        "summary": summary,
        "charts": charts,
        "prepost_rows": prepost_rows,
    })
