from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import chess
import plotly.express as px
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

load_dotenv()

from app.chesscom import ChessComClient
from app.games import (
    _result_summary,
    choose_engine_move,
    load_game_summaries,
    load_game_detail,
    render_board_svgs,
)
from app.parser import parse_pgn_to_dataframe
from app.self_play import (
    SelfPlayConfig,
    get_job_hub,
    load_self_play_job,
    load_self_play_result,
    load_self_play_results,
    prune_old_jobs,
    run_self_play,
    save_self_play_results,
    start_self_play_job,
)
from app.self_play_metrics import (
    OUTCOME_ORDER,
    WEIGHT_DIMENSIONS,
    final_score_by_outcome,
    outcome_counts,
    plies_by_termination,
    rolling_outcome_rates,
    summary as self_play_summary,
    termination_counts,
    to_dataframe as self_play_to_dataframe,
    weight_diff_scores,
    win_rate_by_weight_advantage_all,
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

app = FastAPI(title="Marcus Lion Chess Player Analyser")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


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


def _make_self_play_charts(df: pd.DataFrame) -> dict[str, str]:
    charts = {}

    outcomes = outcome_counts(df)
    if not outcomes.empty:
        fig = px.bar(outcomes, x="outcome", y="games", title="Outcome mix",
                     category_orders={"outcome": OUTCOME_ORDER})
        charts["outcomes"] = _fig_html(fig)

    terminations = termination_counts(df)
    if not terminations.empty:
        fig = px.bar(terminations, x="games", y="termination", orientation="h",
                     title="Termination reasons",
                     category_orders={"termination": terminations["termination"].tolist()})
        fig.update_yaxes(autorange="reversed")
        charts["terminations"] = _fig_html(fig)

    if "plies" in df.columns and not df["plies"].dropna().empty:
        fig = px.histogram(df, x="plies", nbins=40, title="Game length distribution (turns)")
        charts["plies_hist"] = _fig_html(fig)

    avg_plies = plies_by_termination(df)
    if not avg_plies.empty:
        fig = px.bar(avg_plies, x="termination", y="avg_plies", title="Average turns by termination",
                     category_orders={"termination": avg_plies["termination"].tolist()})
        charts["avg_plies_by_termination"] = _fig_html(fig)

    rolling = rolling_outcome_rates(df)
    if not rolling.empty:
        fig = px.line(rolling, x="game_seq", y="rate", color="outcome",
                      title="Rolling outcome rate over games played",
                      category_orders={"outcome": OUTCOME_ORDER})
        charts["rolling_outcomes"] = _fig_html(fig)

    weight_advantage = win_rate_by_weight_advantage_all(df)
    if not weight_advantage.empty:
        fig = px.bar(weight_advantage, x="bucket", y="white_win_rate", facet_col="weight_dim",
                     hover_data=["games"],
                     title="White win rate by (white − black) weight advantage")
        fig.update_xaxes(tickangle=45, matches=None)
        charts["weight_advantage"] = _fig_html(fig)

    score_by_outcome = final_score_by_outcome(df)
    if not score_by_outcome.empty:
        fig = px.box(score_by_outcome, x="outcome", y="final_score", title="Final score spread by outcome",
                     category_orders={"outcome": OUTCOME_ORDER})
        charts["score_by_outcome"] = _fig_html(fig)

    weight_scores = weight_diff_scores(df)
    for dim in WEIGHT_DIMENSIONS:
        points = weight_scores[weight_scores["weight_dim"] == dim]
        if points.empty:
            continue
        label = dim.replace("_", " ")
        fig = px.scatter(points, x="weight_diff", y="final_score", color="outcome",
                          category_orders={"outcome": OUTCOME_ORDER}, opacity=0.6,
                          title=f"Final score vs (white − black) {label} advantage")
        charts[f"weight_scatter_{dim}"] = _fig_html(fig)

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


def _legal_move_options(board: chess.Board) -> list[str]:
    return [board.san(move) for move in board.legal_moves]


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

    svg, svg_moves = render_board_svgs(board, lastmove=last_move)

    return {
        "board_svg": svg,
        "board_svg_moves": svg_moves,
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
    games: int = Form(3),
    max_turns: int = Form(100),
    top_k: int = Form(3),
    workers: str | None = Form(None),
    seed: str | None = Form(None),
    fen: str | None = Form(None),
    white_legal_moves_weight: str | None = Form(None),
    white_material_score_weight: str | None = Form(None),
    white_forward_score_weight: str | None = Form(None),
    white_center_control_weight: str | None = Form(None),
    black_legal_moves_weight: str | None = Form(None),
    black_material_score_weight: str | None = Form(None),
    black_forward_score_weight: str | None = Form(None),
    black_center_control_weight: str | None = Form(None),
):
    fen = fen.strip() if fen and fen.strip() else None
    try:
        workers_value = int(workers) if workers and workers.strip() else None
    except ValueError:
        workers_value = None
    try:
        seed_value = int(seed) if seed and seed.strip() else None
    except ValueError:
        seed_value = None
    config = SelfPlayConfig(
        games=max(1, games),
        max_turns=max(2, max_turns),
        top_k=max(1, top_k),
        workers=(max(1, workers_value) if workers_value else None),
        seed=seed_value,
        fen=fen,
        white_legal_moves_weight=_parse_optional_float(white_legal_moves_weight),
        white_material_score_weight=_parse_optional_float(white_material_score_weight),
        white_forward_score_weight=_parse_optional_float(white_forward_score_weight),
        white_center_control_weight=_parse_optional_float(white_center_control_weight),
        black_legal_moves_weight=_parse_optional_float(black_legal_moves_weight),
        black_material_score_weight=_parse_optional_float(black_material_score_weight),
        black_forward_score_weight=_parse_optional_float(black_forward_score_weight),
        black_center_control_weight=_parse_optional_float(black_center_control_weight),
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
    games: int = Form(3),
    max_turns: int = Form(100),
    top_k: int = Form(3),
    workers: str | None = Form(None),
    seed: str | None = Form(None),
    fen: str | None = Form(None),
    white_legal_moves_weight: str | None = Form(None),
    white_material_score_weight: str | None = Form(None),
    white_forward_score_weight: str | None = Form(None),
    white_center_control_weight: str | None = Form(None),
    black_legal_moves_weight: str | None = Form(None),
    black_material_score_weight: str | None = Form(None),
    black_forward_score_weight: str | None = Form(None),
    black_center_control_weight: str | None = Form(None),
):
    fen = fen.strip() if fen and fen.strip() else None
    try:
        workers_value = int(workers) if workers and workers.strip() else None
    except ValueError:
        workers_value = None
    try:
        seed_value = int(seed) if seed and seed.strip() else None
    except ValueError:
        seed_value = None
    config = SelfPlayConfig(
        games=max(1, games),
        max_turns=max(2, max_turns),
        top_k=max(1, top_k),
        workers=(max(1, workers_value) if workers_value else None),
        seed=seed_value,
        fen=fen,
        white_legal_moves_weight=_parse_optional_float(white_legal_moves_weight),
        white_material_score_weight=_parse_optional_float(white_material_score_weight),
        white_forward_score_weight=_parse_optional_float(white_forward_score_weight),
        white_center_control_weight=_parse_optional_float(white_center_control_weight),
        black_legal_moves_weight=_parse_optional_float(black_legal_moves_weight),
        black_material_score_weight=_parse_optional_float(black_material_score_weight),
        black_forward_score_weight=_parse_optional_float(black_forward_score_weight),
        black_center_control_weight=_parse_optional_float(black_center_control_weight),
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


@app.websocket("/self-play/ws/{job_id}")
async def self_play_ws(websocket: WebSocket, job_id: str) -> None:
    """Push job status the instant a worker reports it, instead of the
    browser polling /self-play/status on a timer -- we already have a live
    socket connection carrying every update into the job hub."""
    await websocket.accept()
    hub = get_job_hub()
    loop = asyncio.get_event_loop()

    job, version = hub.get_job_version(job_id)
    if job is None:
        await websocket.send_json({"job_id": job_id, "state": "missing"})
        await websocket.close()
        return

    try:
        await websocket.send_json(job)
        while job.get("state") not in ("completed", "failed"):
            job, new_version = await loop.run_in_executor(
                None, hub.wait_for_update, job_id, version, 30.0
            )
            if job is None:
                await websocket.send_json({"job_id": job_id, "state": "missing"})
                break
            if new_version == version:
                continue  # wait_for_update timed out with no change; keep waiting
            version = new_version
            await websocket.send_json(job)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/self-play/analysis", response_class=HTMLResponse)
def self_play_analysis(request: Request):
    rows = load_self_play_results(limit=None)
    df = self_play_to_dataframe(rows)
    return templates.TemplateResponse("self_play_analysis.html", {
        "request": request,
        "summary": self_play_summary(df),
        "charts": _make_self_play_charts(df) if not df.empty else {},
    })


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
        "duration_seconds": row.get("duration_seconds"),
        "evaluations": row.get("evaluations"),
        "evaluations_per_move": row.get("evaluations_per_move"),
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
