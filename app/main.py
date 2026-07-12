from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import plotly.express as px
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.chesscom import ChessComClient
from app.games import load_game_summaries, load_game_detail
from app.parser import parse_pgn_to_dataframe
from app.self_play import (
    SelfPlayConfig,
    load_self_play_result,
    load_self_play_results,
    run_self_play,
    save_self_play_results,
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
    max_plies: int = Form(200),
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
    return templates.TemplateResponse("game.html", {
        "request": request,
        "username": f"Self-play {run_id}",
        "index": index,
        "total": 1,
        "detail": detail,
        "positions_data": positions_data,
        "back_url": "/self-play",
        "back_label": "Back to self-play",
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
