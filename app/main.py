from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.chesscom import ChessComClient
from app.parser import parse_pgn_to_dataframe
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


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})


def _make_charts(df: pd.DataFrame) -> dict[str, str]:
    charts = {}

    monthly = monthly_performance(df)
    if not monthly.empty:
        fig = px.line(monthly, x="month", y=["performance_rating", "avg_opponent", "avg_user_rating"], markers=True, title="Monthly performance rating")
        charts["monthly"] = _fig_html(fig)

    rolling = rolling_performance(df)
    if not rolling.empty:
        fig = px.line(rolling, x="mid_game", y=["performance_rating", "avg_opponent", "avg_user_rating"], title="Rolling 100-game performance rating")
        charts["rolling"] = _fig_html(fig)

    hourly = hourly_performance(df)
    if not hourly.empty:
        fig = px.line(hourly, x="local_hour", y="performance_rating", markers=True, title="Performance by local start hour")
        charts["hourly"] = _fig_html(fig)

    day = day_performance(df)
    if not day.empty:
        fig = px.bar(day, x="local_day", y="performance_rating", title="Performance by day of week")
        charts["day"] = _fig_html(fig)

    matrix = time_day_matrix(df)
    if not matrix.empty:
        pivot = matrix.pivot(index="time_bucket", columns="day_group", values="performance_rating")
        pivot = pivot.reindex(["6–8 PM", "8–10 PM", "10 PM–Midnight", "After Midnight"])
        fig = px.imshow(pivot, text_auto=".0f", aspect="auto", title="Time × day performance heatmap")
        charts["time_day"] = _fig_html(fig)

    return charts


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
