from __future__ import annotations

from datetime import datetime
import pandas as pd


def performance_rating(avg_opponent: float, score_pct: float) -> float:
    """Approximation: 50% = avg opponent; every 1 percentage point = 8 Elo."""
    return float(avg_opponent + (score_pct - 0.5) * 800)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"games": 0}
    score_pct = float(df["score"].mean())
    avg_opp = float(df["opponent_rating"].mean())
    return {
        "games": int(len(df)),
        "score_pct": score_pct,
        "avg_opponent": avg_opp,
        "performance_rating": performance_rating(avg_opp, score_pct),
        "wins": int((df["score"] == 1.0).sum()),
        "draws": int((df["score"] == 0.5).sum()),
        "losses": int((df["score"] == 0.0).sum()),
        "first_game": str(df["local_date"].iloc[0]),
        "last_game": str(df["local_date"].iloc[-1]),
    }


def monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("month").agg(
        games=("score", "size"),
        score_pct=("score", "mean"),
        avg_opponent=("opponent_rating", "mean"),
        avg_user_rating=("user_rating", "mean"),
    ).reset_index()
    out["performance_rating"] = out.apply(lambda r: performance_rating(r["avg_opponent"], r["score_pct"]), axis=1)
    return out


def rolling_performance(df: pd.DataFrame, window: int = 100, step: int = 10) -> pd.DataFrame:
    if len(df) < window:
        return pd.DataFrame()
    rows = []
    for start in range(0, len(df) - window + 1, step):
        sub = df.iloc[start:start + window]
        score_pct = float(sub["score"].mean())
        avg_opp = float(sub["opponent_rating"].mean())
        rows.append({
            "start_game": int(sub["game_index"].iloc[0]),
            "end_game": int(sub["game_index"].iloc[-1]),
            "mid_game": int(round((sub["game_index"].iloc[0] + sub["game_index"].iloc[-1]) / 2)),
            "start_date": str(sub["local_date"].iloc[0]),
            "end_date": str(sub["local_date"].iloc[-1]),
            "games": int(len(sub)),
            "score_pct": score_pct,
            "avg_opponent": avg_opp,
            "avg_user_rating": float(sub["user_rating"].mean()),
            "performance_rating": performance_rating(avg_opp, score_pct),
        })
    return pd.DataFrame(rows)


def hourly_performance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("local_hour").agg(
        games=("score", "size"),
        score_pct=("score", "mean"),
        avg_opponent=("opponent_rating", "mean"),
    ).reset_index()
    out["performance_rating"] = out.apply(lambda r: performance_rating(r["avg_opponent"], r["score_pct"]), axis=1)
    return out


def day_performance(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out = df.groupby("local_day").agg(
        games=("score", "size"),
        score_pct=("score", "mean"),
        avg_opponent=("opponent_rating", "mean"),
    ).reindex(order).reset_index()
    out["performance_rating"] = out.apply(lambda r: performance_rating(r["avg_opponent"], r["score_pct"]) if pd.notna(r["games"]) else None, axis=1)
    return out


def time_day_matrix(df: pd.DataFrame) -> pd.DataFrame:
    def bucket(hour: int) -> str | None:
        if 18 <= hour < 20:
            return "6–8 PM"
        if 20 <= hour < 22:
            return "8–10 PM"
        if 22 <= hour < 24:
            return "10 PM–Midnight"
        if 0 <= hour < 2:
            return "After Midnight"
        return None

    tmp = df.copy()
    tmp["time_bucket"] = tmp["local_hour"].apply(bucket)
    tmp = tmp.dropna(subset=["time_bucket"])

    def day_group(day: str) -> str:
        if day in ["Monday", "Tuesday", "Wednesday", "Thursday"]:
            return "Mon–Thu"
        if day in ["Friday", "Saturday"]:
            return "Fri–Sat"
        return "Sunday"

    tmp["day_group"] = tmp["local_day"].apply(day_group)
    out = tmp.groupby(["time_bucket", "day_group"]).agg(
        games=("score", "size"),
        score_pct=("score", "mean"),
        avg_opponent=("opponent_rating", "mean"),
    ).reset_index()
    out["performance_rating"] = out.apply(lambda r: performance_rating(r["avg_opponent"], r["score_pct"]), axis=1)
    return out


def prepost_breakpoint(df: pd.DataFrame, breakpoint_iso: str | None, label: str = "breakpoint") -> pd.DataFrame:
    if not breakpoint_iso:
        return pd.DataFrame()
    try:
        breakpoint = datetime.fromisoformat(breakpoint_iso)
    except (ValueError, TypeError):
        return pd.DataFrame()

    tmp = df.copy()
    tmp["local_dt_obj"] = pd.to_datetime(tmp["local_datetime"])
    tmp["period"] = tmp["local_dt_obj"].apply(lambda x: f"Before {label}" if x < breakpoint else f"After {label}")
    out = tmp.groupby("period").agg(
        games=("score", "size"),
        score_pct=("score", "mean"),
        avg_opponent=("opponent_rating", "mean"),
        avg_user_rating=("user_rating", "mean"),
    ).reset_index()
    out["performance_rating"] = out.apply(lambda r: performance_rating(r["avg_opponent"], r["score_pct"]), axis=1)
    return out
