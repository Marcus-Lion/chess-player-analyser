from __future__ import annotations

import pandas as pd

OUTCOME_ORDER = ["White wins", "Black wins", "Draw"]
WEIGHT_DIMENSIONS = [
    "legal_moves_weight",
    "material_score_weight",
    "forward_score_weight",
    "center_control_weight",
]


def to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["played_at"] = pd.to_datetime(df["played_at"], errors="coerce", utc=True)
    df["game_seq"] = range(1, len(df) + 1)
    df["white_won"] = df["result"] == "1-0"
    df["black_won"] = df["result"] == "0-1"
    df["is_draw"] = df["result"] == "1/2-1/2"

    for side in ("white", "black"):
        weights = df[f"{side}_weights"].apply(lambda w: w if isinstance(w, dict) else {})
        for dim in WEIGHT_DIMENSIONS:
            df[f"{side}_{dim}"] = weights.apply(lambda w, dim=dim: w.get(dim))

    for dim in WEIGHT_DIMENSIONS:
        df[f"weight_diff_{dim}"] = df[f"white_{dim}"] - df[f"black_{dim}"]

    return df


def summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"games": 0}

    games = len(df)
    decisive = int((~df["is_draw"]).sum())
    top_termination = df["termination"].value_counts().idxmax()
    return {
        "games": games,
        "decisive_pct": decisive / games,
        "draw_pct": float(df["is_draw"].mean()),
        "white_win_pct": float(df["white_won"].mean()),
        "avg_plies": float(df["plies"].mean()),
        "top_termination": top_termination,
    }


def outcome_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["outcome"].value_counts()
        .reindex(OUTCOME_ORDER)
        .fillna(0)
        .astype(int)
        .rename_axis("outcome")
        .reset_index(name="games")
    )


def termination_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["termination"].value_counts()
        .rename_axis("termination")
        .reset_index(name="games")
        .sort_values("games", ascending=False)
    )


def plies_by_termination(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("termination")["plies"]
        .mean()
        .rename("avg_plies")
        .reset_index()
        .sort_values("avg_plies", ascending=False)
    )


def rolling_outcome_rates(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    window = min(window, len(df)) or 1
    rolling = pd.DataFrame({
        "game_seq": df["game_seq"],
        "White wins": df["white_won"].rolling(window, min_periods=1).mean(),
        "Black wins": df["black_won"].rolling(window, min_periods=1).mean(),
        "Draw": df["is_draw"].rolling(window, min_periods=1).mean(),
    })
    return rolling.melt(id_vars="game_seq", var_name="outcome", value_name="rate")


def win_rate_by_weight_advantage(df: pd.DataFrame, dim: str, bins: int = 8) -> pd.DataFrame:
    diff_col = f"weight_diff_{dim}"
    valid = df.dropna(subset=[diff_col])
    if valid.empty or valid[diff_col].nunique() < 2:
        return pd.DataFrame(columns=["weight_dim", "bucket", "white_win_rate", "games"])

    bucket = pd.qcut(valid[diff_col], q=min(bins, valid[diff_col].nunique()), duplicates="drop")
    grouped = valid.groupby(bucket, observed=True)["white_won"].agg(white_win_rate="mean", games="size").reset_index()
    grouped["bucket"] = grouped[diff_col].apply(lambda interval: f"{interval.left:.2f} to {interval.right:.2f}")
    grouped["weight_dim"] = dim
    return grouped[["weight_dim", "bucket", "white_win_rate", "games"]]


def win_rate_by_weight_advantage_all(df: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    frames = [win_rate_by_weight_advantage(df, dim, bins=bins) for dim in WEIGHT_DIMENSIONS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["weight_dim", "bucket", "white_win_rate", "games"])
    return pd.concat(frames, ignore_index=True)


def final_score_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    return df[["outcome", "final_score"]].dropna()


def weight_diff_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-game (weight advantage, final score) points, long-form across all weight dims."""
    frames = []
    for dim in WEIGHT_DIMENSIONS:
        diff_col = f"weight_diff_{dim}"
        if diff_col not in df.columns:
            continue
        frame = df[[diff_col, "final_score", "outcome"]].dropna(subset=[diff_col, "final_score"]).copy()
        frame = frame.rename(columns={diff_col: "weight_diff"})
        frame["weight_dim"] = dim
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["weight_diff", "final_score", "outcome", "weight_dim"])
    return pd.concat(frames, ignore_index=True)
