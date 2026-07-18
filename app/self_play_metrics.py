from __future__ import annotations

import math

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


def export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return one flattened row per game for CSV export and tabular display."""
    if df.empty:
        return df.copy()

    out = df.copy()

    if "played_at" in out.columns:
        played_at = pd.to_datetime(out["played_at"], errors="coerce", utc=True)
        out["played_at"] = played_at.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")

    for side, prefix in (("white", "WhiteWeights"), ("black", "BlackWeights")):
        weights_col = f"{side}_weights"
        if weights_col not in out.columns:
            continue
        weights = out[weights_col].apply(lambda w: w if isinstance(w, dict) else {})
        for dim in WEIGHT_DIMENSIONS:
            col = f"{prefix}_{dim}"
            out[col] = weights.apply(lambda w, dim=dim: w.get(dim))

    if {"white_won", "is_draw"}.issubset(out.columns):
        white_score_pct = (
            out["white_won"].astype(float)
            .add(out["is_draw"].astype(float) * 0.5)
            .expanding()
            .mean()
        )
        elo_gap = white_score_pct.apply(score_pct_to_elo)
        out["WhiteElo"] = 1500.0 + elo_gap / 2.0
        out["BlackElo"] = 1500.0 - elo_gap / 2.0
        out["EloGap"] = elo_gap

    drop_cols = [col for col in ("white_weights", "black_weights", "pgn", "played_at_display") if col in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    preferred = [
        "played_at",
        "run_id",
        "index",
        "seed",
        "top_k",
        "max_turns",
        "start_fen",
        "result",
        "termination",
        "plies",
        "final_fen",
        "final_score",
        "outcome",
        "winner",
        "loser",
        "duration_seconds",
        "evaluations",
        "evaluations_per_move",
        "WhiteElo",
        "BlackElo",
        "EloGap",
        "WhiteWeights_legal_moves_weight",
        "WhiteWeights_material_score_weight",
        "WhiteWeights_forward_score_weight",
        "WhiteWeights_center_control_weight",
        "BlackWeights_legal_moves_weight",
        "BlackWeights_material_score_weight",
        "BlackWeights_forward_score_weight",
        "BlackWeights_center_control_weight",
    ]
    ordered = [col for col in preferred if col in out.columns]
    ordered.extend(col for col in out.columns if col not in ordered)
    return out[ordered]


def score_pct_to_elo(score_pct: float) -> float:
    clipped = min(max(float(score_pct), 1e-9), 1.0 - 1e-9)
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def estimate_side_elos(df: pd.DataFrame, baseline: float = 1500.0) -> dict[str, float]:
    if df.empty:
        return {
            "white_score_pct": 0.0,
            "elo_diff": 0.0,
            "white_elo": baseline,
            "black_elo": baseline,
        }

    white_score_pct = float(df["white_won"].mean() + 0.5 * df["is_draw"].mean())
    elo_diff = score_pct_to_elo(white_score_pct)
    return {
        "white_score_pct": white_score_pct,
        "elo_diff": elo_diff,
        "white_elo": baseline + elo_diff / 2.0,
        "black_elo": baseline - elo_diff / 2.0,
    }


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


def absolute_weight_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-game absolute white-vs-black weight pairs, long-form across all weight dims."""
    frames = []
    if "result" in df.columns:
        winner = df["result"].map({"1-0": "White", "0-1": "Black", "1/2-1/2": "Draw"}).fillna("Draw")
    else:
        winner = pd.Series(index=df.index, dtype=object).fillna("Draw")
    for dim in WEIGHT_DIMENSIONS:
        white_col = f"white_{dim}"
        black_col = f"black_{dim}"
        if white_col not in df.columns or black_col not in df.columns:
            continue
        frame = df[[white_col, black_col]].copy()
        frame = frame.rename(columns={white_col: "white_weight", black_col: "black_weight"})
        frame["winner"] = winner
        frame["weight_dim"] = dim
        frame = frame.dropna(subset=["white_weight", "black_weight"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["white_weight", "black_weight", "winner", "weight_dim"])
    return pd.concat(frames, ignore_index=True)
