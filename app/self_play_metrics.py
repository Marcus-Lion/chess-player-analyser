from __future__ import annotations

import os

import pandas as pd

OUTCOME_ORDER = ["White wins", "Black wins", "Draw"]
WEIGHT_DIMENSIONS = [
    "legal_moves_weight",
    "material_score_weight",
    "forward_score_weight",
    "center_control_weight",
]


def _elo_baseline() -> float:
    raw = (os.getenv("BASELINE_ELO") or os.getenv("ELO_BASELINE") or "1500").strip()
    try:
        return float(raw)
    except ValueError:
        return 1500.0


def _floor_elo(value: float, floor: float = 100.0) -> float:
    return max(floor, float(value))


def _elo_k(games_played: int) -> float:
    if games_played < 30:
        return 40.0
    if games_played < 100:
        return 24.0
    return 16.0


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _score_for_result(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "1/2-1/2":
        return 0.5
    if result == "0-1":
        return 0.0
    return 0.5


def _ordered_self_play_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sort_cols = [col for col in ("played_at", "game_seq", "run_id", "index") if col in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)


def _dynamic_elo_state(df: pd.DataFrame, baseline: float | None = None) -> tuple[list[float], list[float], list[float], dict[str, float], dict[str, int]]:
    baseline = _elo_baseline() if baseline is None else baseline
    rows = _ordered_self_play_rows(df)
    white_elos: list[float] = []
    black_elos: list[float] = []
    elo_gaps: list[float] = []
    ratings: dict[str, float] = {}
    games_played: dict[str, int] = {}

    for _, row in rows.iterrows():
        white_id = row.get("white_player_id")
        black_id = row.get("black_player_id")
        white_key = None if pd.isna(white_id) else str(white_id)
        black_key = None if pd.isna(black_id) else str(black_id)
        white_rating = ratings.get(white_key, baseline)
        black_rating = ratings.get(black_key, baseline)

        white_elos.append(_floor_elo(white_rating))
        black_elos.append(_floor_elo(black_rating))
        elo_gaps.append(white_rating - black_rating)

        if white_key is None or black_key is None:
            continue

        score_white = _score_for_result(str(row.get("result", "")))
        score_black = 1.0 - score_white

        white_games = games_played.get(white_key, 0)
        black_games = games_played.get(black_key, 0)
        white_k = _elo_k(white_games)
        black_k = _elo_k(black_games)

        expected_white = _elo_expected(white_rating, black_rating)
        expected_black = 1.0 - expected_white

        ratings[white_key] = _floor_elo(white_rating + white_k * (score_white - expected_white))
        ratings[black_key] = _floor_elo(black_rating + black_k * (score_black - expected_black))
        games_played[white_key] = white_games + 1
        games_played[black_key] = black_games + 1

    return white_elos, black_elos, elo_gaps, ratings, games_played


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

    if {"white_player_id", "black_player_id", "result"}.issubset(out.columns):
        white_elos, black_elos, elo_gaps, _, _ = _dynamic_elo_state(out)
        out["WhiteElo"] = white_elos
        out["BlackElo"] = black_elos
        out["EloGap"] = elo_gaps

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
        "white_player_id",
        "white_player_name",
        "white_player_description",
        "black_player_id",
        "black_player_name",
        "black_player_description",
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
    out = out[ordered]
    return out.where(pd.notna(out), None)


def estimate_side_elos(df: pd.DataFrame, baseline: float | None = None) -> dict[str, float]:
    baseline = _elo_baseline() if baseline is None else baseline
    if df.empty:
        return {
            "white_score_pct": 0.0,
            "elo_diff": 0.0,
            "white_elo": baseline,
            "black_elo": baseline,
        }

    table = export_dataframe(df)
    white_elo = float(table["WhiteElo"].mean()) if "WhiteElo" in table.columns and not table["WhiteElo"].empty else baseline
    black_elo = float(table["BlackElo"].mean()) if "BlackElo" in table.columns and not table["BlackElo"].empty else baseline
    elo_diff = white_elo - black_elo
    white_score_pct = float(df["white_won"].mean() + 0.5 * df["is_draw"].mean())
    return {
        "white_score_pct": white_score_pct,
        "elo_diff": elo_diff,
        "white_elo": _floor_elo(white_elo),
        "black_elo": _floor_elo(black_elo),
    }


def player_participations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "player_id",
            "player_name",
            "player_description",
            "color",
            "opponent_id",
            "opponent_name",
            "opponent_description",
            "score",
            "outcome",
        ])

    frames = []
    score_maps = {
        "white": {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5},
        "black": {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5},
    }

    for side in ("white", "black"):
        id_col = f"{side}_player_id"
        name_col = f"{side}_player_name"
        desc_col = f"{side}_player_description"
        opp_side = "black" if side == "white" else "white"
        opp_id_col = f"{opp_side}_player_id"
        opp_name_col = f"{opp_side}_player_name"
        opp_desc_col = f"{opp_side}_player_description"
        if id_col not in df.columns:
            continue

        frame = df[[
            "played_at",
            "run_id",
            "index",
            "result",
            "termination",
            "plies",
            "final_score",
            "outcome",
            "winner",
            "loser",
            "duration_seconds",
            "evaluations_per_move",
        ]].copy()
        frame["player_id"] = df[id_col]
        frame["player_name"] = df[name_col] if name_col in df.columns else df[id_col]
        frame["player_description"] = df[desc_col] if desc_col in df.columns else ""
        frame["opponent_id"] = df[opp_id_col] if opp_id_col in df.columns else ""
        frame["opponent_name"] = df[opp_name_col] if opp_name_col in df.columns else ""
        frame["opponent_description"] = df[opp_desc_col] if opp_desc_col in df.columns else ""
        for col in ("player_name", "player_description", "opponent_id", "opponent_name", "opponent_description"):
            frame[col] = frame[col].where(pd.notna(frame[col]), "")
        frame["color"] = side.title()
        frame["score"] = frame["result"].map(score_maps[side]).fillna(0.5)
        frame["outcome"] = frame["score"].map({1.0: "Win", 0.5: "Draw", 0.0: "Loss"})
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=[
            "player_id",
            "player_name",
            "player_description",
            "color",
            "opponent_id",
            "opponent_name",
            "opponent_description",
            "score",
            "outcome",
        ])
    return pd.concat(frames, ignore_index=True)


def player_overview(df: pd.DataFrame) -> pd.DataFrame:
    baseline = _elo_baseline()
    parts = player_participations(df)
    if parts.empty:
        return pd.DataFrame(columns=[
            "player_id",
            "player_name",
            "player_description",
            "games",
            "wins",
            "draws",
            "losses",
            "score_pct",
            "white_games",
            "black_games",
            "elo",
        ])

    _, _, _, ratings, _ = _dynamic_elo_state(df, baseline=baseline)
    grouped = parts.groupby(["player_id", "player_name", "player_description"], dropna=False)
    out = grouped.agg(
        games=("score", "size"),
        wins=("score", lambda s: int((s == 1.0).sum())),
        draws=("score", lambda s: int((s == 0.5).sum())),
        losses=("score", lambda s: int((s == 0.0).sum())),
        score_pct=("score", "mean"),
        white_games=("color", lambda s: int((s == "White").sum())),
        black_games=("color", lambda s: int((s == "Black").sum())),
    ).reset_index()
    out["elo"] = out["player_id"].map(lambda pid: _floor_elo(ratings.get(str(pid), baseline)))
    return out.sort_values(["score_pct", "games"], ascending=[False, False]).reset_index(drop=True)


def player_detail(df: pd.DataFrame, player_id: str) -> dict:
    baseline = _elo_baseline()
    parts = player_participations(df)
    if parts.empty:
        return {"overview": {}, "games": pd.DataFrame()}

    player = parts[parts["player_id"] == player_id].copy()
    if player.empty:
        return {"overview": {}, "games": pd.DataFrame()}

    _, _, _, ratings, _ = _dynamic_elo_state(df, baseline=baseline)
    player = player.sort_values("played_at")
    overview = {
        "player_id": player_id,
        "player_name": player["player_name"].iloc[0],
        "player_description": player["player_description"].iloc[0],
        "games": int(len(player)),
        "wins": int((player["score"] == 1.0).sum()),
        "draws": int((player["score"] == 0.5).sum()),
        "losses": int((player["score"] == 0.0).sum()),
        "score_pct": float(player["score"].mean()),
        "white_games": int((player["color"] == "White").sum()),
        "black_games": int((player["color"] == "Black").sum()),
        "elo": float(_floor_elo(ratings.get(str(player_id), baseline))),
    }

    recent = player.sort_values("played_at", ascending=False).copy()
    recent["opponent"] = recent["opponent_name"]
    recent["played_at"] = pd.to_datetime(recent["played_at"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"overview": overview, "games": recent}


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
