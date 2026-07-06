from __future__ import annotations

from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo
import chess.pgn
import pandas as pd


def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except Exception:
        return default


def _clock_annotations(game: chess.pgn.Game) -> int:
    count = 0
    node = game
    while node.variations:
        node = node.variation(0)
        if "%clk" in (node.comment or ""):
            count += 1
    return count


def parse_pgn_to_dataframe(pgn_text: str, username: str, tz_name: str = "America/New_York") -> pd.DataFrame:
    username_l = username.lower()
    pgn = StringIO(pgn_text)
    rows: list[dict] = []
    tz = ZoneInfo(tz_name)

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        h = game.headers
        white = h.get("White", "")
        black = h.get("Black", "")

        if white.lower() == username_l:
            color = "White"
            opponent = black
            opponent_rating = _safe_int(h.get("BlackElo"))
            user_rating = _safe_int(h.get("WhiteElo"))
        elif black.lower() == username_l:
            color = "Black"
            opponent = white
            opponent_rating = _safe_int(h.get("WhiteElo"))
            user_rating = _safe_int(h.get("BlackElo"))
        else:
            continue

        result = h.get("Result", "*")
        if result == "1/2-1/2":
            score = 0.5
        elif result == "1-0":
            score = 1.0 if color == "White" else 0.0
        elif result == "0-1":
            score = 1.0 if color == "Black" else 0.0
        else:
            continue

        try:
            utc_dt = datetime.strptime(
                h.get("UTCDate", "1970.01.01") + " " + h.get("UTCTime", "00:00:00"),
                "%Y.%m.%d %H:%M:%S",
            ).replace(tzinfo=ZoneInfo("UTC"))
        except Exception:
            continue

        local_dt = utc_dt.astimezone(tz)
        rows.append({
            "utc_datetime": utc_dt.isoformat(),
            "local_datetime": local_dt.isoformat(),
            "local_date": local_dt.date().isoformat(),
            "local_hour": local_dt.hour,
            "local_day": local_dt.strftime("%A"),
            "month": local_dt.strftime("%Y-%m"),
            "username": username,
            "color": color,
            "score": score,
            "opponent": opponent,
            "opponent_rating": opponent_rating,
            "user_rating": user_rating,
            "result": result,
            "termination": h.get("Termination", ""),
            "time_control": h.get("TimeControl", ""),
            "moves": len(list(game.mainline_moves())),
            "clock_count": _clock_annotations(game),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("utc_datetime").reset_index(drop=True)
        df["game_index"] = df.index + 1
    return df
