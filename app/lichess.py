from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import requests


LICHESS_API_BASE = "https://lichess.org"


@dataclass(frozen=True)
class LichessChallenge:
    raw: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.raw.get("id", ""))


class LichessClient:
    def __init__(self, token: str, user_agent: str = "MarcusLionChessAnalyser/0.1") -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": user_agent,
                "Accept": "application/x-ndjson, application/json",
            }
        )

    def challenge_ai(
        self,
        *,
        level: int,
        color: str,
        clock_limit: int,
        clock_increment: int,
        fen: str | None = None,
    ) -> LichessChallenge:
        payload: dict[str, Any] = {
            "level": int(level),
            "color": color,
            "clock.limit": int(clock_limit),
            "clock.increment": int(clock_increment),
        }
        if fen:
            payload["fen"] = fen

        response = self.session.post(
            f"{LICHESS_API_BASE}/api/challenge/ai",
            data=payload,
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json() if response.content else {}
        if not isinstance(raw, dict):
            raw = {}
        return LichessChallenge(raw=raw)

    def stream_events(self) -> Iterator[dict[str, Any]]:
        response = self.session.get(
            f"{LICHESS_API_BASE}/api/stream/event",
            stream=True,
            timeout=(10, None),
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)

    def stream_game(self, game_id: str) -> Iterator[dict[str, Any]]:
        response = self.session.get(
            f"{LICHESS_API_BASE}/api/board/game/stream/{game_id}",
            stream=True,
            timeout=(10, None),
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            yield json.loads(line)

    def make_move(self, game_id: str, move: str) -> dict[str, Any]:
        response = self.session.post(
            f"{LICHESS_API_BASE}/api/board/game/{game_id}/move/{move}",
            timeout=30,
        )
        response.raise_for_status()
        return response.json() if response.content else {}
