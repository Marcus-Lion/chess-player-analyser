from __future__ import annotations

import json
import time
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
    _MOVE_MAX_ATTEMPTS = 3

    def __init__(self, token: str, user_agent: str = "MarcusLionChessAnalyser/0.1") -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": user_agent,
                "Accept": "application/x-ndjson, application/json",
            }
        )

    @staticmethod
    def _is_retryable_response(response: requests.Response) -> bool:
        return response.status_code == 429 or 500 <= response.status_code < 600

    def challenge_ai(
        self,
        *,
        level: int,
        color: str,
        clock_limit: int,
        clock_increment: int,
        variant: str | None = None,
        fen: str | None = None,
    ) -> LichessChallenge:
        payload: dict[str, Any] = {
            "level": int(level),
            "color": color,
            "clock.limit": int(clock_limit),
            "clock.increment": int(clock_increment),
        }
        if variant:
            payload["variant"] = variant
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

    @staticmethod
    def _extract_game_id(payload: dict[str, Any]) -> str | None:
        game = payload.get("game")
        if isinstance(game, dict):
            for key in ("id", "gameId"):
                value = game.get(key)
                if value:
                    return str(value)
        for key in ("gameId", "id"):
            value = payload.get(key)
            if value:
                return str(value)
        return None

    def start_ai_game(
        self,
        *,
        level: int,
        color: str,
        clock_limit: int,
        clock_increment: int,
        variant: str | None = None,
        fen: str | None = None,
    ) -> str:
        if fen and not variant:
            variant = "fromPosition"
        challenge = self.challenge_ai(
            level=level,
            color=color,
            clock_limit=clock_limit,
            clock_increment=clock_increment,
            variant=variant,
            fen=fen,
        )
        challenge_id = challenge.id or None
        for event in self.stream_events():
            if challenge_id and event.get("type") == "challenge":
                challenge_event = event.get("challenge")
                if isinstance(challenge_event, dict) and str(challenge_event.get("id", "")) == challenge_id:
                    continue
            if event.get("type") == "gameStart":
                game_id = self._extract_game_id(event)
                if game_id:
                    return game_id
        raise RuntimeError("Lichess did not emit a gameStart event for the AI challenge.")

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
        url = f"{LICHESS_API_BASE}/api/board/game/{game_id}/move/{move}"
        last_error: Exception | None = None
        for attempt in range(1, self._MOVE_MAX_ATTEMPTS + 1):
            try:
                response = self.session.post(url, timeout=30)
                if self._is_retryable_response(response):
                    body = response.text.strip()
                    raise requests.HTTPError(
                        f"Lichess move attempt {attempt} returned {response.status_code}"
                        + (f": {body}" if body else ""),
                        response=response,
                    )
                response.raise_for_status()
                return response.json() if response.content else {}
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_error = exc
                if attempt >= self._MOVE_MAX_ATTEMPTS:
                    break
                delay_seconds = 0.5 * attempt
                print(
                    f"Lichess move {move} for game {game_id} failed on attempt {attempt}/"
                    f"{self._MOVE_MAX_ATTEMPTS}: {exc}. Retrying in {delay_seconds:.1f}s",
                    flush=True,
                )
                time.sleep(delay_seconds)
        assert last_error is not None
        raise last_error
