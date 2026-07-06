import time
from dataclasses import dataclass
from typing import List
import requests


@dataclass
class ChessComArchive:
    username: str
    archives: List[str]


class ChessComClient:
    """Small Chess.com public API client."""

    def __init__(self, user_agent: str = "MarcusLionChessAnalyser/0.1"):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def get_archives(self, username: str) -> ChessComArchive:
        url = f"https://api.chess.com/pub/player/{username}/games/archives"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return ChessComArchive(username=username, archives=payload.get("archives", []))

    def fetch_all_pgn(self, username: str, sleep_seconds: float = 0.15) -> str:
        archive = self.get_archives(username)
        chunks: list[str] = []

        for archive_url in archive.archives:
            pgn_url = archive_url + "/pgn"
            response = self.session.get(pgn_url, timeout=60)
            response.raise_for_status()
            text = response.text.strip()
            if text:
                chunks.append(text)
            time.sleep(sleep_seconds)

        return "\n\n".join(chunks)
