from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

try:
    from neo4j import GraphDatabase, Driver
except Exception:  # pragma: no cover - neo4j is an optional dependency
    GraphDatabase = None  # type: ignore[assignment]
    Driver = None  # type: ignore[assignment]


class Neo4jStore:
    """Optional Neo4j sink for parsed games.

    Builds a small graph model:

        (:Player {username})-[:PLAYED]->(:Game {game_id, ...})
        (:Game)-[:AGAINST]->(:Opponent {name})

    Configuration is read from environment variables so the analytics engine
    stays unchanged when Neo4j is not configured:

        NEO4J_URI       (default: bolt://localhost:7687)
        NEO4J_USER      (default: neo4j)
        NEO4J_PASSWORD  (default: neo4j)
        NEO4J_DATABASE  (default: neo4j)
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        if GraphDatabase is None:
            raise RuntimeError(
                "The 'neo4j' package is not installed. Run `uv sync` to install it."
            )
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver: Driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT player_username IF NOT EXISTS "
            "FOR (p:Player) REQUIRE p.username IS UNIQUE",
            "CREATE CONSTRAINT opponent_name IF NOT EXISTS "
            "FOR (o:Opponent) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT game_id IF NOT EXISTS "
            "FOR (g:Game) REQUIRE g.game_id IS UNIQUE",
        ]
        with self._driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    def save_games(self, username: str, df: pd.DataFrame) -> int:
        """Upsert every parsed game for ``username`` into the graph.

        Returns the number of games written.
        """
        if df is None or df.empty:
            return 0

        rows = df.to_dict(orient="records")
        for row in rows:
            row["game_id"] = f"{username.lower()}:{row.get('utc_datetime')}"

        query = """
        UNWIND $rows AS row
        MERGE (p:Player {username: $username})
        MERGE (o:Opponent {name: row.opponent})
        MERGE (g:Game {game_id: row.game_id})
        SET g.utc_datetime = row.utc_datetime,
            g.local_datetime = row.local_datetime,
            g.local_date = row.local_date,
            g.local_hour = row.local_hour,
            g.local_day = row.local_day,
            g.month = row.month,
            g.color = row.color,
            g.score = row.score,
            g.opponent_rating = row.opponent_rating,
            g.user_rating = row.user_rating,
            g.result = row.result,
            g.termination = row.termination,
            g.time_control = row.time_control,
            g.moves = row.moves,
            g.clock_count = row.clock_count
        MERGE (p)-[:PLAYED]->(g)
        MERGE (g)-[:AGAINST]->(o)
        """

        self.ensure_constraints()
        with self._driver.session(database=self.database) as session:
            session.run(query, rows=rows, username=username)
        return len(rows)

    def ensure_self_play_constraints(self) -> None:
        statement = (
            "CREATE CONSTRAINT self_play_game_key IF NOT EXISTS "
            "FOR (g:SelfPlayGame) REQUIRE g.game_key IS UNIQUE"
        )
        with self._driver.session(database=self.database) as session:
            session.run(statement)

    @staticmethod
    def _encode_self_play_row(game: dict) -> dict:
        row = dict(game)
        row["game_key"] = f"{row.get('run_id')}:{row.get('index')}"
        row["white_weights_json"] = json.dumps(row.pop("white_weights", None))
        row["black_weights_json"] = json.dumps(row.pop("black_weights", None))
        return row

    @staticmethod
    def _decode_self_play_row(props: dict) -> dict:
        row = dict(props)
        row.pop("game_key", None)
        row["white_weights"] = json.loads(row.pop("white_weights_json", "null") or "null")
        row["black_weights"] = json.loads(row.pop("black_weights_json", "null") or "null")
        return row

    def save_self_play_games(self, games: list[dict]) -> int:
        """Upsert self-play game results, keyed on ``run_id``+``index``."""
        if not games:
            return 0

        rows = [self._encode_self_play_row(game) for game in games]

        query = """
        UNWIND $rows AS row
        MERGE (g:SelfPlayGame {game_key: row.game_key})
        SET g += row
        """

        self.ensure_self_play_constraints()
        with self._driver.session(database=self.database) as session:
            session.run(query, rows=rows)
        return len(rows)

    def load_self_play_games(self, limit: int | None = 50) -> list[dict]:
        """Load self-play games, oldest first (matches the old JSONL append order)."""
        query = "MATCH (g:SelfPlayGame) RETURN g ORDER BY g.played_at DESC"
        params: dict = {}
        if limit is not None:
            query += " LIMIT $limit"
            params["limit"] = limit

        with self._driver.session(database=self.database) as session:
            records = list(session.run(query, **params))

        rows = [self._decode_self_play_row(dict(record["g"])) for record in records]
        rows.reverse()
        return rows

    def load_self_play_game(self, run_id: str, index: int) -> dict | None:
        query = "MATCH (g:SelfPlayGame {game_key: $game_key}) RETURN g LIMIT 1"
        with self._driver.session(database=self.database) as session:
            record = session.run(query, game_key=f"{run_id}:{index}").single()
        if record is None:
            return None
        return self._decode_self_play_row(dict(record["g"]))


@contextmanager
def neo4j_store(**kwargs: object) -> Iterator[Neo4jStore]:
    store = Neo4jStore(**kwargs)  # type: ignore[arg-type]
    try:
        yield store
    finally:
        store.close()
