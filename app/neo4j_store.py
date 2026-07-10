from __future__ import annotations

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


@contextmanager
def neo4j_store(**kwargs: object) -> Iterator[Neo4jStore]:
    store = Neo4jStore(**kwargs)  # type: ignore[arg-type]
    try:
        yield store
    finally:
        store.close()
