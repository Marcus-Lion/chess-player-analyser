from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from app.self_play_metrics import player_overview, to_dataframe as self_play_to_dataframe

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

    Self-play results use a separate namespace so they do not collide with
    imported human games:

        (:SelfPlayPlayer {player_id, name, description, weights...})
            {elo}
            <-[:PLAYED_AS_WHITE]- (:SelfPlayGame {game_key, ...})
            <-[:PLAYED_AS_BLACK]-
            -[:ENDED_BY]-> (:SelfPlayTermination {termination_key, label})

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
        self.user = user or os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
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
        statements = [
            "CREATE CONSTRAINT self_play_game_key IF NOT EXISTS "
            "FOR (g:SelfPlayGame) REQUIRE g.game_key IS UNIQUE",
            "CREATE CONSTRAINT self_play_player_id IF NOT EXISTS "
            "FOR (p:SelfPlayPlayer) REQUIRE p.player_id IS UNIQUE",
            "CREATE CONSTRAINT self_play_termination_key IF NOT EXISTS "
            "FOR (t:SelfPlayTermination) REQUIRE t.termination_key IS UNIQUE",
        ]
        with self._driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    @staticmethod
    def _encode_self_play_row(game: dict) -> dict:
        row = dict(game)
        row["game_key"] = f"{row.get('run_id')}:{row.get('index')}"
        white_weights = row.pop("white_weights", None) or {}
        black_weights = row.pop("black_weights", None) or {}
        row["white_weights_json"] = json.dumps(white_weights)
        row["black_weights_json"] = json.dumps(black_weights)
        row["white_legal_moves_weight"] = white_weights.get("legal_moves_weight")
        row["white_material_score_weight"] = white_weights.get("material_score_weight")
        row["white_forward_score_weight"] = white_weights.get("forward_score_weight")
        row["white_center_control_weight"] = white_weights.get("center_control_weight")
        row["black_legal_moves_weight"] = black_weights.get("legal_moves_weight")
        row["black_material_score_weight"] = black_weights.get("material_score_weight")
        row["black_forward_score_weight"] = black_weights.get("forward_score_weight")
        row["black_center_control_weight"] = black_weights.get("center_control_weight")
        row["termination_key"] = row.get("termination")
        row["termination_label"] = row.get("termination")
        return row

    @staticmethod
    def _decode_self_play_row(props: dict) -> dict:
        row = dict(props)
        row.pop("game_key", None)
        row["white_weights"] = json.loads(row.pop("white_weights_json", "null") or "null")
        row["black_weights"] = json.loads(row.pop("black_weights_json", "null") or "null")
        return row

    def save_self_play_games(self, games: list[dict]) -> int:
        """Upsert self-play game results and player nodes, keyed on run/index."""
        if not games:
            return 0

        rows = [self._encode_self_play_row(game) for game in games]

        query = """
        UNWIND $rows AS row
        MERGE (g:SelfPlayGame {game_key: row.game_key})
        SET g += row
        WITH g, row
        FOREACH (_ IN CASE WHEN row.white_player_id IS NULL THEN [] ELSE [1] END |
            MERGE (white:SelfPlayPlayer {player_id: row.white_player_id})
            SET white.name = row.white_player_name,
                white.description = row.white_player_description,
                white.legal_moves_weight = row.white_legal_moves_weight,
                white.material_score_weight = row.white_material_score_weight,
                white.forward_score_weight = row.white_forward_score_weight,
                white.center_control_weight = row.white_center_control_weight
            MERGE (g)-[:PLAYED_AS_WHITE]->(white)
        )
        FOREACH (_ IN CASE WHEN row.black_player_id IS NULL THEN [] ELSE [1] END |
            MERGE (black:SelfPlayPlayer {player_id: row.black_player_id})
            SET black.name = row.black_player_name,
                black.description = row.black_player_description,
                black.legal_moves_weight = row.black_legal_moves_weight,
                black.material_score_weight = row.black_material_score_weight,
                black.forward_score_weight = row.black_forward_score_weight,
                black.center_control_weight = row.black_center_control_weight
            MERGE (g)-[:PLAYED_AS_BLACK]->(black)
        )
        FOREACH (_ IN CASE WHEN row.termination_key IS NULL THEN [] ELSE [1] END |
            MERGE (t:SelfPlayTermination {termination_key: row.termination_key})
            SET t.label = row.termination_label
            MERGE (g)-[:ENDED_BY]->(t)
        )
        """

        self.ensure_self_play_constraints()
        with self._driver.session(database=self.database) as session:
            session.run(query, rows=rows)
        self.refresh_self_play_player_elos()
        return len(rows)

    def refresh_self_play_player_elos(self) -> int:
        """Recompute and persist Elo on every self-play player node."""
        rows = self.load_self_play_games(limit=None)
        if not rows:
            return 0

        df = self_play_to_dataframe(rows)
        if df.empty:
            return 0

        stats = player_overview(df)
        player_rows = [
            {
                "player_id": str(row.player_id),
                "elo": float(row.elo),
            }
            for row in stats.itertuples(index=False)
            if getattr(row, "player_id", None) is not None
        ]
        if not player_rows:
            return 0

        query = """
        UNWIND $rows AS row
        MATCH (p:SelfPlayPlayer {player_id: row.player_id})
        SET p.elo = row.elo
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, rows=player_rows)
        return len(player_rows)

    def load_self_play_players(self) -> list[dict]:
        """Load all self-play players with their current weights and Elo."""
        query = """
        MATCH (p:SelfPlayPlayer)
        RETURN p
        ORDER BY coalesce(p.elo, 1500.0) DESC, p.player_id
        """
        with self._driver.session(database=self.database) as session:
            records = list(session.run(query))
        return [dict(record["p"]) for record in records]

    def update_self_play_player_weights(self, rows: list[dict]) -> int:
        """Persist updated self-play weights for the provided player rows."""
        if not rows:
            return 0

        query = """
        UNWIND $rows AS row
        MATCH (p:SelfPlayPlayer {player_id: row.player_id})
        SET p.name = coalesce(row.player_name, p.name),
            p.description = coalesce(row.player_description, p.description),
            p.legal_moves_weight = row.updated_legal_moves_weight,
            p.material_score_weight = row.updated_material_score_weight,
            p.forward_score_weight = row.updated_forward_score_weight,
            p.center_control_weight = row.updated_center_control_weight,
            p.last_balance_games = row.games,
            p.last_balance_score_pct = row.score_pct,
            p.last_balance_shap_legal_moves_weight = row.shap_legal_moves_weight,
            p.last_balance_shap_material_score_weight = row.shap_material_score_weight,
            p.last_balance_shap_forward_score_weight = row.shap_forward_score_weight,
            p.last_balance_shap_center_control_weight = row.shap_center_control_weight,
            p.last_balance_delta_legal_moves_weight = row.delta_legal_moves_weight,
            p.last_balance_delta_material_score_weight = row.delta_material_score_weight,
            p.last_balance_delta_forward_score_weight = row.delta_forward_score_weight,
            p.last_balance_delta_center_control_weight = row.delta_center_control_weight
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, rows=rows)
        return len(rows)

    def delete_self_play_players(self) -> int:
        """Delete every saved self-play player node and detach its relationships."""
        query = """
        MATCH (p:SelfPlayPlayer)
        WITH collect(p) AS players, count(p) AS deleted
        FOREACH (player IN players | DETACH DELETE player)
        RETURN deleted
        """
        with self._driver.session(database=self.database) as session:
            record = session.run(query).single()
        return int(record["deleted"]) if record is not None else 0

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

    def delete_self_play_games(self) -> int:
        """Delete every saved self-play game result."""
        with self._driver.session(database=self.database) as session:
            games_record = session.run(
                """
                MATCH (g:SelfPlayGame)
                WITH collect(g) AS games, count(g) AS deleted
                FOREACH (game IN games | DETACH DELETE game)
                RETURN deleted
                """
            ).single()
            players_record = session.run(
                """
                MATCH (p:SelfPlayPlayer)
                WHERE NOT (p)--()
                WITH collect(p) AS players, count(p) AS deleted
                FOREACH (player IN players | DETACH DELETE player)
                RETURN deleted
                """
            ).single()
            terminations_record = session.run(
                """
                MATCH (t:SelfPlayTermination)
                WHERE NOT (t)--()
                WITH collect(t) AS terminations, count(t) AS deleted
                FOREACH (termination IN terminations | DETACH DELETE termination)
                RETURN deleted
                """
            ).single()

        games_deleted = int(games_record["deleted"]) if games_record is not None else 0
        players_deleted = int(players_record["deleted"]) if players_record is not None else 0
        terminations_deleted = int(terminations_record["deleted"]) if terminations_record is not None else 0
        return games_deleted + players_deleted + terminations_deleted


@contextmanager
def neo4j_store(**kwargs: object) -> Iterator[Neo4jStore]:
    store = Neo4jStore(**kwargs)  # type: ignore[arg-type]
    try:
        yield store
    finally:
        store.close()
