# Neo4j option

The MVP uses local file caching for analytics. Neo4j is useful when you want
to explore the games as a graph — for example, shared opponents, rating
neighbourhoods, or head-to-head paths between players.

## Graph model

```text
(:Player {username})-[:PLAYED]->(:Game {game_id, ...})
(:Game)-[:AGAINST]->(:Opponent {name})
```

Each `Game` node carries the parsed properties: `utc_datetime`,
`local_datetime`, `local_date`, `local_hour`, `local_day`, `month`, `color`,
`score`, `opponent_rating`, `user_rating`, `result`, `termination`,
`time_control`, `moves`, and `clock_count`.

## Package

```bash
uv add neo4j
```

(Already included in `pyproject.toml`.)

## Run Neo4j locally

```bash
docker run --rm \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5
```

## Configuration

Export to Neo4j is opt-in via environment variables so the analytics engine is
unchanged when Neo4j is not configured:

```bash
export NEO4J_ENABLED=true
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
export NEO4J_DATABASE=neo4j
```

When `NEO4J_ENABLED` is truthy, every `/analyse` request upserts the parsed
games into the graph. Any connection error is swallowed so requests never fail
because of an unavailable database.

## Programmatic use

```python
from app.neo4j_store import Neo4jStore
from app.parser import parse_pgn_to_dataframe

df = parse_pgn_to_dataframe(pgn_text, username="marcus")
with Neo4jStore() as store:
    written = store.save_games("marcus", df)
    print(f"wrote {written} games")
```

## Example queries

Top opponents by games played:

```cypher
MATCH (:Player {username: "marcus"})-[:PLAYED]->(:Game)-[:AGAINST]->(o:Opponent)
RETURN o.name AS opponent, count(*) AS games
ORDER BY games DESC
LIMIT 10;
```

Win rate by day of week:

```cypher
MATCH (:Player {username: "marcus"})-[:PLAYED]->(g:Game)
RETURN g.local_day AS day, avg(g.score) AS win_rate, count(*) AS games
ORDER BY win_rate DESC;
```
