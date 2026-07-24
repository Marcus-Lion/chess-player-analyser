# Chess Player Analyser — Project Document

This repository is a Python/FastAPI application for two related workflows:

1. analysing public Chess.com games for a username, and
2. running a self-play harness that evaluates the local chess engine against itself.

The codebase is split into a web app, a self-play engine, and a small amount of optional persistence in Neo4j.

## What the app does

- Fetches and caches Chess.com game archives.
- Parses PGNs into tabular data for analytics.
- Renders player summaries, charts, and individual game views in the web UI.
- Runs self-play matches in the background.
- Stores self-play results and player weights in Neo4j.
- Exposes a tuning loop that can rebalance player weights after batches of games.

## Main components

| Path | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI routes, page rendering, charts, and web endpoints. |
| `app/chesscom.py` | Chess.com archive fetching. |
| `app/parser.py` | PGN parsing into pandas data frames. |
| `app/metrics.py` | Human-game analytics such as monthly, rolling, and time-based summaries. |
| `app/games.py` | Engine evaluation, move selection, and game logic. |
| `app/self_play.py` | Self-play runner, worker orchestration, tuning, and weight rebalancing. |
| `app/self_play_metrics.py` | Self-play analytics, Elo estimation, termination summaries, and weight analysis. |
| `app/neo4j_store.py` | Optional Neo4j persistence for human games and self-play data. |
| `app/templates/` | Jinja2 HTML templates for the UI. |
| `engine/` | Native engine implementation used by self-play when built. |

## Web UI

The app serves a general analysis experience for human games and a separate self-play area.

Important routes:

- `/` — landing page / main analysis entry point.
- `/analyse` — fetch and analyse a Chess.com username.
- `/games?username=<user>` — list parsed games for a user.
- `/games/<username>/<index>` — inspect a single game move by move.
- `/self-play` — self-play control page.
- `/self-play/analysis` — summary dashboard for saved self-play games.
- `/self-play/terminations` — termination breakdown, including white/black win split.
- `/self-play/players` — self-play player overview.
- `/self-play/players/<player_id>` — per-player timeline and game history.

## Self-play flow

The self-play path works like this:

1. A run is configured from CLI arguments or the web form.
2. Games are launched in a process pool.
3. Each game selects a white and black player profile, then chooses moves with the engine.
4. Each finished game is saved.
5. After a batch reaches the configured rebalance size, player weights are updated.
6. Results are exposed in the web UI and stored in Neo4j when enabled.

The rebalance batch size is controlled by `SELF_PLAY_REBALANCE_BATCH_SIZE` in `.env`.
The current default in this repo is `250`.

## Environment variables

The app reads configuration from environment variables. `.env` is loaded by the app at startup.

| Variable | Purpose | Default / note |
| --- | --- | --- |
| `NEO4J_ENABLED` | Enables export of parsed human games to Neo4j. | Only affects human-game export. |
| `NEO4J_USERNAME` | Neo4j username. | `neo4j` if unset. |
| `NEO4J_PASSWORD` | Neo4j password. | `neo4j` if unset in code paths using `Neo4jStore`. |
| `NEO4J_DATABASE` | Neo4j database name. | `neo4j` |
| `NEO4J_URI` | Neo4j Bolt URI. | `bolt://localhost:7687` |
| `BASELINE_ELO` | Baseline Elo used by self-play analytics. | `1500` |
| `SELF_PLAY_PLAYER_WEIGHT_MIN` | Lower bound for per-player random weights. | `-4.0` |
| `SELF_PLAY_PLAYER_WEIGHT_MAX` | Upper bound for per-player random weights. | `4.0` |
| `SELF_PLAY_PLAYER_WEIGHT_STDDEV` | Present in `.env`, but not currently read by the code. | The player spread is currently hardcoded in `app/players.py` |
| `SELF_PLAY_REBALANCE_BATCH_SIZE` | Number of games between weight updates. | `250` in `.env` |
| `REPETITION_AVOIDANCE_MATERIAL_PAWNS` | Repetition-avoidance tuning parameter. | Used by engine logic |

## Data flow for human-game analysis

1. The app fetches a user’s Chess.com archives.
2. PGN is parsed into a pandas data frame.
3. Metrics and charts are generated from that table.
4. Results are rendered in HTML and optionally exported to Neo4j.

## Data flow for self-play

1. A roster of synthetic players is loaded.
2. Two players are selected for each game.
3. The engine plays the game, tracking the weights used on each move.
4. The game result and metadata are stored.
5. After each batch, the SHAP-style rebalance step updates player weights.
6. The stored history is used for player timelines, win-rate summaries, and termination tables.

## Performance

Performance has two layers in this project: the analytics/web layer and the self-play engine.

The web app is mostly pandas + Plotly + FastAPI. Its performance is dominated by data loading, chart generation, and any optional Neo4j queries.

The self-play engine is where the biggest performance work lives:

- `app/self_play.py` uses the required native `chess_engine` extension.
- That extension implements move generation and negamax search in Rust.
- `app.games.choose_engine_move` is the Python boundary for the native engine.
- The web harness and CLI can cap automatic depth with `max_depth`/
  `--max-depth`; the default cap is 7.
- The engine README describes it as roughly 30× faster at the same evals/move.

The runtime also uses process-level parallelism for multi-game runs:

- each game is scheduled as a separate process-pool task,
- `--workers` controls concurrency,
- the default worker count is capped by the number of games being run,
- on containerized deployments the code uses `os.process_cpu_count()` so it does not over-spawn based on the host machine’s CPU count.

Useful performance signals are printed per game:

- wall-clock duration in seconds,
- average evals per move,
- the chosen white/black player weights.

That makes it possible to spot regressions in search cost or game length.

There is one important persistence bottleneck to know about:

- every saved self-play game currently calls `refresh_self_play_player_elos()`,
- that refresh reloads the full saved self-play history from Neo4j,
- then recomputes the player overview across the entire corpus,
- so the save path gets more expensive as the database grows.

If self-play suddenly drops from something like 30 games/second to around 1 game/second, this full-history Neo4j refresh is a prime suspect.

## Important behavior to know

- White/black win rates are tracked separately in self-play analytics.
- The self-play termination page now shows a top-level white-vs-black summary.
- The timeline chart on a player page shows the weights used in each game, not just rebalance checkpoints.
- Rebalance updates are closed-loop: the result of one batch affects the next batch’s player selection and weights.
- If Neo4j is unavailable, self-play persistence and player-weight loading fall back to in-memory / code defaults where possible.
- If Neo4j is enabled and populated, per-game persistence can dominate total runtime unless the Elo refresh is batched or deferred.

## Running locally

```bash
uv sync
uv run python -m uvicorn app.main:app --port 8134
```

Self-play CLI:

```bash
uv run python -m app.self_play --games 10 --max-turns 55
```

Batch rebalancing and tuning:

```bash
uv run python -m app.self_play --tune-weights --tune-iterations 200 --tune-corpus-size 50
```

## Deployment notes

- [`DEPLOY.md`](DEPLOY.md) covers the Hostinger VPS path.
- `Dockerfile` supports container deployment.
- [`neo4j_notes.md`](neo4j_notes.md) documents the Neo4j setup and the self-play graph model.

## GCP deployment

The project is also deployed on Google Cloud Run from the repo’s `Dockerfile`.

Typical deployment shape:

```bash
gcloud run deploy chess-player-analyser \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 1800
```

Key points:

- Cloud Run is the application host.
- The service is sized up to 2 vCPU / 2 GiB to keep self-play runs from being memory-starved.
- The request timeout is extended to support long self-play sessions.
- The app uses `os.process_cpu_count()` for worker sizing so container CPU limits are respected.
- The live deployment in this repo is:
  - project: `chess-player-502601`
  - region: `us-central1`
  - service: `chess-player-analyser`
- The public URL is the Cloud Run service URL; the app is exposed without auth.

Neo4j for self-play is kept private on GCP:

- it runs on a Compute Engine VM with no external IP,
- the VM is reached from Cloud Run through a VPC connector,
- Neo4j Bolt is firewall-restricted to the connector subnet,
- admin/browser access is intended to go through SSH tunneling rather than public exposure.
- the VM uses Cloud NAT for outbound-only internet access so it can pull images without public ingress.

Operational notes:

- Cloud Run should be deployed with the same environment variables used locally for Neo4j and baseline tuning.
- Self-play depends on Neo4j for persistent results and weight updates; if the database is unreachable, the app can still run but the closed-loop tuning path is incomplete.
- Long self-play runs should be monitored for memory pressure and worker count; the container CPU quota should always be the source of truth.

This architecture keeps the web app public while leaving the graph database private.

## Troubleshooting

- If self-play updates seem to happen too often, check that `.env` is loaded before `SELF_PLAY_REBALANCE_BATCH_SIZE` is read.
- If the UI shows unexpected win-rate trends, check the most recent rebalance batch and the current player roster.
- If Neo4j auth fails, self-play can still run, but persistence and stored-player loading will be incomplete.
- If performance drops sharply, confirm `PYTHON_FALLBACK=false` and that the Rust `chess_engine` extension imports in the deployed image.
