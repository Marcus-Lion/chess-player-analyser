# Marcus Lion Chess Player Analyser

A Python/FastAPI web app that fetches public Chess.com games for any username and generates chess analytics.

## Features

- Monthly performance rating
- Rolling 100-game performance rating
- Opponent strength trends
- Time-of-day performance
- Day-of-week performance
- Time × day matrix
- Before/after breakpoint analysis
- Basic clock/time-management stats when `%clk` annotations are present
- Game browser: list every game for a player, select one, and step through each position on a board

## Quick start

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m uvicorn app.main:app --port 8134
```

Open:

```text
http://127.0.0.1:8134
```

## Self-play harness

The repo also includes a headless harness that plays the current scoring
engine against itself from the starting position, or from a supplied FEN.

```bash
uv run python -m app.self_play --games 10 --max-plies 200 --top-k 3 --seed 1
```

You can also write PGN output to a file:

```bash
uv run python -m app.self_play --games 20 --output selfplay.pgn
```

## Hostinger VPS deployment

### Automated (recommended)

SSH into your VPS as root and run the bundled `deploy.sh`. It installs
system packages and `uv`, clones the repo, syncs dependencies, and sets up
`systemd` + Nginx (and optionally HTTPS via Let's Encrypt):

```bash
curl -LsSf https://raw.githubusercontent.com/marcus-lion/chess-player-analyser/main/deploy.sh -o deploy.sh
chmod +x deploy.sh
sudo DOMAIN=yourdomain.com EMAIL=you@yourdomain.com ./deploy.sh
```

Omit `DOMAIN`/`EMAIL` to deploy over plain HTTP on the server IP.

### Manual

```bash
sudo apt update && sudo apt upgrade -y
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/marcus-lion/chess-player-analyser.git
cd chess-player-analyser

uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8134
```

For production, run with `systemd` and put Nginx in front as a reverse proxy.

## Game browser

Beyond the analytics dashboard you can inspect individual games move by move:

- `GET /games?username=<user>` lists every game in the player's archive
  (white/black, result, your win/loss/draw, date, time control, and the
  opening decoded from its ECO code into plain text) and lets you select one.
- `GET /games/<username>/<index>` opens an interactive viewer that renders
  every position of that game as a chess board (server-side SVG, no external
  assets). Step through positions with the ⏮ ◀ ▶ ⏭ buttons, the slider, the
  left/right arrow keys, or by clicking any move in the "Positions" list.
  Tick **Show all valid moves** to overlay arrows for every legal move in the
  current position and list them in SAN notation. The arrows are drawn thin
  with a dark border so they stay readable even when many overlap.
  The viewer also displays **Forward Control (1st and 2nd order)** metrics,
  a **Material** score that counts each side’s piece points
  (pawn=1, knight/bishop=3, rook=5, queen=9), and a **Legal-move Score** that
  counts how many legal moves are available to the side to move. The control
  score tracks how many squares each side attacks on its forward two ranks; the
  material score tracks who is ahead on raw piece value.
  Based on the control score, the viewer also **suggests the best 3 moves** for
  the current player (the legal moves leading to the best 1st order control
  balance).
  A **Position sub-graph** is shown on the right, next to the board: it draws
  the current position as a central node linked to the previous position and to
  every legal move (the move actually played is highlighted and clickable).

The results page also links straight to the browser via
"Browse all games and step through positions".

## Firebase option

Firebase is useful for:
- user accounts/auth
- saving analysis snapshots
- Firestore for cached profile metadata
- Cloud Storage for raw PGN archives
- hosting a separate frontend

For this MVP, local cache files are simpler. Firebase can be added later without changing the analytics engine.

## Neo4j option

Neo4j lets you explore games as a graph (players, games, opponents) for
queries like shared opponents, head-to-head paths, and rating neighbourhoods.

Export is opt-in and does not change the analytics engine. Enable it with
environment variables:

```bash
export NEO4J_ENABLED=true
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

When enabled, each `/analyse` request upserts the parsed games into Neo4j.
See [`neo4j_notes.md`](neo4j_notes.md) for the graph model and example
Cypher queries.
