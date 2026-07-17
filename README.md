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

Self-play results are stored in Neo4j (not a local cache file), so a running
Neo4j instance is required for `/self-play`, the CLI, and `--tune-weights`.
This reuses the same `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`/`NEO4J_DATABASE`
environment variables as the [Neo4j option](#neo4j-option) below (same
defaults), but is **not** gated by `NEO4J_ENABLED` — that flag only controls
the separate, opt-in export of real chess.com games from `/analyse`. See
[`neo4j_notes.md`](neo4j_notes.md) for local setup and the `:SelfPlayGame`
schema.

```bash
uv run python -m app.self_play --games 10 --max-turns 55
```

To control how many games run in parallel (processes), add `--workers`:

```bash
uv run python -m app.self_play --games 20 --workers 20 --max-turns 55
```

Each move is chosen by a negamax search with alpha-beta pruning, move
ordering (MVV-LVA, killer moves), and a transposition table. By default the
search **depth is inversely proportional to the material remaining on the
board**, scaled linearly from depth 1 at a full board (material 39, the
starting value for one side -- biggest branching factor, most expensive to
search) up to depth 7 once a side is down to material 0 (smallest branching
factor, and deeper search matters most for endgame precision). Pass
`--depth` to pin a fixed depth for the whole game instead (higher is slower
but stronger). See [`gameplay_rules.md`](gameplay_rules.md) for the full
move-selection and game-termination rules:

```bash
uv run python -m app.self_play --games 10 --depth 2 --max-turns 55 --top-k 1 --seed 1
```

The web form has the same knob as a **Parallel workers** field; leave it
blank for the "auto" default (CPU count).

You can also write PGN output to a file:

```bash
uv run python -m app.self_play --games 20 --output selfplay.pgn
```

To search for better score weights from recent self-play games:

```bash
uv run python -m app.self_play --tune-weights --tune-iterations 200 --tune-corpus-size 50
```

That prints the best weights found, then runs self-play with them. Use
`--tune-output weights.json` if you want the search result saved as JSON.

Multi-game self-play runs in a detached worker process that reports its
progress back to the main process over a local socket (a small in-process
job-status server, not a job-queue file). The browser gets those updates
pushed the instant they arrive over a `/self-play/ws/<job_id>` WebSocket,
instead of polling on a timer. The browser remembers the active job id, so
if the dev server reloads while a job is running, reopening the page
resumes the progress bar from the saved job id -- unless the server itself
restarted, since job status lives in memory and a worker whose connection
drops is reported as failed rather than tracked further. If that WebSocket
itself drops without ever reporting a terminal state -- the server process
died or restarted out from under it -- the browser treats the connection
loss itself as the terminal signal: it stops tracking the job, re-enables
the form, and tells the user to run again instead of leaving the progress
bar and submit button stuck forever.

Every game gets its own independently randomized set of weights by default,
for both the CLI and the web form. If you want to pit two fixed weight
profiles against each other instead, set all four weights (legal-moves,
material, forward, center) for a side — the web form's weight fields are
optional and only take effect once a side's full set is filled in.

Games end on checkmate, stalemate, insufficient material, threefold
repetition, or the fifty-move rule (the fivefold repetition and 75-move
rules also apply as automatic backstops). Since self-play has no player to
claim a draw, threefold repetition and the fifty-move rule are adjudicated
automatically as soon as they become claimable. If none of these trigger,
the game is called a draw once `--max-turns` is reached (labelled **Max
turns** in the web form and reported as "max turns reached"; the count
itself covers every half-move, i.e. both White's and Black's turns).

The "Analyse results →" link on `/self-play` opens `/self-play/analysis`, a
chart dashboard over every saved self-play game: outcome mix, termination
reasons, game-length distribution, rolling win/draw/loss rate over games
played, final-score spread by outcome, and — for tuning — white's win rate
bucketed by how much more (or less) of each score weight it had versus black.

The "Recent saved self-play results" table on `/self-play` can be filtered
by Result, Outcome (including an "Anyone wins" option that matches either
color winning), Termination, an absolute-value comparison (`>`/`<`) on
final score, and a Played-at date range.

Each saved game also records how long it took to play (wall-clock seconds
for the whole game) and how many leaf positions the search evaluated per
move on average -- both shown in the results table and on each game's
detail page.

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

## GCP Cloud Run deployment

The app also runs on Cloud Run, built from the repo's `Dockerfile` via Cloud
Build (no Artifact Registry setup needed):

```bash
gcloud run deploy chess-player-analyser \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 1800
```

Live deployment: project `chess-player-502601`, region `us-central1`, service
`chess-player-analyser` at
https://chess-player-analyser-859165106671.us-central1.run.app (public, no
auth -- matches the plain-HTTP Hostinger setup above). Sized at 2 vCPU / 2Gi
memory with a 30-minute request timeout (up from the Cloud Run defaults of 1
vCPU / 512Mi / 5 minutes) after self-play runs were getting OOM-killed --
Cloud Logging showed repeated `Memory limit of 512 MiB exceeded` errors that
took the WebSocket connection down mid-run. The underlying trigger was
`app/self_play.py`'s default worker count reading the *host* machine's CPU
count instead of what the container was actually allocated, over-spawning
worker processes; it now uses `os.process_cpu_count()`, which respects the
container's cgroup CPU quota.

### Private Neo4j on GCP

Self-play (and optional `/analyse` export) needs Neo4j. In production it runs
on a Compute Engine VM with **no external IP**, reachable only from Cloud Run
over a private VPC connection -- never exposed to the public internet:

- VM `neo4j-server` (e2-small, `us-central1-a`, Container-Optimized OS,
  `--no-address`) running the `neo4j:5` container.
- Firewall `allow-neo4j-from-vpc-connector`: tcp:7687 (bolt) only, source
  restricted to the VPC connector's subnet (`10.8.0.0/28`) -- never
  `0.0.0.0/0`.
- Serverless VPC Access connector `cloudrun-to-vpc` (`us-central1`) lets Cloud
  Run reach the VM's internal IP.
- Cloud Router/NAT (`nat-router`/`nat-config`) gives the VM outbound-only
  internet access (needed to pull container images) without an external IP or
  any inbound exposure.
- Cloud Run is wired to it with `--vpc-connector cloudrun-to-vpc` plus the
  same `NEO4J_ENABLED`/`NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` env vars as
  the [Neo4j option](#neo4j-option) below, with `NEO4J_URI` pointing at the
  VM's internal IP instead of `localhost`.

Neo4j Browser isn't exposed publicly; admin access is via an SSH tunnel:

```bash
gcloud compute ssh neo4j-server --zone=us-central1-a --tunnel-through-iap -- -L 7474:localhost:7474
# then open http://localhost:7474 locally
```

Data lives on the VM's boot disk rather than a separate persistent disk, so
deleting/recreating the VM loses it -- the same ephemeral-storage tradeoff
already accepted for Cloud Run's local file cache. Unlike Cloud Run, none of
this scales to zero: budget roughly $20-25/month ongoing for the VM,
connector, and NAT.

### Cost

- **Cloud Run** bills per request-second of actual CPU/memory used and
  scales to zero when idle, so the 1→2 vCPU / 512Mi→2Gi resize roughly
  doubles the *active-second* rate (~$0.000025/s → ~$0.000053/s) rather than
  adding a fixed cost. The free tier (180k vCPU-seconds + 360k GiB-seconds/
  month) covers light/occasional use; the main cost driver is self-play run
  duration now that the request timeout is 1800s instead of 300s -- a ~10
  minute self-play run at 2 vCPU/2Gi costs roughly $0.03.
- **Neo4j VM + connector + NAT** (above) is the real fixed cost: ~$20-25/month,
  billed continuously since none of it scales to zero, independent of how
  much the app is actually used.

## Game browser

Beyond the analytics dashboard you can inspect individual games move by move:

- `GET /games?username=<user>` lists every game in the player's archive
  (white/black, result, your win/loss/draw, date, time control, and the
  opening decoded from its ECO code into plain text) and lets you select one.
- `GET /games/<username>/<index>` opens an interactive viewer that renders
  every position of that game as a chess board (server-side SVG, no external
  assets). Step through positions with the ⏮ ◀ ▶ ⏭ buttons, the slider, the
  left/right arrow keys, or by clicking any move in the "Positions" list.
  The list lays out White's and Black's turns side by side, two turns per
  line, each numbered by its own turn count.
  Tick **Show all valid moves** to overlay arrows for every legal move in the
  current position and list them in SAN notation. The arrows are drawn thin
  with a dark border so they stay readable even when many overlap.
  The viewer also displays **Forward (1st and 2nd order)** metrics,
  a **Material** score that counts each side’s piece points
  (pawn=1, knight/bishop=3, rook=5, queen=9), a **Center** score that counts
  each side's attackers on the four central squares (d4/e4/d5/e5), and a
  **Legal-move Score** that counts how many legal moves are available to the
  side to move. The forward score tracks how many squares each side attacks
  on its forward two ranks; the material score tracks who is ahead on raw
  piece value.
  Based on the forward score, the viewer also **suggests the best 3 moves** for
  the current player (the legal moves leading to the best 1st order control
  balance).
  A **Position sub-graph** is shown on the right, next to the board: it draws
  the current position as a central node linked to the previous position and to
  every legal move (the move actually played is highlighted and clickable).
  Each played move is also checked for **blunders**: a 1-ply lookahead compares
  the position's evaluation right after the move to the worst case after every
  reply the opponent could make, and flags moves whose eval swing crosses one
  of three thresholds -- **Inaccuracy** (`?!`), **Mistake** (`?`), or **Blunder**
  (`??`), color-coded from amber through orange to red in the moves list. A
  forced stalemate is always scored as a draw rather than by raw material (so
  a stalemate escape from a losing position isn't flagged as a further
  mistake), and a reply that delivers checkmate is scored as an outright loss
  regardless of material on the board.

The results page also links straight to the browser via
"Browse all games and step through positions".

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
