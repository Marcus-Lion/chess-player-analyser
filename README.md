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

## Hostinger VPS deployment

```bash
sudo apt update && sudo apt upgrade -y
# Install uv
curl -LsSf https://astral-sh.io/uv/install.sh | sh
source $HOME/.cargo/env

git clone https://github.com/marcus-lion/chess-player-analyser.git
cd chess-player-analyser

uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8134
```

For production, run with `systemd` and put Nginx in front as a reverse proxy.

## Firebase option

Firebase is useful for:
- user accounts/auth
- saving analysis snapshots
- Firestore for cached profile metadata
- Cloud Storage for raw PGN archives
- hosting a separate frontend

For this MVP, local cache files are simpler. Firebase can be added later without changing the analytics engine.
