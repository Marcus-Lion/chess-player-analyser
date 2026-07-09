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

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Hostinger VPS deployment

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv nginx git -y

git clone https://github.com/marcus-lion/chess-player-analyser.git
cd chess-player-analyser

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 127.0.0.1 --port 8000
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
