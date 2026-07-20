# RL subsystem

This document describes the reinforcement-learning scaffold added to the chess-player-analyser project.

## Goal

The RL layer replaces hand-tuned move weights with a learnable policy/value model trained from self-play.

The current implementation is intentionally minimal:

- self-play generates training samples
- self-play uses MCTS visit counts as the policy target
- self-play episodes can run in parallel worker processes
- a small NumPy policy/value network trains on those samples
- checkpoints and sample logs are written to disk
- evaluation plays the model against the existing heuristic engine
- a FastAPI page lets you launch runs from the web UI

There is now also a Gymnasium/SB3 track for experimenting with PPO against the
heuristic baseline using action masking.

## What is currently implemented

### Core package

The RL code lives in `app/rl/`.

- `config.py` — training/runtime config
- `encoding.py` — board and move feature encoding
- `dataset.py` — sample and episode dataclasses
- `replay_buffer.py` — in-memory replay buffer
- `model.py` — simple NumPy policy/value model
- `self_play_rl.py` — self-play generation
- `training.py` — training loop
- `evaluate.py` — evaluation against the heuristic engine
- `gym_env.py` — Gymnasium chess environment with invalid-action masking
- `sb3_train.py` — SB3/PPO training entrypoint for the Gym environment
- `service.py` — background job wrapper for the web app
- `presets.py` — named run presets
- `__main__.py` — CLI entrypoint

### Web integration

The FastAPI app now exposes:

- `GET /rl` — RL training page
- `POST /rl/start` — start a background RL run
- `GET /rl/status` — poll the current job status

The UI links are available from the home page and the self-play page.

## Model design

The first version is deliberately small and dependency-light.

### Inputs

The board encoder uses a flat feature vector with:

- 12 piece planes for white/black × piece type
- side to move
- castling rights
- en passant file
- halfmove clock and fullmove number

### Outputs

The model produces:

- a policy distribution over legal moves
- a scalar value in `[-1, 1]` from the side-to-move perspective

### Training objective

The current loss is:

- policy cross-entropy
- plus value regression loss

The training target is derived from self-play:

- policy target: the chosen move, or a soft target if provided later
- value target: final game result mapped to win/draw/loss

## Current run flow

1. Create or load a checkpoint.
2. Generate one self-play game at a time.
3. Store each position as a training sample.
4. Append samples to a JSONL log on disk.
5. Sample minibatches from replay.
6. Update the network.
7. Save checkpoints periodically.
8. Evaluate the new model against the heuristic engine.

## CLI usage

Run the RL loop from the command line:

```bash
python -m app.rl --preset smoke
```

Available presets:

- `smoke`
- `quick`
- `standard`
- `long`

Useful options:

- `--save-path` — checkpoint output path
- `--samples-path` — JSONL sample log path
- `--load-path` — resume from a saved checkpoint
- `--self-play-workers` — parallel self-play processes per training chunk
- `--episodes` — override preset episode count
- `--max-turns` — override maximum turns per game

Example:

```bash
python -m app.rl --preset quick --save-path cache/rl_model.npz --samples-path cache/rl_samples.jsonl
```

### Gymnasium + SB3 track

The alternative PPO path uses a custom chess environment that plays the
baseline engine as the opponent and exposes an action mask for legal moves.

Run it with:

```bash
python -m app.rl.sb3_train --timesteps 100000 --eval-games 20
```

Useful options:

- `--agent-color` — `white`, `black`, or `random`
- `--max-agent-turns` — cap the number of agent decisions per game
- `--opponent-depth` — baseline opponent search depth
- `--save-path` — SB3 checkpoint output path
- `--load-path` — resume from a saved SB3 checkpoint

The environment is designed around the current heuristic engine and keeps the
reward signal small and shaped, so it is a scaffold rather than a finished
AlphaZero-style setup.

## Web usage

Open `/rl` in the app and submit a run. The page shows:

- current state
- progress
- last policy/value losses
- training curves for policy and value loss
- evaluation summary

The form includes a `Self-play workers` field so you can run more than one
episode at a time using processes.

The job runs in a background thread, so the request returns immediately.

## Files written by default

Typical RL runs write:

- `cache/rl_model.npz` — checkpoint
- `cache/rl_samples.jsonl` — training samples

These paths can be overridden from the CLI or the web form.

## Important limitations

This is not yet a full-strength chess RL system.

Current limitations:

- no MCTS
- no experience replay persistence across restarts
- no distributed training
- no adversarial evaluation pool
- no GPU-accelerated backend
- no full self-play PPO opponent pool yet
- no opening book or curriculum yet

The model is useful as a scaffold, not as a final chess engine.

## Suggested next steps

1. Add persistent replay storage.
2. Add checkpoint-versus-checkpoint evaluation.
3. Introduce a stronger model backend if needed.

## Related files

- [app/rl/__main__.py](app/rl/__main__.py)
- [app/rl/service.py](app/rl/service.py)
- [app/rl/model.py](app/rl/model.py)
- [app/rl/self_play_rl.py](app/rl/self_play_rl.py)
- [app/main.py](app/main.py)
