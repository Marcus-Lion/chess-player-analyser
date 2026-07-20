# RL subsystem

This document describes the reinforcement-learning scaffold added to the chess-player-analyser project.

## Goal

The RL layer replaces hand-tuned move weights with a learnable policy/value model trained from self-play.

The current implementation is intentionally minimal:

- self-play generates training samples
- a small NumPy policy/value network trains on those samples
- checkpoints and sample logs are written to disk
- evaluation plays the model against the existing heuristic engine
- a FastAPI page lets you launch runs from the web UI

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
- `--episodes` — override preset episode count
- `--max-turns` — override maximum turns per game

Example:

```bash
python -m app.rl --preset quick --save-path cache/rl_model.npz --samples-path cache/rl_samples.jsonl
```

## Web usage

Open `/rl` in the app and submit a run. The page shows:

- current state
- progress
- last policy/value losses
- evaluation summary

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
- no opening book or curriculum yet

The model is useful as a scaffold, not as a final chess engine.

## Suggested next steps

1. Replace the plain self-play policy target with MCTS visit counts.
2. Add persistent replay storage.
3. Add checkpoint-versus-checkpoint evaluation.
4. Add training curves to the UI.
5. Introduce a stronger model backend if needed.

## Related files

- [app/rl/__main__.py](app/rl/__main__.py)
- [app/rl/service.py](app/rl/service.py)
- [app/rl/model.py](app/rl/model.py)
- [app/rl/self_play_rl.py](app/rl/self_play_rl.py)
- [app/main.py](app/main.py)
