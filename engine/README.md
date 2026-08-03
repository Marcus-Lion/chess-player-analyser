# chess_engine (native self-play engine)

A Rust extension module that runs the self-play per-move search natively, as a
backend for `app.games.choose_engine_move`. `app/self_play.py` imports it as a
required extension (`import chess_engine`); startup fails with build
instructions when it is unavailable. It implements the evaluation heuristics
and negamax/alpha-beta search natively and is roughly **30× faster** than the
retired Python search at the same evals/move.

Built on [shakmaty](https://docs.rs/shakmaty) (chess rules / move generation /
Zobrist hashing) via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

## Important references
https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf


## Build
From the repo root, into the project's uv venv:

```powershell
.venv/Scripts/maturin.exe develop --release -m engine/Cargo.toml
```

If maturin is not installed in the project venv, build a wheel and install it
explicitly. The PowerShell variable is required because `uv pip` does not
expand the `*.whl` wildcard itself:

```powershell
New-Item -ItemType Directory -Force engine/dist
uv run --with maturin maturin build --release -m engine/Cargo.toml -o engine/dist
$wheel = Get-ChildItem engine/dist -Filter 'chess_engine-*.whl' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
uv pip install --python .venv/Scripts/python.exe $wheel.FullName --force-reinstall
```

(omit `--release` for a faster, unoptimized debug build while iterating.)

Then confirm:

```powershell
.venv/Scripts/python.exe -c "import chess_engine; print('ok')"
```

### Toolchain note (Windows)

This project's Python is an **MSVC** build, but a working MSVC C++ linker isn't
required: this machine builds the extension with the **GNU** Rust toolchain,
which ships a self-contained linker. The resulting `.pyd` loads fine into the
MSVC CPython. The `engine/` directory is pinned to that toolchain via a rustup
directory override:

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu   # one-time
rustup override set stable-x86_64-pc-windows-gnu         # run inside engine/
```

If you install the Visual Studio C++ Build Tools instead, you can drop the
override and build with the default `stable-x86_64-pc-windows-msvc` toolchain.

On Linux/Docker the default host toolchain works as-is (no override needed).

## Exposed functions

- `choose_engine_move(fen, depth, top_k, seed, legal_moves_weight,
  material_score_weight, forward_score_weight, forward_material_score_weight,
  center_control_weight,
  checkmate_weight, repetition_counts, top_k_score_threshold=3.0) ->
  (uci, score, evaluations)`
- `choose_engine_move_from_history(fen, history_fens, ...) ->
  (uci, score, evaluations)` — builds repetition hashes and counts natively
  from historical FENs before searching.
- `play_self_game_native(...) ->
  (result, termination, turns, moves, evaluations)`
  — runs the complete board/move loop in Rust and returns UCI moves for PGN
  formatting by the Python caller.

`forward_material_score_weight` defaults to `1.0`. It controls the material
value assigned to pieces occupying the side's forward zone. Set it to `0` to
disable the heuristic or increase it to reward advanced material more strongly.

- `calculate_forward(fen) -> (forward_1, forward_2, forward_3)` and
  `calculate_forward_4(fen) -> (white, black)` — native forward-control
  metrics used by the game viewer.
- `calculate_forward_material(fen) -> (white, black)` — material value in
  each side's forward zone. The static evaluator blends 25% of this
  differential into the normal material term, rewarding advanced material
  without allowing it to outweigh the actual material count.
- `calculate_center_control`, `calculate_flank_control`,
  `calculate_king_escape_squares`, `calculate_mate_pressure`, and
  `calculate_auto_search_depth` provide the remaining native board metrics
  used by the evaluator and viewer. `calculate_phase_value` keeps the
  material-based opening/endgame phase calculation native as well.
- `evaluate_position(fen, legal_moves_weight, material_score_weight,
  forward_score_weight, center_control_weight, checkmate_weight,
  forward_material_score_weight=1.0) -> float`
  — White-perspective static eval, used for parity checks against
  `app.games._evaluate_position`.

## Single-process self-play throughput

The default automatic self-play profile uses a maximum search depth of 2.
This keeps a 100-ply game under 10 seconds on a single worker process on the
development machine. Higher values can be requested with `max_depth` (or the
`--max-depth` CLI option), but runtime rises sharply with depth because the
search tree expands exponentially.

In the self-play harness, `Max depth` may be left blank to use this default.

## Phase transitions

The game phase is determined after each move from the remaining non-pawn
material. Pawns and kings are excluded. Minor pieces (knights and bishops)
count as 1 point each, rooks as 2 points each, and queens as 4 points each.

| Phase | Material score | Typical emphasis |
| --- | ---: | --- |
| Opening | > 13 | Center control and king safety |
| Middlegame | 10–13 | Balanced positional factors |
| Endgame | 0–9 | Material, mobility, and king activity |

The phase is recalculated from the current board after every move, so a
capture can transition the game immediately. An opening pawn sacrifice alone
does not trigger an endgame transition because pawns are not included in the
phase score.
