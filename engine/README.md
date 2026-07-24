# chess_engine (native self-play engine)

A Rust extension module that runs the self-play per-move search natively, as a
backend for `app.games.choose_engine_move`. `app/self_play.py` imports it as a
required extension (`import chess_engine`); startup fails with build
instructions when it is unavailable. It implements the evaluation heuristics
and negamax/alpha-beta search natively and is roughly **30× faster** than the
retired Python search at the same evals/move.

Built on [shakmaty](https://docs.rs/shakmaty) (chess rules / move generation /
Zobrist hashing) via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

## Build

From the repo root, into the project's uv venv:

```bash
maturin develop --release -m engine/Cargo.toml
```

(omit `--release` for a faster, unoptimized debug build while iterating.)

Then confirm:

```bash
python -c "import chess_engine; print('ok')"
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
  material_score_weight, forward_score_weight, center_control_weight,
  checkmate_weight, history_fens) -> (uci, score, evaluations)`
- `evaluate_position(fen, legal_moves_weight, material_score_weight,
  forward_score_weight, center_control_weight, checkmate_weight) -> float`
  — White-perspective static eval, used for parity checks against
  `app.games._evaluate_position`.
