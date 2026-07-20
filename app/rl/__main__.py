from __future__ import annotations

import argparse
from pathlib import Path

from app.rl.config import RLConfig
from app.rl.evaluate import evaluate_matchup
from app.rl.model import ChessRLModel
from app.rl.presets import PRESETS
from app.rl.training import train_from_self_play


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal RL self-play training loop.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--self-play-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument("--value-hidden-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--exploration", type=float, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-path", type=Path, default=Path("cache/rl_model.npz"))
    parser.add_argument("--samples-path", type=Path, default=Path("cache/rl_samples.jsonl"))
    parser.add_argument("--load-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = PRESETS[args.preset]

    def pick(name: str, fallback: object) -> object:
        value = getattr(args, name)
        return value if value is not None else preset.get(name, fallback)

    config = RLConfig(
        episodes=max(1, int(pick("episodes", 100))),
        max_turns=max(2, int(pick("max_turns", 80))),
        replay_capacity=max(1, int(pick("replay_capacity", 10_000))),
        batch_size=max(1, int(pick("batch_size", 32))),
        self_play_workers=max(1, int(pick("self_play_workers", 1))),
        self_play_temperature=max(0.0, float(pick("temperature", 1.0))),
        self_play_exploration=max(0.0, float(pick("exploration", 0.1))),
        learning_rate=max(1e-6, float(pick("learning_rate", 1e-3))),
        policy_hidden_dim=max(8, args.policy_hidden_dim),
        value_hidden_dim=max(8, args.value_hidden_dim),
        eval_games=max(1, int(pick("eval_games", 10))),
        seed=args.seed,
    )

    if args.load_path is not None and args.load_path.exists():
        model = ChessRLModel.load(args.load_path)
    else:
        model = ChessRLModel.initialize(
            seed=args.seed,
            policy_hidden_dim=config.policy_hidden_dim,
            value_hidden_dim=config.value_hidden_dim,
        )

    run = train_from_self_play(
        model,
        config,
        save_path=args.save_path,
        samples_path=args.samples_path,
        seed=args.seed,
    )
    summary = evaluate_matchup(model, config, games=config.eval_games, seed=args.seed)
    last = run.latest
    if last is not None:
        print(
            f"episode={last.episode} samples={last.samples} "
            f"policy_loss={last.policy_loss:.4f} value_loss={last.value_loss:.4f}"
        )
    print(
        f"evaluation games={summary.games} white_wins={summary.white_wins} "
        f"black_wins={summary.black_wins} draws={summary.draws} "
        f"white_score={summary.white_score:.3f}"
    )
    print(f"saved={args.save_path}")
    print(f"samples={args.samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
