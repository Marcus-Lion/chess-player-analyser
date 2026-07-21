from __future__ import annotations

import argparse
from pathlib import Path

from app.rl.double_dqn import (
    DoubleDQNConfig,
    DoubleDQNModel,
    evaluate_matchup,
    train_from_self_play_double_dqn,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Double DQN chess experiment branch.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--target-update-every", type=int, default=10)
    parser.add_argument("--train-warmup", type=int, default=256)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-path", type=Path, default=Path("cache/rl_dqn_model.npz"))
    parser.add_argument("--load-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DoubleDQNConfig(
        episodes=max(1, args.episodes),
        max_turns=max(2, args.max_turns),
        replay_capacity=max(1, args.replay_capacity),
        batch_size=max(1, args.batch_size),
        learning_rate=max(1e-6, args.learning_rate),
        gamma=min(0.9999, max(0.0, args.gamma)),
        epsilon_start=max(0.0, args.epsilon_start),
        epsilon_end=max(0.0, args.epsilon_end),
        epsilon_decay=min(1.0, max(0.0, args.epsilon_decay)),
        updates_per_episode=max(1, args.updates_per_episode),
        target_update_every=max(1, args.target_update_every),
        train_warmup=max(1, args.train_warmup),
        step_penalty=args.step_penalty,
        hidden_dim=max(8, args.hidden_dim),
        seed=args.seed,
        save_every=max(1, args.save_every),
    )

    if args.load_path is not None and args.load_path.exists():
        model = DoubleDQNModel.load(args.load_path)
    else:
        model = DoubleDQNModel.initialize(seed=args.seed, hidden_dim=config.hidden_dim)

    run = train_from_self_play_double_dqn(
        model,
        config,
        save_path=args.save_path,
        seed=args.seed,
    )

    summary = evaluate_matchup(model, config, games=max(1, args.eval_games), seed=args.seed)
    last = run.latest
    if last is not None:
        print(
            f"episode={last.episode} transitions={last.transitions} "
            f"loss={last.loss:.6f} epsilon={last.epsilon:.4f} buffer={last.buffer_size} result={last.result}"
        )
    print(
        f"evaluation games={summary.games} white_wins={summary.white_wins} "
        f"black_wins={summary.black_wins} draws={summary.draws} white_score={summary.white_score:.3f}"
    )
    print(f"saved={args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
