from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.rl.gym_env import ChessGymConfig, ChessMatchEnv


@dataclass(slots=True)
class SB3EvalSummary:
    games: int
    white_wins: int
    black_wins: int
    draws: int
    agent_wins: int
    agent_losses: int
    agent_draws: int

    @property
    def agent_score(self) -> float:
        return (self.agent_wins + 0.5 * self.agent_draws) / max(1, self.games)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the chess baseline via a Gymnasium environment.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-path", type=Path, default=Path("cache/rl_sb3_model.zip"))
    parser.add_argument("--load-path", type=Path, default=None)
    parser.add_argument("--agent-color", choices=("white", "black", "random"), default="random")
    parser.add_argument("--max-agent-turns", type=int, default=100)
    parser.add_argument("--opponent-depth", type=int, default=3)
    parser.add_argument("--opponent-top-k", type=int, default=1)
    parser.add_argument("--illegal-move-penalty", type=float, default=-1.0)
    parser.add_argument("--step-penalty", type=float, default=-0.005)
    parser.add_argument("--shaping-scale", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser


def _load_sb3():
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "Gym/SB3 training needs gymnasium, stable-baselines3, and sb3-contrib installed."
        ) from exc
    return MaskablePPO, DummyVecEnv


def _build_env_config(args: argparse.Namespace) -> ChessGymConfig:
    return ChessGymConfig(
        max_agent_turns=max(1, args.max_agent_turns),
        agent_color=args.agent_color,
        opponent_depth=max(1, args.opponent_depth),
        opponent_top_k=max(1, args.opponent_top_k),
        illegal_move_penalty=args.illegal_move_penalty,
        step_penalty=args.step_penalty,
        shaping_scale=args.shaping_scale,
    )


def evaluate_agent(
    model,
    *,
    config: ChessGymConfig,
    games: int,
    seed: int | None = None,
) -> SB3EvalSummary:
    env = ChessMatchEnv(config)
    rng_seed = seed
    white_wins = black_wins = draws = 0
    agent_wins = agent_losses = agent_draws = 0

    for game_index in range(max(1, games)):
        agent_color = "white" if game_index % 2 == 0 else "black"
        obs, _info = env.reset(
            seed=None if rng_seed is None else rng_seed + game_index,
            options={"agent_color": agent_color},
        )
        terminated = truncated = False
        last_info: dict = {}
        while not (terminated or truncated):
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, _reward, terminated, truncated, last_info = env.step(int(action))

        result = str(last_info.get("result", ""))
        if truncated and not result:
            draws += 1
            agent_draws += 1
            continue
        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1
        else:
            draws += 1

        if result == "1/2-1/2":
            agent_draws += 1
        elif (agent_color == "white" and result == "1-0") or (agent_color == "black" and result == "0-1"):
            agent_wins += 1
        else:
            agent_losses += 1

    return SB3EvalSummary(
        games=max(1, games),
        white_wins=white_wins,
        black_wins=black_wins,
        draws=draws,
        agent_wins=agent_wins,
        agent_losses=agent_losses,
        agent_draws=agent_draws,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    MaskablePPO, DummyVecEnv = _load_sb3()
    env_config = _build_env_config(args)

    def make_env() -> ChessMatchEnv:
        return ChessMatchEnv(env_config)

    train_env = DummyVecEnv([make_env])
    batch_size = max(1, min(args.batch_size, args.n_steps))

    if args.load_path is not None and args.load_path.exists():
        model = MaskablePPO.load(args.load_path, env=train_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=max(1, args.n_steps),
            batch_size=batch_size,
            gamma=args.gamma,
            seed=args.seed,
            verbose=1,
        )

    model.learn(total_timesteps=max(1, args.timesteps))
    model.save(args.save_path)

    summary = evaluate_agent(model, config=env_config, games=args.eval_games, seed=args.seed)
    print(
        f"evaluation games={summary.games} white_wins={summary.white_wins} "
        f"black_wins={summary.black_wins} draws={summary.draws} "
        f"agent_wins={summary.agent_wins} agent_losses={summary.agent_losses} "
        f"agent_draws={summary.agent_draws} agent_score={summary.agent_score:.3f}"
    )
    print(f"saved={args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
