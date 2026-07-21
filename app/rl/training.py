from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from app.run_groups import build_run_grouping
from app.rl.config import RLConfig
from app.rl.model import ChessRLModel, TrainMetrics
from app.rl.replay_buffer import ReplayBuffer
from app.rl.self_play_rl import play_self_play_game


@dataclass(slots=True)
class TrainingStep:
    episode: int
    samples: int
    policy_loss: float
    value_loss: float
    result: str = ""
    termination: str = ""
    run_name: str = ""
    run_date: str = ""
    run_group: str = ""


@dataclass(slots=True)
class TrainingRun:
    steps: list[TrainingStep] = field(default_factory=list)

    @property
    def latest(self) -> TrainingStep | None:
        return self.steps[-1] if self.steps else None


def _apply_episode_result(
    *,
    model: ChessRLModel,
    config: RLConfig,
    buffer: ReplayBuffer,
    samples_file: Path | None,
    results_file: Path | None,
    rng: random.Random,
    run: TrainingRun,
    episode_number: int,
    episode_samples,
    episode_result: str,
    episode_termination: str,
    grouping,
    save_path: str | Path | None,
    progress_callback: Callable[[TrainingStep], None] | None,
) -> None:
    buffer.add_many(episode_samples)
    if samples_file is not None and episode_samples:
        with samples_file.open("a", encoding="utf-8") as handle:
            for sample in episode_samples:
                handle.write(json.dumps(asdict(sample), ensure_ascii=False))
                handle.write("\n")
    if results_file is not None:
        payload = {
            "episode": episode_number,
            "game_id": episode_samples[0].game_id if episode_samples else "",
            "result": episode_result,
            "termination": episode_termination,
            "samples": len(episode_samples),
            "run_name": grouping.run_name,
            "run_date": grouping.run_date,
            "run_group": grouping.run_group,
        }
        with results_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
    batch = buffer.sample(config.batch_size, rng=rng)
    metrics = model.train_batch(batch, learning_rate=config.learning_rate) if batch else TrainMetrics()
    step = TrainingStep(
        episode=episode_number,
        samples=len(episode_samples),
        policy_loss=metrics.policy_loss,
        value_loss=metrics.value_loss,
        result=episode_result,
        termination=episode_termination,
        run_name=grouping.run_name,
        run_date=grouping.run_date,
        run_group=grouping.run_group,
    )
    run.steps.append(step)
    if progress_callback is not None:
        progress_callback(step)
    if save_path is not None and episode_number % max(1, config.save_every) == 0:
        model.save(save_path)


def train_from_self_play(
    model: ChessRLModel,
    config: RLConfig,
    *,
    save_path: str | Path | None = None,
    samples_path: str | Path | None = None,
    results_path: str | Path | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    progress_callback: Callable[[TrainingStep], None] | None = None,
) -> TrainingRun:
    rng = random.Random(seed if seed is not None else config.seed)
    buffer = ReplayBuffer(config.replay_capacity)
    run = TrainingRun()
    now = datetime.now(timezone.utc)
    run_started_at = now.isoformat()
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    grouping = build_run_grouping(
        run_name=run_name or config.run_name,
        timestamp=run_started_at,
        default_name="rl",
    )
    run_root = Path("cache") / "rl_runs" / grouping.run_date / grouping.run_slug / run_id
    samples_file = Path(samples_path) if samples_path is not None else None
    results_file = Path(results_path) if results_path is not None else None
    if samples_file is None or results_file is None or save_path is None:
        run_root.mkdir(parents=True, exist_ok=True)
    if samples_file is None:
        samples_file = run_root / "samples.jsonl"
    if results_file is None:
        results_file = run_root / "results.jsonl"
    if results_file is not None:
        results_file.parent.mkdir(parents=True, exist_ok=True)
    if samples_file is not None:
        samples_file.parent.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = run_root / "model.npz"

    worker_count = max(1, int(config.self_play_workers))

    if worker_count == 1:
        for episode in range(1, config.episodes + 1):
            episode_result = play_self_play_game(
                model,
                config,
                seed=rng.randint(0, 2**31 - 1),
                game_id=f"{seed or 'rl'}-{episode}",
            )
            _apply_episode_result(
                model=model,
                config=config,
                buffer=buffer,
                samples_file=samples_file,
                results_file=results_file,
                rng=rng,
                run=run,
                episode_number=episode,
                episode_samples=episode_result.samples,
                episode_result=episode_result.result,
                episode_termination=episode_result.termination,
                grouping=grouping,
                save_path=save_path,
                progress_callback=progress_callback,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(worker_count, config.episodes)) as executor:
            episode_number = 1
            while episode_number <= config.episodes:
                chunk_size = min(worker_count, config.episodes - episode_number + 1)
                futures = [
                    executor.submit(
                        play_self_play_game,
                        model,
                        config,
                        seed=rng.randint(0, 2**31 - 1),
                        game_id=f"{seed or 'rl'}-{episode_number + offset}",
                    )
                    for offset in range(chunk_size)
                ]
                for offset, future in enumerate(futures):
                    episode_result = future.result()
                    _apply_episode_result(
                        model=model,
                        config=config,
                        buffer=buffer,
                        samples_file=samples_file,
                        results_file=results_file,
                        rng=rng,
                        run=run,
                        episode_number=episode_number + offset,
                        episode_samples=episode_result.samples,
                        episode_result=episode_result.result,
                        episode_termination=episode_result.termination,
                        grouping=grouping,
                        save_path=save_path,
                        progress_callback=progress_callback,
                    )
                episode_number += chunk_size

    if save_path is not None:
        model.save(save_path)
    return run
