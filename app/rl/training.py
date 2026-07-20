from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
import json
from pathlib import Path
import random
from collections.abc import Callable

from app.rl.config import RLConfig
from app.rl.model import ChessRLModel, TrainMetrics
from app.rl.replay_buffer import ReplayBuffer
from app.rl.self_play_rl import generate_self_play_batch


@dataclass(slots=True)
class TrainingStep:
    episode: int
    samples: int
    policy_loss: float
    value_loss: float


@dataclass(slots=True)
class TrainingRun:
    steps: list[TrainingStep] = field(default_factory=list)

    @property
    def latest(self) -> TrainingStep | None:
        return self.steps[-1] if self.steps else None


def train_from_self_play(
    model: ChessRLModel,
    config: RLConfig,
    *,
    save_path: str | Path | None = None,
    samples_path: str | Path | None = None,
    seed: int | None = None,
    progress_callback: Callable[[TrainingStep], None] | None = None,
) -> TrainingRun:
    rng = random.Random(seed if seed is not None else config.seed)
    buffer = ReplayBuffer(config.replay_capacity)
    run = TrainingRun()
    samples_file = Path(samples_path) if samples_path is not None else None
    if samples_file is not None:
        samples_file.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config.episodes + 1):
        samples = generate_self_play_batch(model, config, episodes=1, seed=rng.randint(0, 2**31 - 1))
        buffer.add_many(samples)
        if samples_file is not None and samples:
            with samples_file.open("a", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(asdict(sample), ensure_ascii=False))
                    handle.write("\n")
        batch = buffer.sample(config.batch_size, rng=rng)
        metrics = model.train_batch(batch, learning_rate=config.learning_rate) if batch else TrainMetrics()
        run.steps.append(
            TrainingStep(
                episode=episode,
                samples=len(samples),
                policy_loss=metrics.policy_loss,
                value_loss=metrics.value_loss,
            )
        )
        if progress_callback is not None:
            progress_callback(run.steps[-1])
        if save_path is not None and episode % max(1, config.save_every) == 0:
            model.save(save_path)

    if save_path is not None:
        model.save(save_path)
    return run
