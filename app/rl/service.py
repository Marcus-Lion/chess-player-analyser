from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from uuid import uuid4

from app.rl.config import RLConfig
from app.rl.evaluate import evaluate_matchup
from app.rl.model import ChessRLModel
from app.rl.training import TrainingStep, train_from_self_play


@dataclass(slots=True)
class RLRunStatus:
    job_id: str
    state: str
    preset: str
    total: int
    completed: int = 0
    message: str = "Queued"
    started_at: str = ""
    finished_at: str = ""
    run_id: str = ""
    save_path: str = ""
    samples_path: str = ""
    load_path: str = ""
    episodes: int = 0
    latest_policy_loss: float = 0.0
    latest_value_loss: float = 0.0
    training_history: list[TrainingStep] = field(default_factory=list)
    eval_white_wins: int = 0
    eval_black_wins: int = 0
    eval_draws: int = 0
    eval_games: int = 0
    white_score: float = 0.0
    error: str = ""


class RLRunHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: RLRunStatus | None = None
        self._thread: threading.Thread | None = None

    def get(self) -> RLRunStatus | None:
        with self._lock:
            return self._status

    def _set(self, status: RLRunStatus) -> None:
        with self._lock:
            self._status = status

    def _update(self, **changes) -> None:
        with self._lock:
            if self._status is None:
                return
            for key, value in changes.items():
                setattr(self._status, key, value)

    def start(
        self,
        config: RLConfig,
        *,
        preset: str,
        save_path: str | Path,
        samples_path: str | Path,
        load_path: str | Path | None = None,
    ) -> RLRunStatus:
        current = self.get()
        if current is not None and current.state in {"queued", "running"}:
            return current

        job_id = uuid4().hex
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        status = RLRunStatus(
            job_id=job_id,
            state="queued",
            preset=preset,
            total=max(1, config.episodes),
            message="Queued",
            started_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            save_path=str(save_path),
            samples_path=str(samples_path),
            load_path=str(load_path or ""),
            episodes=0,
        )
        self._set(status)

        def _progress(step: TrainingStep) -> None:
            history = [*status.training_history, step]
            self._update(
                state="running",
                completed=step.episode,
                episodes=step.episode,
                message=f"Training episode {step.episode} of {status.total}",
                latest_policy_loss=step.policy_loss,
                latest_value_loss=step.value_loss,
                training_history=history,
            )

        def _worker() -> None:
            try:
                self._update(state="running", message="Running")
                if load_path is not None and Path(load_path).exists():
                    model = ChessRLModel.load(load_path)
                else:
                    model = ChessRLModel.initialize(
                        seed=config.seed,
                        policy_hidden_dim=config.policy_hidden_dim,
                        value_hidden_dim=config.value_hidden_dim,
                    )
                train_from_self_play(
                    model,
                    config,
                    save_path=save_path,
                    samples_path=samples_path,
                    seed=config.seed,
                    progress_callback=_progress,
                )
                self._update(
                    state="evaluating",
                    message="Evaluating against heuristic engine",
                    completed=status.total,
                    episodes=status.total,
                )
                summary = evaluate_matchup(model, config, games=config.eval_games, seed=config.seed)
                self._update(
                    state="completed",
                    completed=status.total,
                    episodes=status.total,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    message=(
                        f"Completed: white {summary.white_wins}, black {summary.black_wins}, "
                        f"draws {summary.draws}"
                    ),
                    eval_white_wins=summary.white_wins,
                    eval_black_wins=summary.black_wins,
                    eval_draws=summary.draws,
                    eval_games=summary.games,
                    white_score=summary.white_score,
                )
            except Exception as exc:
                self._update(
                    state="failed",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    message="Failed",
                    error=str(exc),
                )

        thread = threading.Thread(target=_worker, daemon=True, name=f"rl-run-{job_id[:8]}")
        self._thread = thread
        thread.start()
        return status


RL_HUB = RLRunHub()
