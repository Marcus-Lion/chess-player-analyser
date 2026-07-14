from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4
import threading
from collections.abc import Callable

import chess
import chess.pgn

import random

from app.games import (
    CENTER_CONTROL_WEIGHT,
    CHECKMATE_WEIGHT,
    FORWARD_SCORE_WEIGHT,
    LEGAL_MOVES_WEIGHT,
    MATERIAL_SCORE_WEIGHT,
    _calculate_center_control,
    _calculate_forward,
    _calculate_material,
    _calculate_total_score,
    _mate_pressure,
    _result_summary,
    choose_engine_move,
)


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SELF_PLAY_RESULTS_PATH = CACHE_DIR / "self_play_results.jsonl"
SELF_PLAY_JOBS_DIR = CACHE_DIR / "self_play_jobs"
SELF_PLAY_JOBS_DIR.mkdir(parents=True, exist_ok=True)
SELF_PLAY_RESULTS_LOCK = threading.Lock()
DEFAULT_SELF_PLAY_WORKERS = max(1, os.cpu_count() or 1)
# Job status lives in memory (see SelfPlayJobHub); only each job's worker log
# file is on disk. Delete a job's status/log once it has been idle for this
# long so neither grows without bound.
JOB_RETENTION_SECONDS = 60


@dataclass
class SelfPlayGame:
    index: int
    result: str
    termination: str
    plies: int
    pgn: str
    final_fen: str
    final_score: float
    outcome: str = ""
    winner: str = ""
    loser: str = ""
    run_id: str = ""
    played_at: str = ""
    seed: int | None = None
    top_k: int = 3
    max_plies: int = 100
    start_fen: str = "startpos"
    white_weights: dict[str, float] | None = None
    black_weights: dict[str, float] | None = None
    duration_seconds: float = 0.0
    evaluations: int = 0
    evaluations_per_move: float = 0.0


@dataclass
class SelfPlayConfig:
    games: int = 3
    max_plies: int = 100
    top_k: int = 3
    # Max parallel worker processes used when running more than one game.
    # If None, defaults to DEFAULT_SELF_PLAY_WORKERS (usually CPU count).
    workers: int | None = None
    seed: int | None = None
    fen: str | None = None
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT
    material_score_weight: float = MATERIAL_SCORE_WEIGHT
    forward_score_weight: float = FORWARD_SCORE_WEIGHT
    center_control_weight: float = CENTER_CONTROL_WEIGHT
    # Shared "goal is checkmate" pressure applied to both sides (not
    # per-player randomized): the objective is the same for everyone.
    checkmate_weight: float = CHECKMATE_WEIGHT
    randomize_player_weights: bool = True
    player_weight_min_multiplier: float = 0.5
    player_weight_max_multiplier: float = 1.5
    # Fixed per-side overrides: when all four are set for a side, that
    # side skips randomization and always uses these exact weights.
    white_legal_moves_weight: float | None = None
    white_material_score_weight: float | None = None
    white_forward_score_weight: float | None = None
    white_center_control_weight: float | None = None
    black_legal_moves_weight: float | None = None
    black_material_score_weight: float | None = None
    black_forward_score_weight: float | None = None
    black_center_control_weight: float | None = None


@dataclass
class SelfPlayJobStatus:
    job_id: str
    state: str
    total: int
    completed: int = 0
    message: str = ""
    played_at: str = ""
    run_id: str = ""
    error: str = ""


def _score_weights(config: SelfPlayConfig) -> tuple[float, float, float, float]:
    return (
        config.legal_moves_weight,
        config.material_score_weight,
        config.forward_score_weight,
        config.center_control_weight,
    )


def _weight_tuple_to_dict(weights: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "legal_moves_weight": weights[0],
        "material_score_weight": weights[1],
        "forward_score_weight": weights[2],
        "center_control_weight": weights[3],
    }


def _randomize_weight_tuple(
    base_weights: tuple[float, float, float, float],
    rng: random.Random,
    *,
    min_multiplier: float,
    max_multiplier: float,
) -> tuple[float, float, float, float]:
    log_min = math.log(min_multiplier)
    log_max = math.log(max_multiplier)
    return tuple(
        base * math.exp(rng.uniform(log_min, log_max))
        for base in base_weights
    )


def _fixed_side_weights(
    config: SelfPlayConfig, side: str
) -> dict[str, float] | None:
    lm = getattr(config, f"{side}_legal_moves_weight")
    mat = getattr(config, f"{side}_material_score_weight")
    fwd = getattr(config, f"{side}_forward_score_weight")
    cc = getattr(config, f"{side}_center_control_weight")
    if lm is None or mat is None or fwd is None or cc is None:
        return None
    return _weight_tuple_to_dict((lm, mat, fwd, cc))


def _player_weight_sets(config: SelfPlayConfig, rng: random.Random) -> tuple[dict[str, float], dict[str, float]]:
    base = _score_weights(config)
    fixed_white = _fixed_side_weights(config, "white")
    fixed_black = _fixed_side_weights(config, "black")

    if fixed_white is not None and fixed_black is not None:
        return fixed_white, fixed_black

    if not config.randomize_player_weights:
        shared = _weight_tuple_to_dict(base)
        return fixed_white or shared, fixed_black or shared.copy()

    white = fixed_white or _weight_tuple_to_dict(
        _randomize_weight_tuple(
            base,
            rng,
            min_multiplier=config.player_weight_min_multiplier,
            max_multiplier=config.player_weight_max_multiplier,
        )
    )
    black = fixed_black or _weight_tuple_to_dict(
        _randomize_weight_tuple(
            base,
            rng,
            min_multiplier=config.player_weight_min_multiplier,
            max_multiplier=config.player_weight_max_multiplier,
        )
    )
    return white, black


def _seed_for_game(config: SelfPlayConfig, game_index: int) -> int | None:
    if config.seed is None:
        return random.SystemRandom().randint(0, 2**31 - 1)
    return config.seed + game_index - 1


def _config_for_game(config: SelfPlayConfig, game_index: int) -> SelfPlayConfig:
    return SelfPlayConfig(
        games=1,
        max_plies=config.max_plies,
        top_k=config.top_k,
        seed=_seed_for_game(config, game_index),
        fen=config.fen,
        legal_moves_weight=config.legal_moves_weight,
        material_score_weight=config.material_score_weight,
        forward_score_weight=config.forward_score_weight,
        center_control_weight=config.center_control_weight,
        checkmate_weight=config.checkmate_weight,
        randomize_player_weights=config.randomize_player_weights,
        player_weight_min_multiplier=config.player_weight_min_multiplier,
        player_weight_max_multiplier=config.player_weight_max_multiplier,
        white_legal_moves_weight=config.white_legal_moves_weight,
        white_material_score_weight=config.white_material_score_weight,
        white_forward_score_weight=config.white_forward_score_weight,
        white_center_control_weight=config.white_center_control_weight,
        black_legal_moves_weight=config.black_legal_moves_weight,
        black_material_score_weight=config.black_material_score_weight,
        black_forward_score_weight=config.black_forward_score_weight,
        black_center_control_weight=config.black_center_control_weight,
    )


def _job_log_path(job_id: str) -> Path:
    return SELF_PLAY_JOBS_DIR / f"{job_id}.log"


def _prune_old_logs(max_age_seconds: int = JOB_RETENTION_SECONDS) -> None:
    """Delete worker log files idle past the retention window.

    Job status itself lives in ``SelfPlayJobHub``'s in-memory dict and is
    pruned there; this only cleans up the stdout/stderr capture files left
    behind by detached worker subprocesses.
    """
    now = time.time()
    try:
        log_files = list(SELF_PLAY_JOBS_DIR.glob("*.log"))
    except OSError:
        return
    for log_path in log_files:
        try:
            age = now - log_path.stat().st_mtime
        except OSError:
            continue
        if age <= max_age_seconds:
            continue
        try:
            log_path.unlink()
        except OSError:
            pass


class SelfPlayJobHub:
    """In-process socket server that receives job-status updates pushed by
    detached self-play worker subprocesses.

    Jobs live in memory only. If the main process restarts, all in-flight job
    status is lost -- a worker still finishes its games and saves results to
    disk independently, but nothing is left to report its progress to. That
    trade-off is intentional: it replaces the old file-based job queue with a
    much simpler live socket connection per worker, at the cost of surviving
    a server crash/restart.
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port: int | None = None
        self._server: socket.socket | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jobs: dict[str, dict] = {}

    def start(self) -> None:
        if self._server is not None:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, 0))
        server.listen(64)
        self._server = server
        self.port = server.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        server = self._server
        if server is None:
            return
        while True:
            try:
                conn, _addr = server.accept()
            except OSError:
                return  # socket closed -> shut down the accept loop
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket) -> None:
        job_id = None
        try:
            with conn, conn.makefile("r", encoding="utf-8") as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    job_id = payload.get("job_id") or job_id
                    if job_id:
                        self._merge(job_id, payload)
        except OSError:
            pass
        finally:
            # A worker that disconnects without ever reporting a terminal
            # state crashed (or was killed) mid-job; surface that instead of
            # leaving the job stuck at "running" forever.
            if job_id:
                self._mark_disconnected(job_id)

    def _merge(self, job_id: str, payload: dict) -> None:
        with self._condition:
            job = self._jobs.setdefault(job_id, {})
            job.update(payload)
            job["_updated_at"] = time.time()
            job["_version"] = job.get("_version", 0) + 1
            self._condition.notify_all()

    def _mark_disconnected(self, job_id: str) -> None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is not None and job.get("state") not in ("completed", "failed"):
                job["state"] = "failed"
                job["error"] = job.get("error") or "Worker disconnected unexpectedly"
                job["message"] = "Failed"
                job["_updated_at"] = time.time()
                job["_version"] = job.get("_version", 0) + 1
                self._condition.notify_all()

    def send(self, status: "SelfPlayJobStatus") -> None:
        """Record a status update from within the main process itself."""
        self._merge(status.job_id, asdict(status))

    @staticmethod
    def _strip(job: dict) -> dict:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._strip(job) if job is not None else None

    def get_job_version(self, job_id: str) -> tuple[dict | None, int]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None, 0
            return self._strip(job), job.get("_version", 0)

    def wait_for_update(
        self, job_id: str, since_version: int, timeout: float = 30.0
    ) -> tuple[dict | None, int]:
        """Block (in a worker thread, not the event loop) until ``job_id``'s
        version moves past ``since_version``, or ``timeout`` elapses. Lets a
        websocket push fresh status the instant a worker reports it, instead
        of the browser polling on a fixed interval."""
        with self._condition:
            deadline = time.monotonic() + timeout
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    return None, 0
                version = job.get("_version", 0)
                if version != since_version:
                    return self._strip(job), version
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._strip(job), since_version
                self._condition.wait(remaining)

    def prune(self, max_age_seconds: int = JOB_RETENTION_SECONDS) -> None:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            stale = [
                jid
                for jid, job in self._jobs.items()
                if job.get("state") in ("completed", "failed") and job.get("_updated_at", 0) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]


_job_hub: SelfPlayJobHub | None = None
_job_hub_lock = threading.Lock()


def get_job_hub() -> SelfPlayJobHub:
    global _job_hub
    if _job_hub is None:
        with _job_hub_lock:
            if _job_hub is None:
                hub = SelfPlayJobHub()
                hub.start()
                _job_hub = hub
    return _job_hub


class SelfPlayJobClient:
    """Socket client used by a detached worker subprocess to report status
    back to the main process's ``SelfPlayJobHub``.

    Never raises: if the main process is gone or unreachable, updates are
    silently dropped rather than crashing the worker mid-game -- losing job
    status on a crash is an accepted trade-off of this design.
    """

    def __init__(self, host: str, port: int) -> None:
        try:
            self._sock: socket.socket | None = socket.create_connection((host, port), timeout=10)
        except OSError:
            self._sock = None

    def send(self, status: "SelfPlayJobStatus") -> None:
        if self._sock is None:
            return
        line = json.dumps(asdict(status), ensure_ascii=False) + "\n"
        try:
            self._sock.sendall(line.encode("utf-8"))
        except OSError:
            self._sock = None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


def prune_old_jobs(max_age_seconds: int = JOB_RETENTION_SECONDS) -> None:
    get_job_hub().prune(max_age_seconds)
    _prune_old_logs(max_age_seconds)


def load_self_play_job(job_id: str) -> dict | None:
    return get_job_hub().get_job(job_id)


def _evaluate_board(board: chess.Board, config: SelfPlayConfig | None = None) -> float:
    legal_moves = len(list(board.legal_moves))
    f1, f2 = _calculate_forward(board)
    material = _calculate_material(board)
    center = _calculate_center_control(board)
    forward_score = (f1["White"] + f2["White"]) - (f1["Black"] + f2["Black"])
    material_score = material["White"] - material["Black"]
    center_score = center["White"] - center["Black"]
    legal_moves_weight, material_score_weight, forward_score_weight, center_control_weight = _score_weights(
        config or SelfPlayConfig()
    )
    return _calculate_total_score(
        legal_moves,
        material_score,
        forward_score,
        center_score,
        legal_moves_weight=legal_moves_weight,
        material_score_weight=material_score_weight,
        forward_score_weight=forward_score_weight,
        center_control_weight=center_control_weight,
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _result_target(board: chess.Board, result: str) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if board.turn == chess.BLACK else 0.0
    return 0.5


def _extract_samples_from_pgn(pgn_text: str) -> list[tuple[int, int, int, float, int, float]]:
    game = chess.pgn.read_game(StringIO(pgn_text))
    if game is None:
        return []

    result = game.headers.get("Result", "*")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return []

    board = game.board()
    node = game
    samples: list[tuple[int, int, int, float, int, float]] = []

    while node.variations:
        f1, f2 = _calculate_forward(board)
        material = _calculate_material(board)
        forward_score = (f1["White"] + f2["White"]) - (f1["Black"] + f2["Black"])
        material_score = material["White"] - material["Black"]
        samples.append((
            len(list(board.legal_moves)),
            material_score,
            forward_score,
            _mate_pressure(board),
            1 if board.turn == chess.WHITE else -1,
            _result_target(board, result),
        ))
        node = node.variation(0)
        board.push(node.move)

    return samples


def _log_loss(probability: float, target: float) -> float:
    clipped = min(max(probability, 1e-9), 1.0 - 1e-9)
    return -(target * math.log(clipped) + (1.0 - target) * math.log(1.0 - clipped))


def _evaluate_candidate(
    samples: list[tuple[int, int, int, float, int, float]],
    weights: tuple[float, float, float, float, float],
    *,
    temperature: float,
) -> float:
    if not samples:
        return float("inf")

    lm_w, mat_w, fwd_w, cc_w, mate_w = weights
    total_loss = 0.0
    for legal_moves, material_score, forward_score, mate_pressure, side_sign, target in samples:
        # Note: tuning samples don't have center_score; using 0 for simplicity during tuning
        score = _calculate_total_score(
            legal_moves,
            material_score,
            forward_score,
            center_score=0,
            legal_moves_weight=lm_w,
            material_score_weight=mat_w,
            forward_score_weight=fwd_w,
            center_control_weight=cc_w,
        ) + mate_w * mate_pressure
        utility = score * side_sign
        probability = _sigmoid(utility / max(temperature, 1e-6))
        total_loss += _log_loss(probability, target)
    return total_loss / len(samples)


def tune_score_weights(
    corpus: list[dict],
    *,
    iterations: int = 100,
    seed: int | None = None,
    temperature: float = 8.0,
    min_multiplier: float = 0.25,
    max_multiplier: float = 4.0,
) -> dict:
    rng = random.Random(seed)
    samples: list[tuple[int, int, int, float, int, float]] = []
    for row in corpus:
        samples.extend(_extract_samples_from_pgn(row.get("pgn", "")))

    if not samples:
        raise ValueError("No labeled positions available for tuning")

    rng.shuffle(samples)
    split = max(1, int(len(samples) * 0.8))
    train_samples = samples[:split]
    validation_samples = samples[split:] or samples[:]

    base_weights = (LEGAL_MOVES_WEIGHT, MATERIAL_SCORE_WEIGHT, FORWARD_SCORE_WEIGHT, CENTER_CONTROL_WEIGHT, CHECKMATE_WEIGHT)
    best_weights = base_weights
    best_loss = _evaluate_candidate(validation_samples, base_weights, temperature=temperature)
    history: list[dict[str, float]] = [
        {
            "legal_moves_weight": base_weights[0],
            "material_score_weight": base_weights[1],
            "forward_score_weight": base_weights[2],
            "center_control_weight": base_weights[3],
            "checkmate_weight": base_weights[4],
            "validation_loss": best_loss,
        }
    ]

    log_min = math.log(min_multiplier)
    log_max = math.log(max_multiplier)

    for _ in range(max(1, iterations)):
        candidate = tuple(
            base * math.exp(rng.uniform(log_min, log_max))
            for base in base_weights
        )
        loss = _evaluate_candidate(train_samples, candidate, temperature=temperature)
        validation_loss = _evaluate_candidate(validation_samples, candidate, temperature=temperature)
        history.append({
            "legal_moves_weight": candidate[0],
            "material_score_weight": candidate[1],
            "forward_score_weight": candidate[2],
            "center_control_weight": candidate[3],
            "checkmate_weight": candidate[4],
            "training_loss": loss,
            "validation_loss": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = candidate

    return {
        "best_weights": {
            "legal_moves_weight": best_weights[0],
            "material_score_weight": best_weights[1],
            "forward_score_weight": best_weights[2],
            "center_control_weight": best_weights[3],
            "checkmate_weight": best_weights[4],
        },
        "best_validation_loss": best_loss,
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "history": history,
    }


def _terminal_reason(board: chess.Board) -> tuple[str, str]:
    if board.is_checkmate():
        return ("1-0" if board.turn == chess.BLACK else "0-1", "checkmate")
    if board.is_stalemate():
        return ("1/2-1/2", "stalemate")
    if board.is_insufficient_material():
        return ("1/2-1/2", "insufficient material")
    if board.is_fivefold_repetition():
        return ("1/2-1/2", "fivefold repetition")
    if board.is_seventyfive_moves():
        return ("1/2-1/2", "75-move rule")
    if board.can_claim_threefold_repetition():
        return ("1/2-1/2", "threefold repetition")
    if board.can_claim_fifty_moves():
        return ("1/2-1/2", "fifty-move rule")
    return ("", "")


def play_self_game(config: SelfPlayConfig, game_index: int, rng: random.Random | None = None) -> SelfPlayGame:
    rng = rng or random.Random(config.seed)
    board = chess.Board(config.fen) if config.fen else chess.Board()
    white_weights, black_weights = _player_weight_sets(config, rng)
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-play harness"
    game.headers["Site"] = "Local"
    game.headers["Round"] = str(game_index)
    game.headers["White"] = "Heuristic"
    game.headers["Black"] = "Heuristic"
    game.headers["WhiteWeights"] = json.dumps(white_weights, sort_keys=True)
    game.headers["BlackWeights"] = json.dumps(black_weights, sort_keys=True)

    node = game
    plies = 0
    eval_counter = [0]

    start_time = time.perf_counter()
    result, termination = _terminal_reason(board)
    while plies < config.max_plies and not result:
        active_weights = white_weights if board.turn == chess.WHITE else black_weights
        move, _ = choose_engine_move(
            board,
            rng,
            config.top_k,
            legal_moves_weight=active_weights["legal_moves_weight"],
            material_score_weight=active_weights["material_score_weight"],
            forward_score_weight=active_weights["forward_score_weight"],
            center_control_weight=active_weights["center_control_weight"],
            checkmate_weight=config.checkmate_weight,
            eval_counter=eval_counter,
        )
        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)
        node.comment = san
        plies += 1
        result, termination = _terminal_reason(board)
    duration_seconds = time.perf_counter() - start_time

    if not result:
        result = "1/2-1/2"
        termination = "max turns reached"

    game.headers["Result"] = result
    game.headers["Termination"] = termination

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    pgn_text = game.accept(exporter)
    final_score = _evaluate_board(board)
    summary = _result_summary(result, white="Heuristic", black="Heuristic")
    evaluations = eval_counter[0]

    return SelfPlayGame(
        index=game_index,
        result=result,
        termination=termination,
        plies=plies,
        pgn=pgn_text,
        final_fen=board.fen(),
        final_score=final_score,
        outcome=summary["status"],
        winner=summary["winner"],
        loser=summary["loser"],
        white_weights=white_weights,
        black_weights=black_weights,
        duration_seconds=duration_seconds,
        evaluations=evaluations,
        evaluations_per_move=(evaluations / plies) if plies else 0.0,
    )


def run_self_play(
    config: SelfPlayConfig,
    *,
    progress_callback: Callable[[int, SelfPlayGame, list[SelfPlayGame]], None] | None = None,
    run_id: str | None = None,
) -> list[SelfPlayGame]:
    played_at = datetime.now(timezone.utc).isoformat()
    run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    start_fen = config.fen or "startpos"
    games: list[SelfPlayGame] = []

    if config.games <= 1:
        game_config = _config_for_game(config, 1)
        game = play_self_game(game_config, 1)
        game.run_id = run_id
        game.played_at = played_at
        game.seed = game_config.seed
        game.top_k = config.top_k
        game.max_plies = config.max_plies
        game.start_fen = start_fen
        games.append(game)
        if progress_callback is not None:
            progress_callback(1, game, games)
        return games

    requested_workers = config.workers or DEFAULT_SELF_PLAY_WORKERS
    max_workers = max(1, min(int(requested_workers), config.games))
    future_to_index: dict = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(1, config.games + 1):
            game_config = _config_for_game(config, i)
            future = executor.submit(play_self_game, game_config, i)
            future_to_index[future] = i

        completed_games: dict[int, SelfPlayGame] = {}
        completed_count = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            game = future.result()
            game.run_id = run_id
            game.played_at = played_at
            game.seed = _seed_for_game(config, index)
            game.top_k = config.top_k
            game.max_plies = config.max_plies
            game.start_fen = start_fen
            completed_games[index] = game
            completed_count += 1
            ordered_games = [completed_games[i] for i in sorted(completed_games)]
            if progress_callback is not None:
                progress_callback(completed_count, game, ordered_games)

    return [completed_games[i] for i in sorted(completed_games)]


def save_self_play_results(games: list[SelfPlayGame]) -> None:
    if not games:
        return

    with SELF_PLAY_RESULTS_LOCK:
        with SELF_PLAY_RESULTS_PATH.open("a", encoding="utf-8") as handle:
            for game in games:
                payload = {
                    "played_at": game.played_at or datetime.now(timezone.utc).isoformat(),
                    "run_id": game.run_id,
                    "index": game.index,
                    "seed": game.seed,
                    "top_k": game.top_k,
                    "max_plies": game.max_plies,
                    "start_fen": game.start_fen,
                    "result": game.result,
                    "termination": game.termination,
                    "plies": game.plies,
                    "final_fen": game.final_fen,
                    "final_score": game.final_score,
                    "outcome": game.outcome,
                    "winner": game.winner,
                    "loser": game.loser,
                    "white_weights": game.white_weights,
                    "black_weights": game.black_weights,
                    "duration_seconds": game.duration_seconds,
                    "evaluations": game.evaluations,
                    "evaluations_per_move": game.evaluations_per_move,
                    "pgn": game.pgn,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_self_play_results(limit: int | None = 50) -> list[dict]:
    if not SELF_PLAY_RESULTS_PATH.exists():
        return []

    rows: list[dict] = []
    with SELF_PLAY_RESULTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_normalize_result(json.loads(line)))
            except json.JSONDecodeError:
                continue

    return rows if limit is None else rows[-limit:]


def load_self_play_result(run_id: str, index: int) -> dict | None:
    if not SELF_PLAY_RESULTS_PATH.exists():
        return None

    with SELF_PLAY_RESULTS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") == run_id and int(row.get("index", 0)) == index:
                return _normalize_result(row)
    return None


def _normalize_result(row: dict) -> dict:
    if "forward_1" not in row and "control_1" in row:
        row["forward_1"] = row.pop("control_1")
    if "forward_2" not in row and "control_2" in row:
        row["forward_2"] = row.pop("control_2")
    if "forward_score" not in row and "control_score" in row:
        row["forward_score"] = row.pop("control_score")
    row.setdefault("duration_seconds", 0.0)
    row.setdefault("evaluations", 0)
    row.setdefault("evaluations_per_move", 0.0)
    row["played_at_display"] = _format_played_at(row.get("played_at", ""))
    return row


def _format_played_at(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value.split(".", 1)[0].replace("T", " ")

    local_tz = datetime.now().astimezone().tzinfo
    if local_tz is not None:
        parsed = parsed.astimezone(local_tz)
    return parsed.replace(tzinfo=None, microsecond=0).isoformat(sep=" ")


def load_tuning_corpus(limit: int = 50) -> list[dict]:
    corpus = load_self_play_results(limit=limit)
    if corpus:
        return corpus

    bootstrap_config = SelfPlayConfig(games=max(1, min(5, limit)))
    bootstrap_games = run_self_play(bootstrap_config)
    save_self_play_results(bootstrap_games)
    return load_self_play_results(limit=limit)


def _run_self_play_job(job_id: str, run_id: str, config_data: dict, reporter: "SelfPlayJobClient") -> None:
    config = SelfPlayConfig(**config_data)
    status = SelfPlayJobStatus(
        job_id=job_id,
        state="running",
        total=max(1, config.games),
        completed=0,
        message="Running",
        played_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )

    status_lock = threading.Lock()
    stop_heartbeat = threading.Event()

    def _send_status() -> None:
        with status_lock:
            reporter.send(status)

    def _heartbeat_loop(interval_seconds: float = 5.0) -> None:
        # The browser has a "no progress" watchdog to avoid trapping the UI
        # behind a dead/orphaned job. Long games can legitimately go a while
        # between completions, so send periodic "still running" updates.
        while not stop_heartbeat.wait(interval_seconds):
            with status_lock:
                if status.state not in ("running", "queued"):
                    continue
                if status.completed:
                    status.message = f"Completed {status.completed} of {status.total}"
                else:
                    status.message = status.message or "Running"
                reporter.send(status)

    _send_status()
    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    try:
        def progress_callback(completed: int, game: SelfPlayGame, games: list[SelfPlayGame]) -> None:
            save_self_play_results([game])
            with status_lock:
                status.completed = completed
                status.message = f"Completed {completed} of {status.total}"
                status.run_id = game.run_id or run_id
                reporter.send(status)

        games = run_self_play(config, progress_callback=progress_callback, run_id=run_id)
        if not config.games:
            save_self_play_results(games)
        with status_lock:
            status.state = "completed"
            status.completed = status.total
            status.message = "Completed"
            status.run_id = run_id
            reporter.send(status)
    except Exception as exc:
        with status_lock:
            status.state = "failed"
            status.error = str(exc)
            status.message = "Failed"
            reporter.send(status)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)


def start_self_play_job(config: SelfPlayConfig) -> dict:
    hub = get_job_hub()
    prune_old_jobs()
    job_id = uuid4().hex
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    status = SelfPlayJobStatus(
        job_id=job_id,
        state="queued",
        total=max(1, config.games),
        message="Queued",
        played_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )
    hub.send(status)

    # Launch by file path rather than ``-m app.self_play_worker``. The ``-m``
    # form resolves the module against sys.path *before any code runs*, which
    # fails under a debugger (PyCharm/pydevd) that rewrites the launch and drops
    # the project root -> "No module named app.self_play_worker". A file-path
    # launch has no module-resolution step; the worker fixes sys.path itself.
    worker_path = Path(__file__).resolve().parent / "self_play_worker.py"
    cmd = [
        sys.executable,
        str(worker_path),
        "--job-id",
        job_id,
        "--run-id",
        run_id,
        "--host",
        hub.host,
        "--port",
        str(hub.port),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )

    # Ensure the worker can import the ``app`` package regardless of how the
    # subprocess is launched. ``python -m`` normally relies on the current
    # working directory being on sys.path, but that breaks when a debugger
    # (e.g. PyCharm/pydevd auto-attaching to subprocesses) rewrites the launch
    # machinery, yielding "No module named app.self_play_worker". Putting the
    # project root on PYTHONPATH makes the import robust in every environment.
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(BASE_DIR) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )

    # Capture the worker's stdout/stderr to a per-job log file instead of
    # discarding it. If the detached worker fails to start (e.g. the server's
    # interpreter can't import the app package), the traceback lands here
    # instead of vanishing and leaving the job stuck at "queued". The job's
    # own progress/status now travels over the socket connection back to
    # ``hub``, not through this file.
    log_path = _job_log_path(job_id)
    log_handle = open(log_path, "w", encoding="utf-8")
    try:
        log_handle.write(f"launching worker: {cmd}\ncwd={BASE_DIR}\nexecutable={sys.executable}\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            creationflags=creationflags,
            close_fds=True,
        )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(asdict(config)).encode("utf-8"))
        proc.stdin.close()
    finally:
        # The child has inherited its own copy of the handle; the parent no
        # longer needs it.
        log_handle.close()
    return asdict(status)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the position scorer against itself.")
    parser.add_argument("--games", type=int, default=3, help="Number of self-play games to run.")
    parser.add_argument("--max-plies", type=int, default=100, help="Stop each game after this many plies.")
    parser.add_argument("--top-k", type=int, default=3, help="Randomly choose among the top K evaluated moves.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel worker processes for multi-game self-play (default: CPU count).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for move selection.")
    parser.add_argument("--fen", type=str, default=None, help="Optional starting FEN.")
    parser.add_argument("--legal-moves-weight", type=float, default=LEGAL_MOVES_WEIGHT, help="Weight for legal move count.")
    parser.add_argument("--material-score-weight", type=float, default=MATERIAL_SCORE_WEIGHT, help="Weight for material balance.")
    parser.add_argument("--forward-score-weight", type=float, default=FORWARD_SCORE_WEIGHT, help="Weight for forward control.")
    parser.add_argument("--center-control-weight", type=float, default=CENTER_CONTROL_WEIGHT, help="Weight for center control.")
    parser.add_argument("--checkmate-weight", type=float, default=CHECKMATE_WEIGHT, help="Weight for the mate-pressure heuristic (drive the enemy king toward checkmate).")
    parser.add_argument("--fixed-player-weights", action="store_true", help="Use the same weights for both sides.")
    parser.add_argument("--player-weight-min-multiplier", type=float, default=0.5, help="Lower bound for per-player randomization.")
    parser.add_argument("--player-weight-max-multiplier", type=float, default=1.5, help="Upper bound for per-player randomization.")
    parser.add_argument("--tune-weights", action="store_true", help="Search for better score weights before playing.")
    parser.add_argument("--tune-iterations", type=int, default=100, help="Number of random weight candidates to test.")
    parser.add_argument("--tune-corpus-size", type=int, default=50, help="How many recent self-play games to use as tuning data.")
    parser.add_argument("--tune-temperature", type=float, default=8.0, help="Temperature used when turning scores into probabilities.")
    parser.add_argument("--tune-output", type=Path, default=None, help="Optional JSON file to write the tuning result.")
    parser.add_argument("--output", type=Path, default=None, help="Optional file to write PGN output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = SelfPlayConfig(
        games=max(1, args.games),
        max_plies=max(1, args.max_plies),
        top_k=max(1, args.top_k),
        workers=(max(1, int(args.workers)) if args.workers else None),
        seed=args.seed,
        fen=args.fen,
        legal_moves_weight=args.legal_moves_weight,
        material_score_weight=args.material_score_weight,
        forward_score_weight=args.forward_score_weight,
        center_control_weight=args.center_control_weight,
        checkmate_weight=args.checkmate_weight,
        randomize_player_weights=not args.fixed_player_weights,
        player_weight_min_multiplier=args.player_weight_min_multiplier,
        player_weight_max_multiplier=args.player_weight_max_multiplier,
    )

    if args.tune_weights:
        corpus = load_tuning_corpus(limit=max(1, args.tune_corpus_size))
        tuning = tune_score_weights(
            corpus,
            iterations=max(1, args.tune_iterations),
            seed=args.seed,
            temperature=max(0.001, args.tune_temperature),
        )
        best = tuning["best_weights"]
        config.legal_moves_weight = best["legal_moves_weight"]
        config.material_score_weight = best["material_score_weight"]
        config.forward_score_weight = best["forward_score_weight"]
        config.center_control_weight = best["center_control_weight"]
        config.checkmate_weight = best["checkmate_weight"]
        print(
            "Best weights: "
            f"legal_moves={config.legal_moves_weight:.6f}, "
            f"material={config.material_score_weight:.6f}, "
            f"forward={config.forward_score_weight:.6f}, "
            f"center={config.center_control_weight:.6f}, "
            f"checkmate={config.checkmate_weight:.6f}"
        )
        print(f"Validation loss: {tuning['best_validation_loss']:.6f}")
        if args.tune_output:
            args.tune_output.write_text(json.dumps(tuning, indent=2), encoding="utf-8")

    games = run_self_play(config)
    save_self_play_results(games)

    if args.output:
        args.output.write_text("\n\n".join(game.pgn for game in games) + "\n", encoding="utf-8")

    for game in games:
        white = game.white_weights or {}
        black = game.black_weights or {}
        print(
            f"Game {game.index}: {game.result} after {game.plies} plies ({game.termination}); "
            f"final score {game.final_score}; "
            f"took {game.duration_seconds:.2f}s, {game.evaluations_per_move:.0f} evals/move; "
            f"W[lm={white.get('legal_moves_weight', 0):.6f}, mat={white.get('material_score_weight', 0):.6f}, fwd={white.get('forward_score_weight', 0):.6f}] "
            f"B[lm={black.get('legal_moves_weight', 0):.6f}, mat={black.get('material_score_weight', 0):.6f}, fwd={black.get('forward_score_weight', 0):.6f}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
