from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from io import StringIO
import socket
import subprocess
import sys
import time
import signal
from pathlib import Path
from uuid import uuid4
import threading
from collections.abc import Callable


# Keep Windows Job Object handles alive for as long as the parent process is
# alive.  When the parent exits, Windows closes these handles and terminates
# every process in the job, including ProcessPoolExecutor descendants.
_WINDOWS_WORKER_JOBS: list[int] = []


def _put_process_in_parent_lifetime_job(proc: subprocess.Popen[Any]) -> None:
    """Make ``proc`` and its descendants die when this process dies (Windows).

    A detached subprocess has no parent-lifetime semantics on Windows.  A Job
    Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` supplies those semantics
    and also covers grandchildren created by the worker's process pool.
    """
    if os.name != "nt":
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION = 9;
    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000.
    info = _JobExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = 0x2000
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    # PROCESS_SET_QUOTA | PROCESS_TERMINATE are required by
    # AssignProcessToJobObject.
    process_handle = kernel32.OpenProcess(0x0100 | 0x0001, False, proc.pid)
    if not process_handle:
        kernel32.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel32.SetInformationJobObject(
            job,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(job, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:
        kernel32.CloseHandle(job)
        raise
    finally:
        kernel32.CloseHandle(process_handle)

    _WINDOWS_WORKER_JOBS.append(int(job))

import chess
import chess.pgn

import random

import dotenv

dotenv.load_dotenv()

from app.games import (
    CENTER_CONTROL_WEIGHT,
    CHECKMATE_WEIGHT,
    FORWARD_SCORE_WEIGHT,
    FORWARD_MATERIAL_SCORE_WEIGHT,
    LEGAL_MOVES_WEIGHT,
    MATERIAL_SCORE_WEIGHT,
    MAX_AUTO_SEARCH_DEPTH,
    _auto_search_depth,
    _calculate_center_control,
    _calculate_forward,
    _calculate_material,
    _calculate_total_score,
    _mate_pressure,
    _result_summary,
    choose_engine_move,
    play_self_game_native,
)
from app.run_groups import build_run_grouping
from app.players import PlayerProfile, get_player_roster, pick_two_players
from app.neo4j_store import Neo4jStore
from app.self_play_metrics import (
    SHAP_BALANCE_LEARNING_RATE,
    player_overview,
    shap_balance_player_weights,
    to_dataframe as self_play_to_dataframe,
)

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SELF_PLAY_JOBS_DIR = CACHE_DIR / "self_play_jobs"
SELF_PLAY_JOBS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SELF_PLAY_WORKERS = min(max(1, os.process_cpu_count() or 1), 48)
# Job status lives in memory (see SelfPlayJobHub); only each job's worker log
# file is on disk. Delete a job's status/log once it has been idle for this
# long so neither grows without bound.
JOB_RETENTION_SECONDS = 60*60 # 1hr
# Five complete 16-player double round-robins: 5 * 240 games.
SELF_PLAY_REBALANCE_BATCH_SIZE = max(1, _env_int("SELF_PLAY_REBALANCE_BATCH_SIZE", 1_200))
# Completed games are persisted immediately. Elo is deliberately recalculated
# less often because each refresh reads the complete saved-game history.
SELF_PLAY_ELO_BATCH_SIZE = max(1, _env_int("SELF_PLAY_ELO_BATCH_SIZE", 32))
SELF_PLAY_ELITE_COUNT = 1 # Players with the highest win rate
SELF_PLAY_ELITE_MUTATION_STDDEV = _env_float("SELF_PLAY_ELITE_MUTATION_STDDEV", 0.10)


@dataclass
class SelfPlayGame:
    index: int
    result: str
    termination: str
    turns: int
    pgn: str
    final_fen: str
    final_score: float
    termination_display: str = ""
    outcome: str = ""
    winner: str = ""
    loser: str = ""
    run_id: str = ""
    run_name: str = ""
    run_date: str = ""
    run_group: str = ""
    played_at: str = ""
    seed: int | None = None
    top_k: int = 1
    top_k_score_threshold: float | None = 3.0
    # Probability that a move is selected from the complete legal move set,
    # allowing intentional blunders. Zero keeps normal Top-K selection.
    blunder_control: float = 0.0
    max_turns: int = 100
    start_fen: str = "startpos"
    white_weights: dict[str, float] | None = None
    black_weights: dict[str, float] | None = None
    white_player_id: str | None = None
    white_player_name: str | None = None
    white_player_description: str | None = None
    black_player_id: str | None = None
    black_player_name: str | None = None
    black_player_description: str | None = None
    duration_seconds: float = 0.0
    evaluations: int = 0
    evaluations_per_move: float = 0.0
    turn_durations_seconds: list[float] | None = None


@dataclass
class SelfPlayConfig:
    games: int = 3
    max_turns: int = 100
    top_k: int = 1
    # Optional maximum score loss from the best move for random Top-K choice.
    # Defaults to 3.0; None allows every candidate up to the Top-K count.
    top_k_score_threshold: float | None = 3.0
    # Probability that a move is selected from the complete legal move set.
    blunder_control: float = 0.0
    # Negamax search depth. None (the default) auto-derives depth per move
    # from remaining material via ``_auto_search_depth`` -- shallow while the
    # board is full, deeper once material has thinned out. Set an explicit
    # depth (e.g. 1 or 2) to pin it for the whole game instead.
    depth: int | None = None
    # Upper bound for automatically derived search depth. Ignored when a
    # fixed ``depth`` is configured.
    max_depth: int = MAX_AUTO_SEARCH_DEPTH
    # Max parallel worker processes used when running more than one game.
    # If None, defaults to DEFAULT_SELF_PLAY_WORKERS (usually CPU count).
    workers: int | None = None
    seed: int | None = None
    run_name: str | None = None
    fen: str | None = None
    legal_moves_weight: float = LEGAL_MOVES_WEIGHT
    material_score_weight: float = MATERIAL_SCORE_WEIGHT
    forward_score_weight: float = FORWARD_SCORE_WEIGHT
    forward_material_score_weight: float = FORWARD_MATERIAL_SCORE_WEIGHT
    center_control_weight: float = CENTER_CONTROL_WEIGHT
    # Shared "goal is checkmate" pressure applied to both sides (not
    # per-player randomized): the objective is the same for everyone.
    checkmate_weight: float = CHECKMATE_WEIGHT
    randomize_player_weights: bool = True
    player_weight_min: float = -4.0
    player_weight_max: float = 4.0
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
    mirror_colors: bool = False


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


def _result_counts(games: list[SelfPlayGame]) -> Counter[str]:
    return Counter(game.result for game in games)


def _print_result_summary(label: str, games: list[SelfPlayGame]) -> None:
    if not games:
        print(f"{label}: no games")
        return

    counts = _result_counts(games)
    total = len(games)
    white = counts.get("1-0", 0)
    black = counts.get("0-1", 0)
    draws = counts.get("1/2-1/2", 0)
    other = total - white - black - draws
    summary = (
        f"{label}: {total} games | "
        f"White wins {white} ({white / total:.1%}), "
        f"Black wins {black} ({black / total:.1%}), "
        f"Draws {draws} ({draws / total:.1%})"
    )
    if other:
        summary += f", Other {other} ({other / total:.1%})"
    print(summary)


def _print_batch_summary(batch_number: int, games: list[SelfPlayGame], *, batch_size: int) -> None:
    if not games:
        return

    start = (batch_number - 1) * batch_size + 1
    end = start + len(games) - 1
    print(f"Rebalance batch {batch_number} ({start}-{end}):")
    _print_result_summary("  batch results", games)


def _saved_self_play_game(row: dict) -> SelfPlayGame:
    """Rehydrate a saved result for a rebalance batch spanning multiple runs."""
    values = {
        field.name: row[field.name]
        for field in fields(SelfPlayGame)
        if field.name in row
    }
    values.setdefault("index", 0)
    values.setdefault("result", "*")
    values.setdefault("termination", "")
    values.setdefault("turns", 0)
    values.setdefault("pgn", "")
    values.setdefault("final_fen", "")
    values.setdefault("final_score", 0.0)
    return SelfPlayGame(**values)


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


def _player_profile_from_row(base: PlayerProfile, row: dict) -> PlayerProfile:
    return PlayerProfile(
        player_id=str(row.get("player_id", base.player_id)),
        name=str(row.get("name", base.name)),
        description=str(row.get("description", base.description)),
        legal_moves_weight=float(row.get("legal_moves_weight", base.legal_moves_weight)),
        material_score_weight=float(row.get("material_score_weight", base.material_score_weight)),
        forward_score_weight=float(row.get("forward_score_weight", base.forward_score_weight)),
        center_control_weight=float(row.get("center_control_weight", base.center_control_weight)),
    )


def load_current_player_roster() -> list[PlayerProfile]:
    """Load the latest self-play player weights from Neo4j, falling back to code defaults."""
    base_roster = {player.player_id: player for player in get_player_roster()}
    try:
        with Neo4jStore() as store:
            rows = store.load_self_play_players()
        for row in rows:
            player_id = str(row.get("player_id", ""))
            base = base_roster.get(player_id)
            if base is None:
                continue
            base_roster[player_id] = _player_profile_from_row(base, row)
    except Exception:
        pass
    return list(base_roster.values())


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


def _player_weight_sets(
    config: SelfPlayConfig, rng: random.Random
) -> tuple[
    PlayerProfile | None,
    dict[str, float],
    PlayerProfile | None,
    dict[str, float],
]:
    base = _score_weights(config)
    fixed_white = _fixed_side_weights(config, "white")
    fixed_black = _fixed_side_weights(config, "black")

    if fixed_white is not None and fixed_black is not None:
        return None, fixed_white, None, fixed_black

    if not config.randomize_player_weights:
        shared = _weight_tuple_to_dict(base)
        return None, fixed_white or shared, None, fixed_black or shared.copy()

    roster = load_current_player_roster()
    white_player, black_player = pick_two_players(rng, _current_player_skill_levels(), roster=roster)
    white = fixed_white or white_player.weights
    black = fixed_black or black_player.weights
    return (
        None if fixed_white is not None else white_player,
        white,
        None if fixed_black is not None else black_player,
        black,
    )


def _current_player_skill_levels() -> dict[str, float]:
    """Return the latest per-player Elo estimates from the player nodes."""
    try:
        with Neo4jStore() as store:
            rows = store.load_self_play_players()
        return {
            str(row.get("player_id")): float(row.get("elo", _self_play_elo_baseline()))
            for row in rows
            if row.get("player_id")
        }
    except Exception:
        return {}


def _seed_for_game(config: SelfPlayConfig, game_index: int) -> int | None:
    if config.seed is None:
        return random.SystemRandom().randint(0, 2**31 - 1)
    return config.seed + game_index - 1


def _paired_game_count(requested_games: int) -> int:
    # Keep the requested count exact. Even runs still use mirrored pairs;
    # an odd run gets one additional unmirrored game.
    return max(1, requested_games)


def _config_for_game(
    config: SelfPlayConfig,
    game_index: int,
    *,
    seed: int | None = None,
    mirror_colors: bool = False,
) -> SelfPlayConfig:
    return SelfPlayConfig(
        games=1,
        max_turns=config.max_turns,
        top_k=config.top_k,
        top_k_score_threshold=config.top_k_score_threshold,
        forward_material_score_weight=config.forward_material_score_weight,
        blunder_control=config.blunder_control,
        depth=config.depth,
        max_depth=config.max_depth,
        seed=_seed_for_game(config, game_index) if seed is None else seed,
        run_name=config.run_name,
        fen=config.fen,
        legal_moves_weight=config.legal_moves_weight,
        material_score_weight=config.material_score_weight,
        forward_score_weight=config.forward_score_weight,
        center_control_weight=config.center_control_weight,
        checkmate_weight=config.checkmate_weight,
        randomize_player_weights=config.randomize_player_weights,
        player_weight_min=config.player_weight_min,
        player_weight_max=config.player_weight_max,
        white_legal_moves_weight=config.white_legal_moves_weight,
        white_material_score_weight=config.white_material_score_weight,
        white_forward_score_weight=config.white_forward_score_weight,
        white_center_control_weight=config.white_center_control_weight,
        black_legal_moves_weight=config.black_legal_moves_weight,
        black_material_score_weight=config.black_material_score_weight,
        black_forward_score_weight=config.black_forward_score_weight,
        black_center_control_weight=config.black_center_control_weight,
        mirror_colors=mirror_colors,
    )


def _job_log_path(job_id: str) -> Path:
    return SELF_PLAY_JOBS_DIR / f"{job_id}.log"


def _job_pid_path(job_id: str) -> Path:
    return SELF_PLAY_JOBS_DIR / f"{job_id}.pid"


def _write_job_pid_file(job_id: str, run_id: str, pid: int, cmd: list[str]) -> None:
    payload = {
        "job_id": job_id,
        "run_id": run_id,
        "pid": pid,
        "cmd": cmd,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _job_pid_path(job_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _remove_job_pid_file(job_id: str) -> None:
    try:
        _job_pid_path(job_id).unlink()
    except OSError:
        pass


def _read_job_pid(pid_path: Path) -> int | None:
    try:
        raw = pid_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if not raw.strip():
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return int(raw.strip())
        except ValueError:
            return None

    try:
        return int(data["pid"])
    except (KeyError, TypeError, ValueError):
        return None


def _terminate_process_tree(pid: int) -> None:
    if pid <= 0:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        except OSError:
            return
        if sig == signal.SIGTERM:
            time.sleep(0.5)


def terminate_orphaned_self_play_workers() -> None:
    """Kill detached worker trees left behind by earlier app runs."""

    try:
        pid_files = list(SELF_PLAY_JOBS_DIR.glob("*.pid"))
    except OSError:
        return

    for pid_path in pid_files:
        pid = _read_job_pid(pid_path)
        if pid is None:
            try:
                pid_path.unlink()
            except OSError:
                pass
            continue

        _terminate_process_tree(pid)
        try:
            pid_path.unlink()
        except OSError:
            pass


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


def _evaluate_board(board: chess.Board, config: SelfPlayConfig | None = None, legal_moves: int | None = None) -> float:
    if legal_moves is None:
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


def _score_pct_to_elo(score_pct: float) -> float:
    clipped = min(max(score_pct, 1e-9), 1.0 - 1e-9)
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def _candidate_score_pct(
    samples: list[tuple[int, int, int, float, int, float]],
    weights: tuple[float, float, float, float, float],
    *,
    temperature: float,
) -> float:
    if not samples:
        return 0.0

    lm_w, mat_w, fwd_w, cc_w, mate_w = weights
    total_score = 0.0
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
        total_score += probability if target >= 0.5 else (1.0 - probability)
    return total_score / len(samples)


def _evaluate_candidate(
    samples: list[tuple[int, int, int, float, int, float]],
    weights: tuple[float, float, float, float, float],
    *,
    temperature: float,
) -> float:
    return _score_pct_to_elo(_candidate_score_pct(samples, weights, temperature=temperature))


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
    best_validation_elo = _evaluate_candidate(validation_samples, base_weights, temperature=temperature)
    history: list[dict[str, float]] = [
        {
            "legal_moves_weight": base_weights[0],
            "material_score_weight": base_weights[1],
            "forward_score_weight": base_weights[2],
            "center_control_weight": base_weights[3],
            "checkmate_weight": base_weights[4],
            "validation_elo": best_validation_elo,
        }
    ]

    log_min = math.log(min_multiplier)
    log_max = math.log(max_multiplier)

    for _ in range(max(1, iterations)):
        candidate = tuple(
            base * math.exp(rng.uniform(log_min, log_max))
            for base in base_weights
        )
        training_elo = _evaluate_candidate(train_samples, candidate, temperature=temperature)
        validation_elo = _evaluate_candidate(validation_samples, candidate, temperature=temperature)
        history.append({
            "legal_moves_weight": candidate[0],
            "material_score_weight": candidate[1],
            "forward_score_weight": candidate[2],
            "center_control_weight": candidate[3],
            "checkmate_weight": candidate[4],
            "training_elo": training_elo,
            "validation_elo": validation_elo,
        })
        if validation_elo > best_validation_elo:
            best_validation_elo = validation_elo
            best_weights = candidate

    return {
        "best_weights": {
            "legal_moves_weight": best_weights[0],
            "material_score_weight": best_weights[1],
            "forward_score_weight": best_weights[2],
            "center_control_weight": best_weights[3],
            "checkmate_weight": best_weights[4],
        },
        "best_validation_elo": best_validation_elo,
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "history": history,
    }


def _terminal_reason(
    board: chess.Board,
    repetition_counts: Counter | None = None,
) -> tuple[str, str]:
    """Return the game's terminal result without replaying its move stack.

    ``Board.can_claim_threefold_repetition()`` reconstructs and scans the
    complete reversible history on every call. That is disproportionately
    expensive in self-play, where this check runs after every ply. The caller
    can provide counts maintained while moves are pushed; the only remaining
    work for a claim is checking the resulting position for each legal move.
    """
    if board.is_checkmate():
        return ("1-0" if board.turn == chess.BLACK else "0-1", "checkmate")
    if board.is_stalemate():
        return ("1/2-1/2", "stalemate")
    if board.is_insufficient_material():
        return ("1/2-1/2", "insufficient material")
    if repetition_counts is None:
        fivefold = board.is_fivefold_repetition()
    else:
        fivefold = repetition_counts[board._transposition_key()] >= 5
    if fivefold:
        return ("1/2-1/2", "5-fold-rep")
    if board.is_seventyfive_moves():
        return ("1/2-1/2", "75-move rule")

    def _is_perpetual_check(
        board: chess.Board,
        *,
        lookback_plies: int = 16,
        min_king_moves: int = 4,
    ) -> bool:
        """Detect a perpetual-check loop (heuristic label) from recent moves.

        This does not change the draw mechanics (threefold repetition is the
        actual rule). It only classifies a claimable threefold repetition as
        "perpetual check" when the recent move sequence matches:

        - one side gives check on every turn, and
        - the defending king responds by moving back and forth between the same
          two squares.
        """
        needed_plies = 2 * min_king_moves
        if min_king_moves <= 0 or len(board.move_stack) < needed_plies:
            return False

        tmp = board.copy(stack=True)
        after = tmp.copy(stack=False)
        records: list[tuple[bool, bool, bool, int | None]] = []
        for _ in range(min(int(lookback_plies), len(tmp.move_stack))):
            move = tmp.pop()
            mover = not after.turn  # side that played `move`
            gave_check = after.is_check()
            piece = tmp.piece_at(move.from_square)
            moved_king = piece is not None and piece.piece_type == chess.KING
            king_to = move.to_square if moved_king else None
            records.append((mover, gave_check, moved_king, king_to))
            after = tmp.copy(stack=False)
        if len(records) < needed_plies:
            return False

        records.reverse()  # chronological order
        suffix = records[-needed_plies:]
        for defender in (chess.WHITE, chess.BLACK):
            attacker = not defender
            defender_moves = [r for r in suffix if r[0] == defender]
            attacker_moves = [r for r in suffix if r[0] == attacker]
            if len(defender_moves) != min_king_moves or len(attacker_moves) != min_king_moves:
                continue
            if not all(moved_king for (_, _, moved_king, _) in defender_moves):
                continue
            king_squares = [sq for (_, _, _, sq) in defender_moves]
            if any(sq is None for sq in king_squares):
                continue
            if len(set(king_squares)) != 2:
                continue
            if any(king_squares[i] != king_squares[i % 2] for i in range(len(king_squares))):
                continue
            if not all(gave_check for (_, gave_check, _, _) in attacker_moves):
                continue
            return True

        return False

    if repetition_counts is None:
        can_claim_threefold = board.can_claim_threefold_repetition()
    else:
        current_key = board._transposition_key()
        can_claim_threefold = repetition_counts[current_key] >= 3
        if not can_claim_threefold:
            for move in board.generate_legal_moves():
                board.push(move)
                try:
                    if repetition_counts[board._transposition_key()] >= 2:
                        can_claim_threefold = True
                        break
                finally:
                    board.pop()
    if can_claim_threefold:
        if _is_perpetual_check(board):
            return ("1/2-1/2", "perpetual check")
        return ("1/2-1/2", "3-fold-rep")
    if board.can_claim_fifty_moves():
        return ("1/2-1/2", "50-moves")
    return ("", "")


def play_self_game(config: SelfPlayConfig, game_index: int, run_id: str | None = None, rng: random.Random | None = None) -> SelfPlayGame:
    rng = rng or random.Random(config.seed)
    board = chess.Board(config.fen) if config.fen else chess.Board()
    white_player, white_weights, black_player, black_weights = _player_weight_sets(config, rng)
    if config.mirror_colors:
        white_player, black_player = black_player, white_player
        white_weights, black_weights = black_weights, white_weights
    white_player_name = white_player.name if white_player is not None else "Custom White"
    black_player_name = black_player.name if black_player is not None else "Custom Black"
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-play harness"
    game.headers["Site"] = "Local"
    game.headers["Round"] = str(game_index)
    game.headers["White"] = white_player_name
    game.headers["Black"] = black_player_name
    game.headers["WhiteWeights"] = json.dumps(white_weights, sort_keys=True)
    game.headers["BlackWeights"] = json.dumps(black_weights, sort_keys=True)
    if white_player is not None:
        game.headers["WhitePlayerId"] = white_player.player_id
        game.headers["WhitePlayerDescription"] = white_player.description
    if black_player is not None:
        game.headers["BlackPlayerId"] = black_player.player_id
        game.headers["BlackPlayerDescription"] = black_player.description

    node = game
    turn = 0
    evaluations = 0
    turn_durations_seconds: list[float] = []
    start_time = time.perf_counter()
    result = ""
    termination = ""
    try:
        result, termination, turn, moves, evaluations, turn_durations_seconds = play_self_game_native(
            board.fen(),
            config.max_turns,
            config.top_k,
            rng.getrandbits(64),
            legal_moves_weight=white_weights["legal_moves_weight"],
            material_score_weight=white_weights["material_score_weight"],
            forward_score_weight=white_weights["forward_score_weight"],
            forward_material_score_weight=config.forward_material_score_weight,
            center_control_weight=white_weights["center_control_weight"],
            black_legal_moves_weight=black_weights["legal_moves_weight"],
            black_material_score_weight=black_weights["material_score_weight"],
            black_forward_score_weight=black_weights["forward_score_weight"],
            black_forward_material_score_weight=config.forward_material_score_weight,
            black_center_control_weight=black_weights["center_control_weight"],
            checkmate_weight=config.checkmate_weight,
            depth=config.depth,
            max_depth=config.max_depth,
            top_k_score_threshold=config.top_k_score_threshold,
            blunder_control=config.blunder_control,
        )
        # Rebuild the presentation board/PGN after the native loop. This is
        # linear formatting work; search, move generation, and board mutation
        # all remain inside Rust.
        for uci in moves:
            move = chess.Move.from_uci(uci)
            san = board.san(move)
            board.push(move)
            node = node.add_variation(move)
            node.comment = san
    except Exception:
        # A crashed game shouldn't take the rest of the batch down with it
        # (this is submitted as one ProcessPoolExecutor unit of work per
        # game): record it as a terminal "Crash" result instead of letting
        # the exception propagate out of the worker.
        traceback.print_exc()
        result, termination = "0-0", "Crash"
    duration_seconds = time.perf_counter() - start_time

    if not result:
        result = "1/2-1/2"
        termination = "max turns"

    game.headers["Result"] = result
    game.headers["Termination"] = termination

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    pgn_text = game.accept(exporter)
    final_legal_moves = len(list(board.legal_moves))
    final_score = _evaluate_board(board, legal_moves=final_legal_moves)
    if termination == "Crash":
        summary = {"status": "Crash", "winner": "", "loser": ""}
    else:
        summary = _result_summary(result, white=white_player_name, black=black_player_name)
    return SelfPlayGame(
        index=game_index,
        result=result,
        termination=termination,
        turns=turn,
        pgn=pgn_text,
        final_fen=board.fen(),
        final_score=final_score,
        outcome=summary["status"],
        winner=summary["winner"],
        loser=summary["loser"],
        white_weights=white_weights,
        black_weights=black_weights,
        white_player_id=white_player.player_id if white_player is not None else None,
        white_player_name=white_player_name,
        white_player_description=white_player.description if white_player is not None else None,
        black_player_id=black_player.player_id if black_player is not None else None,
        black_player_name=black_player_name,
        black_player_description=black_player.description if black_player is not None else None,
        duration_seconds=duration_seconds,
        evaluations=evaluations,
        evaluations_per_move=(evaluations / turn) if turn else 0.0,
        turn_durations_seconds=turn_durations_seconds,
    )


def _play_and_save_game(
    config: SelfPlayConfig,
    game_index: int,
    run_id: str,
    played_at: str,
) -> SelfPlayGame:
    """Play one game and return it to the caller for persistence."""
    played_at = datetime.now(timezone.utc).isoformat()
    print(f"Starting self-play game {game_index} in process {os.getpid()}", flush=True)
    game = play_self_game(config, game_index, run_id=run_id)
    game.run_id = run_id
    grouping = build_run_grouping(
        run_name=config.run_name,
        timestamp=played_at,
        default_name="self-play",
    )
    game.run_name = grouping.run_name
    game.run_date = grouping.run_date
    game.run_group = grouping.run_group
    game.played_at = played_at
    game.seed = config.seed
    game.top_k = config.top_k
    game.top_k_score_threshold = config.top_k_score_threshold
    game.blunder_control = config.blunder_control
    game.max_turns = config.max_turns
    game.start_fen = config.fen or "startpos"
    print(
        f"Finished self-play game {game_index} in process {os.getpid()}: "
        f"{game.result} after {game.turns} turns ({game.termination})",
        flush=True,
    )
    return game


def _announce_self_play_worker() -> None:
    """Announce each process created by the self-play process pool."""
    print(f"Launched self-play pool worker process {os.getpid()}", flush=True)


def run_self_play(
    config: SelfPlayConfig,
    *,
    progress_callback: Callable[[int, SelfPlayGame, list[SelfPlayGame]], None] | None = None,
    run_id: str | None = None,
) -> list[SelfPlayGame]:
    played_at = datetime.now(timezone.utc).isoformat()
    run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    games: list[SelfPlayGame] = []

    if config.games <= 1:
        game_config = _config_for_game(config, 1)
        game = _play_and_save_game(game_config, 1, run_id, played_at)
        save_self_play_results([game], refresh_player_elos=True)
        games.append(game)
        if progress_callback is not None:
            progress_callback(1, game, games)
        return games

    total_games = _paired_game_count(config.games)
    requested_workers = config.workers or DEFAULT_SELF_PLAY_WORKERS
    max_workers = max(1, min(int(requested_workers), total_games))
    print(f"Starting self-play process pool with {max_workers} workers", flush=True)
    future_to_game: dict = {}
    saved_results = load_self_play_results(limit=None)
    completed_rebalance_batches = len(saved_results) // SELF_PLAY_REBALANCE_BATCH_SIZE
    pending_saved_games = len(saved_results) % SELF_PLAY_REBALANCE_BATCH_SIZE
    rebalance_batch = [
        _saved_self_play_game(row)
        for row in saved_results[-pending_saved_games:]
    ] if pending_saved_games else []
    rebalance_batch_number = completed_rebalance_batches
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_announce_self_play_worker,
    ) as executor:
        game_index = 1
        for pair_index in range(1, total_games // 2 + 1):
            pair_seed = _seed_for_game(config, pair_index)
            first_config = _config_for_game(config, game_index, seed=pair_seed, mirror_colors=False)
            second_config = _config_for_game(config, game_index + 1, seed=pair_seed, mirror_colors=True)
            future = executor.submit(_play_and_save_game, first_config, game_index, run_id, played_at)
            future_to_game[future] = (game_index, first_config)
            future = executor.submit(_play_and_save_game, second_config, game_index + 1, run_id, played_at)
            future_to_game[future] = (game_index + 1, second_config)
            game_index += 2

        if total_games % 2:
            single_config = _config_for_game(config, game_index, mirror_colors=False)
            future = executor.submit(_play_and_save_game, single_config, game_index, run_id, played_at)
            future_to_game[future] = (game_index, single_config)

        completed_games: dict[int, SelfPlayGame] = {}
        completed_count = 0
        elo_batch: list[SelfPlayGame] = []
        for future in as_completed(future_to_game):
            index, game_config = future_to_game[future]
            try:
                game = future.result()
            except Exception:
                # The worker process itself died (crashed/killed) before it could
                # return or save a result -- distinct from an in-game exception,
                # which play_self_game already catches and reports as "Crash".
                traceback.print_exc()
                game = SelfPlayGame(
                    index=index,
                    result="0-0",
                    termination="disconnect",
                    turns=0,
                    pgn="",
                    final_fen="",
                    final_score=0.0,
                    outcome="Disconnected",
                    run_id=run_id,
                    played_at=played_at,
                    seed=game_config.seed,
                    top_k=config.top_k,
                    top_k_score_threshold=config.top_k_score_threshold,
                    max_turns=config.max_turns,
                    start_fen=config.fen or "startpos",
                )
            # Persist each result as soon as its worker completes. This keeps
            # completed games durable even if a later worker or the job itself
            # fails. Elo refreshes remain batched because they scan all games.
            save_self_play_results([game], refresh_player_elos=False)
            elo_batch.append(game)
            if len(elo_batch) >= SELF_PLAY_ELO_BATCH_SIZE:
                refresh_self_play_player_elos()
                elo_batch.clear()
            rebalance_batch.append(game)
            if len(rebalance_batch) >= SELF_PLAY_REBALANCE_BATCH_SIZE:
                rebalance_batch_number += 1
                _print_batch_summary(rebalance_batch_number, rebalance_batch, batch_size=SELF_PLAY_REBALANCE_BATCH_SIZE)
                print(
                    f"Rebalancing player weights after batch {rebalance_batch_number} "
                    f"({len(rebalance_batch)} games)..."
                )
                try:
                    updated = rebalance_self_play_players(rebalance_batch)
                    print(f"  updated {updated} players")
                except Exception:
                    traceback.print_exc()
                try:
                    refresh_self_play_player_elos()
                except Exception:
                    traceback.print_exc()
                rebalance_batch.clear()
            completed_games[index] = game
            completed_count += 1
            ordered_games = [completed_games[i] for i in sorted(completed_games)]
            if progress_callback is not None:
                progress_callback(completed_count, game, ordered_games)

    if rebalance_batch:
        print(
            "Weight update deferred for partial rebalance batch: "
            f"{len(rebalance_batch)}/{SELF_PLAY_REBALANCE_BATCH_SIZE} games."
        )

    # Refresh ratings for a final partial Elo batch as well.
    try:
        refresh_self_play_player_elos()
    except Exception:
        traceback.print_exc()

    return [completed_games[i] for i in sorted(completed_games)]


def refresh_self_play_player_elos() -> int:
    with Neo4jStore() as store:
        return store.refresh_self_play_player_elos()


def save_self_play_results(games: list[SelfPlayGame], *, refresh_player_elos: bool = True) -> None:
    if not games:
        return

    payloads = []
    for game in games:
        grouping = build_run_grouping(
            run_name=game.run_name or None,
            timestamp=game.played_at or None,
            default_name="self-play",
        )
        payloads.append(
            {
                "played_at": game.played_at or datetime.now(timezone.utc).isoformat(),
                "run_id": game.run_id,
                "run_name": grouping.run_name,
                "run_date": grouping.run_date,
                "run_group": grouping.run_group,
                "index": game.index,
                "seed": game.seed,
                "top_k": game.top_k,
                "top_k_score_threshold": game.top_k_score_threshold,
                "blunder_control": game.blunder_control,
                "max_turns": game.max_turns,
                "start_fen": game.start_fen,
                "result": game.result,
                "termination": game.termination,
                "turns": game.turns,
                "final_fen": game.final_fen,
                "final_score": game.final_score,
                "outcome": game.outcome,
                "winner": game.winner,
                "loser": game.loser,
                "white_weights": game.white_weights,
                "black_weights": game.black_weights,
                "white_player_id": game.white_player_id,
                "white_player_name": game.white_player_name,
                "white_player_description": game.white_player_description,
                "black_player_id": game.black_player_id,
                "black_player_name": game.black_player_name,
                "black_player_description": game.black_player_description,
                "duration_seconds": game.duration_seconds,
                "evaluations": game.evaluations,
                "evaluations_per_move": game.evaluations_per_move,
                "turn_durations_seconds": game.turn_durations_seconds,
                "pgn": game.pgn,
            }
        )

    with Neo4jStore() as store:
        store.save_self_play_games(payloads)
        if refresh_player_elos:
            store.refresh_self_play_player_elos()


def _replace_worst_player_from_elite(batch_df, store: Neo4jStore) -> int:
    """Replace the batch's worst player with a lightly mutated elite profile."""
    overview = player_overview(batch_df)
    if len(overview) < SELF_PLAY_ELITE_COUNT * 2:
        return 0

    ranked = overview.sort_values(
        ["score_pct", "games", "player_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    elite_ids = {str(value) for value in ranked.head(SELF_PLAY_ELITE_COUNT)["player_id"]}
    worst = ranked.iloc[-1]
    worst_id = str(worst["player_id"])
    if worst_id in elite_ids:
        return 0

    players = {
        str(row.get("player_id")): row
        for row in store.load_self_play_players()
        if row.get("player_id") is not None
    }
    elite_rows = [players[player_id] for player_id in elite_ids if player_id in players]
    worst_row = players.get(worst_id)
    if not elite_rows or worst_row is None:
        return 0

    donor = random.choice(elite_rows)
    rng = random.Random()
    dimensions = (
        "legal_moves_weight",
        "material_score_weight",
        "forward_score_weight",
        "center_control_weight",
    )
    mutated = {
        dimension: min(
            4.0,
            max(-4.0, float(donor.get(dimension) or 0.0) + rng.gauss(0.0, SELF_PLAY_ELITE_MUTATION_STDDEV)),
        )
        for dimension in dimensions
    }
    update = {
        "player_id": worst_id,
        "player_name": worst_row.get("name", worst.get("player_name", worst_id)),
        "player_description": worst_row.get("description", worst.get("player_description", "")),
        "games": int(worst.get("games", 0)),
        "score_pct": float(worst.get("score_pct", 0.0)),
        "shap_legal_moves_weight": 0.0,
        "shap_material_score_weight": 0.0,
        "shap_forward_score_weight": 0.0,
        "shap_center_control_weight": 0.0,
        "delta_legal_moves_weight": mutated["legal_moves_weight"] - float(worst_row.get("legal_moves_weight") or 0.0),
        "delta_material_score_weight": mutated["material_score_weight"] - float(worst_row.get("material_score_weight") or 0.0),
        "delta_forward_score_weight": mutated["forward_score_weight"] - float(worst_row.get("forward_score_weight") or 0.0),
        "delta_center_control_weight": mutated["center_control_weight"] - float(worst_row.get("center_control_weight") or 0.0),
    }
    update.update({f"updated_{dimension}": value for dimension, value in mutated.items()})
    store.update_self_play_player_weights([update])
    print(
        f"  elite replacement: {worst_id} replaced from "
        f"{donor.get('player_id', 'elite')} with mutation ±{SELF_PLAY_ELITE_MUTATION_STDDEV:.3f}"
    )
    return 1


def rebalance_self_play_players(games: list[SelfPlayGame]) -> int:
    """Update player weights from a batch using SHAP and elite replacement."""
    if not games:
        return 0

    batch_df = self_play_to_dataframe([asdict(game) for game in games])
    if batch_df.empty:
        return 0

    updates = shap_balance_player_weights(batch_df, learning_rate=SHAP_BALANCE_LEARNING_RATE)
    with Neo4jStore() as store:
        updated = 0
        if not updates.empty:
            updated = store.update_self_play_player_weights(updates.to_dict(orient="records"))
        updated += _replace_worst_player_from_elite(batch_df, store)
        return updated


def load_self_play_results(limit: int | None = 50) -> list[dict]:
    with Neo4jStore() as store:
        rows = store.load_self_play_games(limit)
    return [_normalize_result(row) for row in rows]


def load_self_play_result(run_id: str, index: int) -> dict | None:
    with Neo4jStore() as store:
        row = store.load_self_play_game(run_id, index)
    return _normalize_result(row) if row is not None else None


def _normalize_result(row: dict) -> dict:
    row.setdefault("duration_seconds", 0.0)
    row.setdefault("evaluations", 0)
    row.setdefault("evaluations_per_move", 0.0)
    row.setdefault("turn_durations_seconds", None)
    if "turns" not in row and "plies" in row:
        row["turns"] = row["plies"]
    if "plies" not in row and "turns" in row:
        row["plies"] = row["turns"]
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
    run_self_play(bootstrap_config)
    return load_self_play_results(limit=limit)


def _run_self_play_job(job_id: str, run_id: str, config_data: dict, reporter: "SelfPlayJobClient") -> None:
    config = SelfPlayConfig(**config_data)
    total_games = _paired_game_count(config.games)
    worker_count = max(
        1,
        min(int(config.workers or DEFAULT_SELF_PLAY_WORKERS), total_games),
    )
    started_at = time.monotonic()
    status = SelfPlayJobStatus(
        job_id=job_id,
        state="running",
        total=total_games,
        completed=0,
        message=f"Completed 0 of {total_games} | Running {worker_count} | ETA calculating",
        played_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )

    status_lock = threading.Lock()
    stop_heartbeat = threading.Event()

    def _progress_message(completed: int) -> str:
        running = min(worker_count, max(0, total_games - completed))
        if completed:
            elapsed = time.monotonic() - started_at
            remaining_seconds = max(0.0, (elapsed / completed) * (total_games - completed))
            remaining_minutes = int(math.ceil(remaining_seconds / 60))
            if remaining_minutes >= 60:
                eta = f"{remaining_minutes // 60}h {remaining_minutes % 60:02d}m"
            else:
                eta = f"{remaining_minutes}m"
        else:
            eta = "calculating"
        return f"Completed {completed} of {total_games} | Running {running} | ETA {eta}"

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
                status.message = _progress_message(status.completed)
                reporter.send(status)

    _send_status()
    heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    try:
        def progress_callback(completed: int, game: SelfPlayGame, games: list[SelfPlayGame]) -> None:
            with status_lock:
                status.completed = completed
                status.message = _progress_message(completed)
                status.run_id = game.run_id or run_id
                reporter.send(status)

        run_self_play(config, progress_callback=progress_callback, run_id=run_id)
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


def _tee_output(infile: Any, log_file: Any, stdout: Any) -> None:
    """Read lines from infile and write to each outfile, flushing each time."""
    try:
        for line in infile:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            # Write to log file
            try:
                log_file.write(line)
                log_file.flush()
            except Exception:
                pass
            # Write to stdout
            try:
                stdout.write(line)
                stdout.flush()
            except Exception:
                pass
    except Exception:
        pass
    finally:
        try:
            infile.close()
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass


def start_self_play_job(config: SelfPlayConfig) -> dict:
    hub = get_job_hub()
    prune_old_jobs()
    job_id = uuid4().hex
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    status = SelfPlayJobStatus(
        job_id=job_id,
        state="queued",
        total=_paired_game_count(config.games),
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
            subprocess.CREATE_NEW_PROCESS_GROUP
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
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            creationflags=creationflags,
            close_fds=True,
            start_new_session=os.name != "nt",
        )
        _put_process_in_parent_lifetime_job(proc)
        _write_job_pid_file(job_id, run_id, proc.pid, cmd)
        print(f"Launched self-play process {proc.pid} for job {job_id}", flush=True)

        # Tee worker output to both the log file and the main process's stdout
        # so it's visible in Cloud Run logs.
        threading.Thread(
            target=_tee_output,
            args=(proc.stdout, log_handle, sys.stdout),
            daemon=True,
            name=f"job-{job_id}-tee",
        ).start()

        assert proc.stdin is not None
        proc.stdin.write(json.dumps(asdict(config)).encode("utf-8"))
        proc.stdin.close()
    except Exception as e:
        print(f"FAILED to launch worker: {e}", flush=True)
        _remove_job_pid_file(job_id)
        log_handle.close()
        raise
    return asdict(status)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the position scorer against itself.")
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of self-play games to run; multi-game runs are mirrored so odd values are rounded up.",
    )
    parser.add_argument("--max-turns", type=int, default=100, help="Stop each game after this many turns.")
    parser.add_argument("--top-k", type=int, default=1, help="Randomly choose among the top K evaluated moves.")
    parser.add_argument(
        "--top-k-score-threshold",
        type=float,
        default=3.0,
        help="Only choose Top-K moves within this score distance of the best move (default: 3.0).",
    )
    parser.add_argument(
        "--blunder-control",
        type=float,
        default=0.0,
        help="Probability (0-1) of selecting any legal move instead of a Top-K move.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Fixed negamax search depth (1, 2, 3, ...). Omit to auto-derive "
        "depth per move from remaining material, capped by --max-depth.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_AUTO_SEARCH_DEPTH,
        help="Maximum search depth used by automatic depth scaling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel worker processes for multi-game self-play (default: CPU count).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for move selection.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional human-readable run name.")
    parser.add_argument("--fen", type=str, default=None, help="Optional starting FEN.")
    parser.add_argument("--legal-moves-weight", type=float, default=LEGAL_MOVES_WEIGHT, help="Weight for legal move count.")
    parser.add_argument("--material-score-weight", type=float, default=MATERIAL_SCORE_WEIGHT, help="Weight for material balance.")
    parser.add_argument("--forward-score-weight", type=float, default=FORWARD_SCORE_WEIGHT, help="Weight for forward control.")
    parser.add_argument("--forward-material-score-weight", type=float, default=FORWARD_MATERIAL_SCORE_WEIGHT, help="Weight for material in the forward zone.")
    parser.add_argument("--center-control-weight", type=float, default=CENTER_CONTROL_WEIGHT, help="Weight for center control.")
    parser.add_argument("--checkmate-weight", type=float, default=CHECKMATE_WEIGHT, help="Weight for the mate-pressure heuristic (drive the enemy king toward checkmate).")
    parser.add_argument("--fixed-player-weights", action="store_true", help="Use the same weights for both sides.")
    parser.add_argument(
        "--player-weight-min",
        "--player-weight-min-multiplier",
        dest="player_weight_min",
        type=float,
        default=_env_float("SELF_PLAY_PLAYER_WEIGHT_MIN", -4.0),
        help="Lower bound for absolute per-player random weights.",
    )
    parser.add_argument(
        "--player-weight-max",
        "--player-weight-max-multiplier",
        dest="player_weight_max",
        type=float,
        default=_env_float("SELF_PLAY_PLAYER_WEIGHT_MAX", 4.0),
        help="Upper bound for absolute per-player random weights.",
    )
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
        max_turns=max(2, args.max_turns),
        top_k=max(1, args.top_k),
        top_k_score_threshold=(
            max(0.0, args.top_k_score_threshold)
            if args.top_k_score_threshold is not None
            else None
        ),
        blunder_control=max(0.0, min(1.0, args.blunder_control)),
        depth=(max(1, args.depth) if args.depth is not None else None),
        max_depth=max(1, args.max_depth),
        workers=(max(1, int(args.workers)) if args.workers else None),
        seed=args.seed,
        run_name=args.run_name,
        fen=args.fen,
        legal_moves_weight=args.legal_moves_weight,
        material_score_weight=args.material_score_weight,
        forward_score_weight=args.forward_score_weight,
        forward_material_score_weight=args.forward_material_score_weight,
        center_control_weight=args.center_control_weight,
        checkmate_weight=args.checkmate_weight,
        randomize_player_weights=not args.fixed_player_weights,
        player_weight_min=args.player_weight_min,
        player_weight_max=args.player_weight_max,
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
        print(f"Validation Elo: {tuning['best_validation_elo']:.2f}")
        if args.tune_output:
            args.tune_output.write_text(json.dumps(tuning, indent=2), encoding="utf-8")

    print(f"Self-play rebalance batch size: {SELF_PLAY_REBALANCE_BATCH_SIZE}")
    games = run_self_play(config)

    if args.output:
        args.output.write_text("\n\n".join(game.pgn for game in games) + "\n", encoding="utf-8")

    _print_result_summary("Overall results", games)
    _print_result_summary("Last 250 games", games[-250:])

    for game in games:
        white = game.white_weights or {}
        black = game.black_weights or {}
        print(
            f"Game {game.index}: {game.result} after {game.turns} turns ({game.termination}); "
            f"final score {game.final_score}; "
            f"took {game.duration_seconds:.2f}s, {game.evaluations_per_move:.0f} evals/move; "
            f"W[lm={white.get('legal_moves_weight', 0):.6f}, mat={white.get('material_score_weight', 0):.6f}, fwd={white.get('forward_score_weight', 0):.6f}] "
            f"B[lm={black.get('legal_moves_weight', 0):.6f}, mat={black.get('material_score_weight', 0):.6f}, fwd={black.get('forward_score_weight', 0):.6f}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
