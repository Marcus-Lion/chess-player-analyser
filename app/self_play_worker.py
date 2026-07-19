from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import dotenv

# Make the ``app`` package importable no matter how this worker was launched.
# When run by file path (``python app/self_play_worker.py``) or via a debugger
# that rewrites the launch (PyCharm/pydevd), the project root is not on
# sys.path by default, which otherwise breaks ``from app.self_play import ...``.
# Doing this before importing ``app`` runs in the main process and in any
# multiprocessing spawn child (which re-imports this module).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.self_play import (
    SelfPlayJobClient,
    SelfPlayJobStatus,
    _remove_job_pid_file,
    _run_self_play_job,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single self-play job in a detached worker process.")
    parser.add_argument("--job-id", required=True, help="Self-play job id.")
    parser.add_argument("--run-id", required=True, help="Self-play run id.")
    parser.add_argument("--host", required=True, help="Host of the main process's job-status socket server.")
    parser.add_argument("--port", required=True, type=int, help="Port of the main process's job-status socket server.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # The job request (config dict) travels over stdin rather than a request
    # file, so nothing about this job touches disk except its log file.
    client = SelfPlayJobClient(args.host, args.port)
    try:
        config_data = json.loads(sys.stdin.read())
        _run_self_play_job(args.job_id, args.run_id, config_data, client)
    except Exception as exc:  # pragma: no cover - defensive worker guard
        # Never exit silently: report the failure over the socket so the UI
        # can surface it instead of the job appearing to hang forever. If the
        # main process is gone, SelfPlayJobClient.send() drops this quietly.
        import traceback

        traceback.print_exc()
        client.send(
            SelfPlayJobStatus(
                job_id=args.job_id,
                state="failed",
                total=1,
                message="Worker crashed",
                error=str(exc),
            )
        )
        return 1
    finally:
        client.close()
        _remove_job_pid_file(args.job_id)
    return 0


if __name__ == "__main__":
    dotenv.load_dotenv()
    raise SystemExit(main())
