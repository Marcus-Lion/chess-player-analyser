from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the ``app`` package importable no matter how this worker was launched.
# When run by file path (``python app/self_play_worker.py``) or via a debugger
# that rewrites the launch (PyCharm/pydevd), the project root is not on
# sys.path by default, which otherwise breaks ``from app.self_play import ...``.
# Doing this before importing ``app`` runs in the main process and in any
# multiprocessing spawn child (which re-imports this module).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.self_play import _run_self_play_job


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single self-play job in a detached worker process.")
    parser.add_argument("--job-id", required=True, help="Self-play job id.")
    parser.add_argument("--request-path", required=True, type=Path, help="Path to the job request JSON file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        request = json.loads(args.request_path.read_text(encoding="utf-8"))
        job_id = request.get("job_id") or args.job_id
        run_id = request["run_id"]
        config_data = request["config"]

        _run_self_play_job(job_id, run_id, config_data)
    except Exception as exc:  # pragma: no cover - defensive worker guard
        # Never exit silently: record the failure so the UI can surface it
        # instead of the job appearing to hang forever.
        import traceback

        traceback.print_exc()
        try:
            from app.self_play import SelfPlayJobStatus, _write_job_status

            _write_job_status(
                SelfPlayJobStatus(
                    job_id=args.job_id,
                    state="failed",
                    total=1,
                    message="Worker crashed",
                    error=str(exc),
                )
            )
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
