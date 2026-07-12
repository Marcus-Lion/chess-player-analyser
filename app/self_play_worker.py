from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.self_play import _run_self_play_job


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single self-play job in a detached worker process.")
    parser.add_argument("--job-id", required=True, help="Self-play job id.")
    parser.add_argument("--request-path", required=True, type=Path, help="Path to the job request JSON file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    request = json.loads(args.request_path.read_text(encoding="utf-8"))
    job_id = request.get("job_id") or args.job_id
    run_id = request["run_id"]
    config_data = request["config"]

    _run_self_play_job(job_id, run_id, config_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
