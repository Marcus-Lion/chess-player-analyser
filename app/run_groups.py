from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str | None, *, default: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    slug = _SLUG_RE.sub("-", raw).strip("-")
    return slug or default


def iso_date(value: str | None = None) -> str:
    if not value:
        return datetime.now(timezone.utc).date().isoformat()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc).date().isoformat()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


def normalize_run_name(value: str | None, *, default: str) -> str:
    name = (value or "").strip()
    return name if name else default


@dataclass(slots=True)
class RunGrouping:
    run_name: str
    run_date: str
    run_group: str
    run_slug: str


def build_run_grouping(*, run_name: str | None, timestamp: str | None, default_name: str) -> RunGrouping:
    normalized_name = normalize_run_name(run_name, default=default_name)
    run_date = iso_date(timestamp)
    run_slug = slugify(normalized_name, default=default_name)
    return RunGrouping(
        run_name=normalized_name,
        run_date=run_date,
        run_group=f"{run_date}/{run_slug}",
        run_slug=run_slug,
    )
