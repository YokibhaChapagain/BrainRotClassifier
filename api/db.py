"""Tiger Data (TimescaleDB) access. Everything no-ops / returns empty when DATABASE_URL is unset,
so the API and tests run without a database."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime

log = logging.getLogger("brainrot.db")


def enabled() -> bool:
    url = os.getenv("DATABASE_URL", "")
    return bool(url) and "user:pass" not in url


@contextmanager
def connect():
    import psycopg

    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        yield conn


def _rows(sql: str, params: tuple) -> list[tuple]:
    if not enabled():
        return []
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------- writes
def log_view(v: dict) -> None:
    if not enabled():
        return
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO reel_views (time, user_id, session_id, video_id, platform, watch_ms, "
                "video_duration_ms, loops, skipped, prob, flagged) "
                "VALUES (%(started_at)s, %(user_id)s, %(session_id)s, %(video_id)s, %(platform)s, %(watch_ms)s, "
                "%(video_duration_ms)s, %(loops)s, %(skipped)s, %(prob)s, %(flagged)s)",
                v,
            )
    except Exception as e:  # noqa: BLE001  never break the extension because of logging
        log.warning("log_view failed: %s", e)


# ---------------------------------------------------------------- dashboard queries
def summary(user_id: str, days: int) -> dict:
    rows = _rows(
        "SELECT coalesce(sum(views),0), coalesce(sum(rot_views),0), coalesce(sum(watch_ms),0), "
        "coalesce(sum(rot_watch_ms),0), coalesce(sum(skips),0) "
        "FROM views_hourly WHERE user_id = %s AND bucket > now() - make_interval(days => %s)",
        (user_id, days),
    )
    views, rot, watch, rot_watch, skips = rows[0] if rows else (0, 0, 0, 0, 0)
    streak = _rows(
        # gaps-and-islands: longest run of consecutive flagged views
        "WITH v AS (SELECT time, flagged, row_number() OVER (ORDER BY time) "
        "  - row_number() OVER (PARTITION BY flagged ORDER BY time) AS grp "
        "  FROM reel_views WHERE user_id = %s AND time > now() - make_interval(days => %s)) "
        "SELECT coalesce(max(n),0) FROM (SELECT count(*) n FROM v WHERE flagged GROUP BY grp) s",
        (user_id, days),
    )
    return {
        "views": int(views), "rot_views": int(rot), "rot_share": rot / views if views else 0.0,
        "watch_min": watch / 60000, "rot_watch_min": rot_watch / 60000,
        "rot_time_share": rot_watch / watch if watch else 0.0,
        "skip_rate": skips / views if views else 0.0,
        "longest_rot_streak": int(streak[0][0]) if streak else 0,
    }


def timeline(user_id: str, days: int, bucket: str) -> list[dict]:
    view = "views_daily" if bucket == "day" else "views_hourly"
    rows = _rows(
        f"SELECT bucket, views, rot_views, watch_ms, rot_watch_ms FROM {view} "  # noqa: S608 (whitelisted)
        "WHERE user_id = %s AND bucket > now() - make_interval(days => %s) ORDER BY bucket",
        (user_id, days),
    )
    return [{"bucket": _iso(b), "views": v, "rot_views": r, "watch_min": w / 60000, "rot_watch_min": rw / 60000}
            for b, v, r, w, rw in rows]


def heatmap(user_id: str, days: int) -> list[dict]:
    rows = _rows(
        "SELECT extract(isodow FROM bucket)::int, extract(hour FROM bucket)::int, sum(views), sum(rot_views) "
        "FROM views_hourly WHERE user_id = %s AND bucket > now() - make_interval(days => %s) GROUP BY 1, 2",
        (user_id, days),
    )
    return [{"dow": d, "hour": h, "views": int(v), "rot_share": (r / v) if v else 0.0} for d, h, v, r in rows]


def sessions(user_id: str, days: int) -> list[dict]:
    rows = _rows(
        "SELECT session_id, min(time), sum(watch_ms), count(*), count(*) FILTER (WHERE flagged) "
        "FROM reel_views WHERE user_id = %s AND time > now() - make_interval(days => %s) "
        "GROUP BY session_id ORDER BY 2 DESC LIMIT 50",
        (user_id, days),
    )
    return [{"session_id": s, "start": _iso(t), "minutes": w / 60000, "views": n, "rot_views": r}
            for s, t, w, n, r in rows]


def engagement() -> list[dict]:
    """Dataset-level: do brain rot videos get more likes/comments/shares per view?"""
    rows = _rows(
        "SELECT brainrot, count(*), avg(likes::float / nullif(plays,0)), avg(comments::float / nullif(plays,0)), "
        "avg(shares::float / nullif(plays,0)), percentile_cont(0.5) WITHIN GROUP (ORDER BY plays) "
        "FROM dataset_videos GROUP BY brainrot ORDER BY brainrot", (),
    )
    return [{"brainrot": bool(b), "videos": n, "like_rate": lr, "comment_rate": cr, "share_rate": sr,
             "median_plays": mp} for b, n, lr, cr, sr, mp in rows]


def trend() -> list[dict]:
    rows = _rows(
        "SELECT time_bucket('1 year', time) AS y, count(*), avg(brainrot::float) "
        "FROM dataset_videos GROUP BY y ORDER BY y", (),
    )
    return [{"year": _iso(y)[:4], "videos": n, "rot_share": s} for y, n, s in rows]


def _iso(t) -> str:
    return t.isoformat() if isinstance(t, datetime) else str(t)
