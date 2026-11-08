"""Insert SYNTHETIC viewing history for user_id="demo" so the dashboard has something to show while
the extension is still being built. Clearly fake data — never present these numbers as real results.

    DATABASE_URL=... python scripts/seed_demo_views.py --days 7
    open https://<api>/dashboard?user=demo
"""

from __future__ import annotations

import argparse
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

import psycopg


def main(days: int, user: str, seed: int) -> None:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    rows = []
    for d in range(days):
        for hour in rng.sample(range(8, 26), k=rng.randint(2, 5)):  # a few sessions a day, some after midnight
            t = (now - timedelta(days=d)).replace(hour=hour % 24, minute=rng.randint(0, 50), second=0)
            sid = str(uuid.uuid4())
            late = hour >= 22
            for _ in range(rng.randint(10, 60)):
                p = min(0.99, max(0.01, rng.betavariate(2, 3) + (0.15 if late else 0)))
                flagged = p >= 0.6
                dur = rng.randint(7_000, 60_000)
                watch = int(dur * (rng.uniform(0.6, 2.2) if flagged else rng.uniform(0.1, 1.2)))
                rows.append((t, user, sid, f"demo{rng.randint(0, 10**9)}", "tiktok", watch, dur,
                             watch // dur, watch < 2000 or watch < 0.25 * dur, p, flagged))
                t += timedelta(milliseconds=watch + rng.randint(300, 1500))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO reel_views (time, user_id, session_id, video_id, platform, watch_ms, video_duration_ms, "
            "loops, skipped, prob, flagged) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
    print(f"inserted {len(rows)} synthetic views for user '{user}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--user", default="demo")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.days, a.user, a.seed)
