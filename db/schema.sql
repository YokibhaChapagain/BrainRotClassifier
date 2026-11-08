-- Tiger Data / TimescaleDB schema for the BrainRot dashboard.
-- Run once:  psql "$DATABASE_URL" -f db/schema.sql

-- 1) Every reel a user watched (one row per view). This is the core time series.
--    We log watch time + skips now so "does brain rot make me scroll more?" can be analysed later.
CREATE TABLE IF NOT EXISTS reel_views (
    time              TIMESTAMPTZ NOT NULL,          -- when the view started
    user_id           TEXT        NOT NULL,
    session_id        TEXT        NOT NULL,          -- one scrolling session (resets after 10 min idle)
    video_id          TEXT        NOT NULL,
    platform          TEXT        NOT NULL,          -- tiktok | instagram
    watch_ms          INTEGER     NOT NULL,          -- how long the reel was on screen and playing
    video_duration_ms INTEGER,
    loops             SMALLINT    NOT NULL DEFAULT 0,
    skipped           BOOLEAN     NOT NULL,          -- left before 2 s or before 25 % of the video
    prob              DOUBLE PRECISION,              -- model P(brain rot); NULL if the API never answered
    flagged           BOOLEAN
);
SELECT create_hypertable('reel_views', by_range('time'), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS reel_views_user_time ON reel_views (user_id, time DESC);

-- 2) Hourly rollup per user (continuous aggregate = Timescale keeps it up to date automatically).
CREATE MATERIALIZED VIEW IF NOT EXISTS views_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time)                          AS bucket,
       user_id,
       count(*)                                             AS views,
       count(*) FILTER (WHERE flagged)                      AS rot_views,
       sum(watch_ms)                                        AS watch_ms,
       coalesce(sum(watch_ms) FILTER (WHERE flagged), 0)    AS rot_watch_ms,
       count(*) FILTER (WHERE skipped)                      AS skips,
       avg(prob)                                            AS avg_prob
FROM reel_views
GROUP BY bucket, user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('views_hourly',
  start_offset => INTERVAL '30 days', end_offset => INTERVAL '1 minute',
  schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE);
-- show the newest (not yet materialised) minutes too
ALTER MATERIALIZED VIEW views_hourly SET (timescaledb.materialized_only = false);

-- 3) Daily rollup built on top of the hourly one (hierarchical continuous aggregate).
CREATE MATERIALIZED VIEW IF NOT EXISTS views_daily
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', bucket) AS bucket, user_id,
       sum(views) AS views, sum(rot_views) AS rot_views,
       sum(watch_ms) AS watch_ms, sum(rot_watch_ms) AS rot_watch_ms, sum(skips) AS skips
FROM views_hourly
GROUP BY 1, user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('views_daily',
  start_offset => INTERVAL '90 days', end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '10 minutes', if_not_exists => TRUE);
ALTER MATERIALIZED VIEW views_daily SET (timescaledb.materialized_only = false);

-- 4) The labelled Hugging Face videos (time = date_posted): engagement of rot vs normal, rot over the years.
CREATE TABLE IF NOT EXISTS dataset_videos (
    time            TIMESTAMPTZ NOT NULL,
    video_id        TEXT        NOT NULL,
    creator         TEXT,
    is_ai_generated BOOLEAN,
    duration_ms     INTEGER,
    plays           BIGINT,
    likes           BIGINT,
    comments        BIGINT,
    shares          BIGINT,
    brainrot        SMALLINT    NOT NULL,   -- Gemini label 0/1
    score           SMALLINT,               -- Gemini score 0-10
    UNIQUE (video_id, time)
);
SELECT create_hypertable('dataset_videos', by_range('time', INTERVAL '90 days'), if_not_exists => TRUE);
