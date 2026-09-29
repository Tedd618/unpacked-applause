-- Analysis panel: one row per keynote.
-- analysis.py registers four inputs before running this file:
--   events   (data/catalog/events.csv)
--   features (data/processed/applause_features.parquet)
--   study    (data/processed/event_study.parquet)
--   vix      (date, vix_close) daily CBOE VIX closes

CREATE OR REPLACE TABLE panel AS
SELECT
    e.*,
    f.* EXCLUDE (event_id),
    s.* EXCLUDE (event_id),
    v.vix_close AS vix_prev          -- VIX close on the last session before t0
FROM events AS e
LEFT JOIN features AS f USING (event_id)
LEFT JOIN study AS s USING (event_id)
ASOF LEFT JOIN vix AS v ON s.t0 > v.date;

-- Main sample: live-audience keynotes with applause measures and a primary outcome.
-- Applause measures are standardised within company so that company fixed effects and
-- company-specific audio mixing do not drive the comparison.
CREATE OR REPLACE TABLE live AS
SELECT
    *,
    (applause_share - avg(applause_share) OVER w)
        / nullif(stddev_samp(applause_share) OVER w, 0)                AS z_applause_share,
    (applause_per_min - avg(applause_per_min) OVER w)
        / nullif(stddev_samp(applause_per_min) OVER w, 0)              AS z_applause_per_min,
    (ln(1 + applause_max_s) - avg(ln(1 + applause_max_s)) OVER w)
        / nullif(stddev_samp(ln(1 + applause_max_s)) OVER w, 0)        AS z_applause_max,
    (clap_mass - avg(clap_mass) OVER w)
        / nullif(stddev_samp(clap_mass) OVER w, 0)                     AS z_clap_mass,
    (cheer_share - avg(cheer_share) OVER w)
        / nullif(stddev_samp(cheer_share) OVER w, 0)                   AS z_cheer_share,
    count(*) OVER w                                                    AS n_company_live
FROM panel
WHERE audience = 'live'
  AND applause_share IS NOT NULL
  AND scar_0_1_market IS NOT NULL
WINDOW w AS (PARTITION BY company);
