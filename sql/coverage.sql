-- Catalogue coverage by company: how many keynotes, how many with a usable video,
-- and how the sample splits by audience type.
SELECT
    company,
    count(*)                                                    AS keynotes,
    count(*) FILTER (WHERE youtube_id <> '')                    AS with_video,
    count(*) FILTER (WHERE audience = 'live')                   AS live,
    count(*) FILTER (WHERE audience = 'hybrid')                 AS hybrid,
    count(*) FILTER (WHERE audience = 'virtual')                AS virtual,
    min(date_local)                                             AS first_event,
    max(date_local)                                             AS last_event,
    round(median(youtube_duration_s) / 60)                      AS median_video_min
FROM events
GROUP BY company
ORDER BY keynotes DESC;
