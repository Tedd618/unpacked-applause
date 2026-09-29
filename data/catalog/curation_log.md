# Curation log

Manual decisions applied to the research files in `parts/` after review. Each entry says
what changed and why, so the catalogue can be audited.

## Tesla

* Dropped `tesla_2025-09-08_las-megas`: a product walkthrough published as an edited video
  nine days later, not a keynote with a live feed.
* Dropped `tesla_2026-03-21_terafab`: joint Tesla/SpaceX/xAI event; only an edited
  "presentation only" upload exists.
* `tesla_2011-10-01_model-s-beta`: kept as an event, but its only full video is an audience
  camcorder recording whose floor microphone exaggerates applause relative to broadcast
  feeds, so `youtube_id` was cleared (the event stays in the average-effect sample only).

## Google

* `google_2016-10-04_pixel`: switched from CNET's livestream archive (`eyq0p7kvpG4`, ~35 min
  of CNET hosts before Google's feed) to Google's own VOD (`q4y0KOeXViI`), which ends ~13
  minutes early. A truncated official feed biases a per-minute applause rate far less than
  half an hour of studio chatter inside the speech-based keynote span.

## Samsung

* `samsung_2010-03-23_galaxy-s`: no start time was found; 10:00 local (Las Vegas) is assumed.
  Any time before 13:00 PT maps to the same US session, and Samsung trades in Seoul, where the
  next session is the same either way.
* `samsung_2011-03-22_galaxy-tab-ctia`: the only video is a recording made from inside the
  room, not the broadcast feed; cleared for the same reason as Tesla's 2011 camcorder video.
* Several videos list YouTube auto-dubbed audio tracks. The downloader excludes tracks whose
  format note says "dubbed" and records the track it used in `data/raw/audio/<event>.json`.

## Audio tracks (all companies)

* `data/raw/audio/_track_check.csv` re-resolved the track of every downloaded keynote: 22
  videos carry YouTube auto-dubbed tracks and all 22 resolved to the original English track;
  no dubbed and no DRC ("stable volume") track was used.
* `meta_2016-04-12_f8` (SiliconANGLE re-upload) is tagged `vi` by its uploader; the audio is
  the English keynote. `tesla_2022-03-22_giga-berlin-opening` is tagged `de` because the
  ceremony includes German speeches.

## Apple

* Dropped `apple_2010-07-16_iphone-4-press-conference` (the "Antennagate" press conference is
  crisis communication, not a keynote) and `apple_2026-03-04_apple-experience` (a few minutes
  of remarks before a hands-on; no recording).
* Seven Apple keynotes close with a music performance (Foo Fighters, Elvis Costello, U2,
  The Weeknd, OneRepublic, Sia, Lana Del Rey). Rather than hand-trimming, `keynote_span` ends
  the span where a run of >= 120 s of music-dominated audio starts in the last quarter of
  the keynote; the rule applies to every company.

## Microsoft

* Dropped `microsoft_2022-10-12_ignite-2022`: same company and same trading day as the
  (also virtual) Surface event, which would count one stock reaction twice.
* `microsoft_2021-03-02_ignite-spring-2021`: the only video is the 11-hour all-day stream, so
  no video is used (virtual event; it only enters the average-effect sample).

## NVIDIA and AMD

* NVIDIA CES press events 2010, 2011, 2012 and 2016 had no documented start time; 18:00 local
  (Las Vegas) is assumed because NVIDIA's CES press events were held in the evening. Flagged
  `start_time_confidence = low`.
* `nvidia_2017-05-30_computex-2017`: the only video may carry commentary over the feed, so no
  video is used.
* `amd_2017-06-20_epyc-launch`: no start time or video; 10:00 local assumed (average-effect
  sample only).
* AMD events whose in-person audience could not be confirmed are classed `hybrid`, which keeps
  them out of the live-audience sample.
