# Event catalog

`events.csv` is the hand-curated universe of company keynotes analysed in this project.
It is assembled from per-company research files in `parts/` and validated by
`applause catalog` (see `src/unpacked_applause/catalog.py`).

## Scope

* Companies: Apple, Samsung Electronics, Alphabet (Google), Microsoft, Meta, Amazon,
  NVIDIA, Tesla, AMD.
* Dates: 2010-01-01 to 2026-08-31, and never before the company's IPO
  (Meta 2012-05-18, Tesla 2010-06-29).
* Events: the company's own main keynote (CEO / lead executive on stage or on video):
  product launches, developer conferences, enterprise conferences, investor events and
  the company's keynote slot at trade shows (CES, Computex). Gaming showcases (E3),
  earnings calls and interviews are out of scope.
* Virtual and pre-recorded keynotes are **kept**: they have no audience and act as a
  no-applause comparison group.

## Columns

| column | type | description |
|---|---|---|
| `event_id` | str | `<company>_<YYYY-MM-DD>_<slug>`, lowercase, e.g. `apple_2017-09-12_iphone-x` |
| `company` | str | `apple`, `samsung`, `google`, `microsoft`, `meta`, `amazon`, `nvidia`, `tesla`, `amd` |
| `ticker` | str | Yahoo Finance ticker (`AAPL`, `005930.KS`, `GOOGL`, `MSFT`, `META`, `AMZN`, `NVDA`, `TSLA`, `AMD`) |
| `event_name` | str | Official or commonly used name |
| `series` | str | Recurring series (`wwdc`, `apple_event`, `galaxy_unpacked`, `google_io`, `made_by_google`, `google_hardware`, `build`, `ignite`, `microsoft_devices`, `f8`, `connect`, `reinvent`, `amazon_devices`, `gtc`, `ces`, `computex`, `amd_ai`, `amd_launch`, `tesla_product`, `tesla_ai`, `tesla_shareholder`, `investor_day`, `other`) |
| `event_type` | str | `product_launch`, `developer_conference`, `enterprise_conference`, `investor_event` |
| `headline_products` | str | Main announcements, `;`-separated |
| `date_local` | date | Local calendar date at the venue / broadcast origin |
| `start_time_local` | HH:MM | Scheduled keynote start, 24h local time |
| `tz` | str | IANA time zone of `start_time_local` (e.g. `America/Los_Angeles`) |
| `start_time_confidence` | str | `high` (official invite, press release or stream timestamp), `medium` (news reports), `low` (inferred) |
| `city` | str | City of the venue (or `online`) |
| `venue` | str | Venue name (or `online`) |
| `audience` | str | `live` (in-person audience audible in the stream), `virtual` (no in-person audience; pre-recorded or empty room), `hybrid` (small / partial audience or mixed pre-recorded segments) |
| `audience_notes` | str | One-line justification for `audience` |
| `youtube_id` | str | 11-character id of a full-length keynote video; empty if none exists |
| `youtube_title` | str | Title as returned by yt-dlp |
| `youtube_channel` | str | Channel name as returned by yt-dlp |
| `youtube_duration_s` | int | Duration in seconds as returned by yt-dlp |
| `youtube_upload_date` | YYYYMMDD | Upload date as returned by yt-dlp |
| `official_channel` | bool | `true` if uploaded by the company's own channel |
| `vod_notes` | str | Caveats about the video (countdown pre-roll, trimmed, re-upload, etc.) |
| `sources` | str | `;`-separated URLs backing date, time and audience facts |
