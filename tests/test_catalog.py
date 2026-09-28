import pandas as pd

from unpacked_applause.catalog import COLUMNS, validate


def row(**kw):
    base = dict.fromkeys(COLUMNS, "")
    base.update(event_id="apple_2017-09-12_iphone-x", company="apple", ticker="AAPL",
                event_type="product_launch", date_local="2017-09-12", start_time_local="10:00",
                tz="America/Los_Angeles", start_time_confidence="high", audience="live",
                youtube_id="abcdefghijk", youtube_duration_s="7000", official_channel="true")
    base.update(kw)
    return base


def test_valid_row_passes():
    assert validate(pd.DataFrame([row()])) == []


def test_validation_catches_common_mistakes():
    df = pd.DataFrame([
        row(event_id="apple_2017-09-12_a", tz="Pacific Time"),
        row(event_id="apple_2017-09-12_b", youtube_id="short", start_time_local="10am"),
        row(event_id="meta_2011-04-01_f8", company="meta", ticker="META", date_local="2011-04-01"),
    ])
    problems = "\n".join(validate(df))
    assert "unknown IANA time zone" in problems
    assert "malformed youtube_id" in problems
    assert "start_time_local not HH:MM" in problems
    assert "event before IPO" in problems
