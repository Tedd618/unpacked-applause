import json

from unpacked_applause.captions import applause_labels, parse_json3


def test_parse_json3_finds_applause_tags(tmp_path):
    doc = {"events": [
        {"tStartMs": 1000, "dDurationMs": 3000, "segs": [{"utf8": "[ Applause and Cheering ]"}]},
        {"tStartMs": 5000, "dDurationMs": 2000, "segs": [{"utf8": "Thank you."}]},
        {"tStartMs": 8000, "dDurationMs": 1500, "segs": [{"utf8": "(MUSIC)"}]},
        {"tStartMs": 9500, "dDurationMs": 1000},
    ]}
    path = tmp_path / "x.en.json3"
    path.write_text(json.dumps(doc))
    tags = parse_json3(path)
    assert tags["tag"].tolist() == ["applause and cheering", "music"]
    assert tags["is_applause"].tolist() == [True, False]
    y = applause_labels(tags, 12)
    assert y.tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
