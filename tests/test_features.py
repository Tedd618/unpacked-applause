import numpy as np
import pandas as pd

from unpacked_applause.features import EpisodeParams, episodes, event_features, keynote_span, runs


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert runs(np.zeros(3, bool)) == []


def test_episodes_hysteresis_merge_and_min_duration():
    s = np.zeros(40)
    s[5:10] = [0.3, 0.6, 0.7, 0.4, 0.3]  # reaches on=0.5 -> episode 5..10
    s[11:13] = 0.35  # gap of 1 s, below `on` alone but merged into the previous episode
    s[20:24] = 0.4  # never reaches `on` -> dropped
    s[30] = 0.9  # single second -> shorter than min_dur
    eps = episodes(s, EpisodeParams(on=0.5, off=0.25, min_dur=2, merge_gap=1))
    assert eps == [(5, 13)]


def make_timeline(n=1200, speech_from=200, speech_to=1100, claps=((300, 305), (600, 612))):
    cols = ["speech", "applause", "clapping", "cheering", "crowd", "chatter", "hubbub",
            "laughter", "whoop", "yell", "music", "horn", "silence"]
    tl = pd.DataFrame(0.0, index=range(n), columns=cols)
    tl.insert(0, "t", np.arange(n))
    tl.insert(1, "rms_db", -30.0)
    tl.loc[speech_from:speech_to - 1, "speech"] = 0.9
    for a, b in claps:
        tl.loc[a:b - 1, ["applause", "speech"]] = [0.8, 0.1]
        tl.loc[a:b - 1, "rms_db"] = -20.0
    return tl


def test_keynote_span_trims_countdown_and_outro():
    s0, s1 = keynote_span(make_timeline())
    assert 170 <= s0 <= 200 and 1100 <= s1 <= 1130


def test_event_features_counts_applause():
    f = event_features(make_timeline(), EpisodeParams(on=0.5, off=0.25))
    assert f["n_applause"] == 2
    assert f["applause_max_s"] == 12
    assert f["applause_rel_db"] == 10.0
    assert 0 < f["applause_share"] < 0.03
