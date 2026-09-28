# Analysis plan (pre-registered)

Written on 2026-09-28 and committed **before any applause measure was joined to stock
returns**. The git history of this file is the time stamp. Anything reported later that is
not listed here is labelled *exploratory*.

## Question

Is the intensity of audience applause during a company's keynote associated with the
stock's abnormal return around the keynote?

Three stories predict different signs:

1. **Signal** – applause reveals how compelling the products are, so more applause predicts
   higher abnormal returns (β > 0).
2. **Hype / sell-the-news** – applause is staged enthusiasm; big ovations mark priced-in hype
   and are followed by flat or negative returns (β ≤ 0, possibly a post-event reversal).
3. **Noise** – applause reflects the venue, the audio mix and the audience, not
   information (β = 0).

## Sample

* Events in `data/catalog/events.csv` with a full-length keynote video.
* Main sample: `audience == "live"`, at least 120 returns in the estimation window and a
  complete [0, +1] window.
* Virtual and hybrid keynotes are used only as detector negative controls and in the
  descriptive live-vs-virtual comparison.
* No event is dropped because of its return.

## Event-time conventions

* `t0` = first trading session whose close is after the scheduled keynote start (exchange
  calendars from `exchange_calendars`, including early closes and KRX's 2016 close-time
  change). A keynote during US market hours therefore has t0 = the keynote day; an evening
  keynote, or any Samsung keynote held in the US or Europe, has t0 = the next session.
* Normal returns: market model estimated on trading days [-250, -30] relative to t0 with the
  S&P 500 as the market for US listings and KOSPI for Samsung Electronics (005930.KS).
* SCAR = CAR / (σ̂ε · √L).

## Applause measurement

* Audio: YouTube audio track of the keynote video, decoded to 32 kHz mono.
* Detector: PANNs `Cnn14_DecisionLevelMax` (AudioSet, 527 classes), per-second
  probabilities. Clap score = max(P(Applause), P(Clapping)).
* Keynote span: the speech-dominated stretch of the video (removes countdown pre-rolls and
  post-show music).
* Episodes: hysteresis thresholding of the clap score. Thresholds are tuned to maximise
  episode-level F1 against **human-written caption tags** (`[Applause]`, `[Cheering]`, ...) on a
  development half of the caption-labelled videos and evaluated on the other half. Returns
  are never used for tuning.
* **Primary measure:** `applause_share` = share of keynote seconds inside applause episodes,
  standardised within company (company fixed effects absorb company style and audio-mix
  levels).
* Secondary measures: applause episodes per minute, longest ovation (seconds),
  mean clap probability (`clap_mass`, threshold-free), cheering share, and a composite
  *hype index* (first principal component of the z-scored measures above, signed to
  increase with `applause_share`).

## Primary specification

    SCAR[0,+1]_i = α_company(i) + β · z(applause_share)_i + ε_i

* Two-sided test of β = 0 at α = 0.05 using the wild cluster restricted bootstrap
  (Webb weights, 9,999 draws, clusters = companies). HC3 standard errors reported alongside.

## Secondary analyses

Each family is corrected with Benjamini–Hochberg at q = 0.10.

* **Outcome family** (regressor = primary measure): SCAR[0,0], SCAR[-1,+1], SCAR[0,+5],
  SCAR[+2,+20], |SCAR[0,+1]| (size of the move) and standardised abnormal log volume [0,+1].
* **Measure family** (outcome = SCAR[0,+1]): each secondary applause measure.

## Robustness (reported whatever the result)

1. Normal-return models: market-adjusted, constant-mean, Fama–French 3-factor (US listings),
   Nasdaq-100 as market.
2. Controls: pre-event run-up CAR[-10,-1], log VIX on t0−1, keynote length, official vs
   third-party video, event type.
3. Samples: without Tesla, without Samsung, product launches only.
4. Non-parametric: within-company Spearman correlation with a permutation p-value; tercile
   sort (top minus bottom tercile of within-company applause).
5. Placebo: keep each event's applause but move it to a random non-event trading day of the
   same company (1,000 draws) to obtain the distribution of β when no keynote happened.

## Descriptive analyses

* CAAR paths over [-10, +20] for all keynotes, by company and by audience type.
* Average event effects with cross-sectional t, BMP and generalised sign tests.
* Live vs virtual: SCAR[0,+1], |SCAR[0,+1]| and abnormal volume (Welch t). Virtual events
  are concentrated in 2020–2022, so this comparison is confounded by the pandemic era and is
  reported as descriptive only.

## Detector validation

* Second-level ROC-AUC and average precision against human caption tags; episode-level
  precision, recall and F1 with a ±2 s tolerance.
* Negative control: virtual keynotes should show near-zero applause.

## Power

With about 120 live keynotes, a two-sided 5% test has 80% power for a correlation of
|ρ| ≈ 0.25. Smaller effects cannot be distinguished from zero, and a null result should be
read as "no effect of that size", not "no effect".
