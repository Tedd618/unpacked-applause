# Narrative sections of the report (rendered with Jinja; numbers come from reports/tables)

## abstract
Technology keynotes are staged to create enthusiasm, and their loudest moments are replayed
across financial media. This project asks whether that enthusiasm tells investors anything.
I catalogue {{ results.n_keynotes }} keynotes held by nine large technology companies between
2010 and 2026, measure audience applause second by second from the broadcast audio with a
pretrained sound-event detection network (validated against human caption tags; held-out
ROC-AUC {{ "%.2f"|format(validation.test_auc_pooled) }}), and relate within-company variation
in applause to daily abnormal returns. In a pre-registered specification over
{{ results.n_live_sample }} live-audience keynotes, one standard deviation more applause goes
with a change of {{ "%+.3f"|format(results.primary.beta) }} standard deviations in the two-day
standardised abnormal return (95% CI {{ "%+.2f"|format(results.primary.ci_lo) }} to
{{ "%+.2f"|format(results.primary.ci_hi) }}; wild cluster bootstrap p =
{{ fmt_p(results.primary.p_wcb) }}). No secondary outcome, alternative applause measure,
robustness check or placebo changes this conclusion: applause is not a return signal.
Keynotes themselves barely move prices on average, but the stocks rise
{{ pct(A["All keynotes|[-10,-1]"].mean, 1) }} over the ten sessions before them.

## intro
Every year Apple, Samsung, Google, NVIDIA and their peers spend heavily on keynotes that are
built around a handful of ovation moments: the "one more thing", the price reveal, the demo
that works. Clips of the loudest reactions circulate within minutes, and commentators read
the room as a verdict on the products. For an investor the question is simple: does the
room know something the market does not? If audience enthusiasm reflects how compelling the
announcements are, louder keynotes should be followed by higher abnormal returns. If it is
staged hype, the loudest keynotes may mark expectations that are already priced, and they
could even precede disappointment ("buy the rumour, sell the news"). If it is noise—a
function of the venue, the audience and the audio mix—there should be no relation at all.

Answering this needs three things that do not exist off the shelf: a complete list of
keynotes with precise start times (whether the market was open decides which day reacts),
a reliable measure of applause for each keynote, and an event-study design that separates
the keynote's effect from the market's. This project builds all three: a hand-curated
catalogue of {{ results.n_keynotes }} keynotes; an audio pipeline that turns
{{ results.n_with_video }} broadcast recordings into second-level applause timelines and is
validated against human-written caption tags; and a daily event study with a
pre-registered cross-sectional test and inference that respects the small number of
companies. It also treats the pandemic years, when keynotes went virtual, as a
no-applause comparison group.

## avg
Keynotes move prices less than their reputation suggests. Across all
{{ A["All keynotes|[0,+1]"].n }} keynotes the mean two-day abnormal return is
{{ pct(A["All keynotes|[0,+1]"].mean) }} (95% CI {{ pct(A["All keynotes|[0,+1]"].ci_lo) }} to
{{ pct(A["All keynotes|[0,+1]"].ci_hi) }}), and fewer than half are positive. The build-up is
where the action is: in the ten sessions before a keynote the stocks earn
{{ pct(A["All keynotes|[-10,-1]"].mean) }} of abnormal return (BMP p
{{ fmt_p(A["All keynotes|[-10,-1]"].p_bmp) }}; {{ "%.0f%%"|format(100*A["All keynotes|[-10,-1]"].pos_share) }}
positive), and after the keynote the cumulative path flattens and drifts down (Figure 4).
The pattern is the textbook shape of anticipation followed by a muted announcement.

Company averages differ. Apple's keynotes are followed by a mean two-day abnormal return of
{{ pct(C["Apple"].mean) }} (p = {{ fmt_p(C["Apple"].p_t) }}; only
{{ "%.0f%%"|format(100*C["Apple"].pos_share) }} positive), and Tesla's by
{{ pct(C["Tesla"].mean) }} (p = {{ fmt_p(C["Tesla"].p_t) }}), in line with the
"sell-the-news" folklore around both companies. These per-company averages are descriptive:
nine companies are tested, and Tesla's worst reactions coincide with other news such as
delivery reports released over the same weekend.

## primary
The pre-registered test finds no relation between applause and the stock's reaction. The
coefficient on the within-company z-score of applause share is
{{ "%+.3f"|format(results.primary.beta) }} (HC3 95% CI {{ "%+.3f"|format(results.primary.ci_lo) }}
to {{ "%+.3f"|format(results.primary.ci_hi) }}; wild cluster bootstrap p =
{{ fmt_p(results.primary.p_wcb) }}; N = {{ results.primary.n }} keynotes in
{{ results.primary.n_clusters }} companies). The within-company rank correlation is
{{ "%+.2f"|format(results.spearman.rho) }} (permutation p = {{ fmt_p(results.spearman.p_perm) }}).

The estimate is not just insignificant; it is small and precise. Because the outcome is
standardised, the interval says that one standard deviation more applause shifts the
two-day abnormal return by at most about {{ "%.2f"|format([results.primary.ci_hi, -results.primary.ci_lo]|max) }}
residual standard deviations—roughly {{ "%.1f"|format(100 * typ_sd2 * [results.primary.ci_hi, -results.primary.ci_lo]|max) }}
percentage points for a typical stock in the sample. The secondary families agree (Table 2):
neither the direction of the move, its size, the post-event drift nor abnormal trading
volume is related to applause, and neither are the alternative measures (the smallest
Benjamini–Hochberg q-value is {{ "%.2f"|format(sec_min_q) }}).

## robust
Sorting keynotes into within-company terciles of applause shows no monotone pattern: the
loudest third minus the quietest third is {{ "%+.2f"|format(results.terciles.diff) }} standard
deviations (95% CI {{ "%+.2f"|format(results.terciles.ci_lo) }} to
{{ "%+.2f"|format(results.terciles.ci_hi) }}; Figure 6). Across every alternative
normal-return model, control set and subsample—including the exploratory exclusions of
keynotes near an earnings release and of keynotes with uncertain start times—the coefficient
stays between {{ "%+.2f"|format(rob_beta_range[0]) }} and {{ "%+.2f"|format(rob_beta_range[1]) }},
with no bootstrap p-value below {{ "%.2f"|format(rob_min_p) }} (Figure 7).

The placebo test pairs each keynote's applause with the stock's reaction on a random
non-event day of the same company. The actual coefficient sits in the middle of the placebo
distribution ({{ "%.0f%%"|format(100*results.placebo.p_placebo) }} of placebo draws are at least
as extreme; Figure 8), which is what one would expect if applause carried no information
about keynote-day returns. Company by company (Figure 9) the slopes scatter around zero with
no company showing a clear relation.

## lvv
Virtual keynotes offer a crude natural experiment: no audience, no applause. They are
followed by reactions that are statistically indistinguishable from live keynotes in
direction, size and trading volume (Table 3). Because virtual keynotes cluster in 2020–2022,
this comparison also carries the pandemic market, so it is descriptive only.

## discussion
Why is the applause signal empty? Three explanations fit the evidence. First, keynote
audiences are not a random sample of investors: they are employees, developers, partners and
fans, and companies script the moments that invite applause, so the volume of the room
reflects the production as much as the products. Second, the information that matters—
specifications, prices, launch dates, AI capabilities—is public the moment it is spoken and
is analysed by professionals within minutes; the room's reaction adds little beyond it.
Third, much of the news is anticipated: the significant pre-keynote run-up suggests that
expectations are formed, and priced, before anyone claps.

The null is informative because the design had power. With {{ results.n_live_sample }} live
keynotes, the confidence interval excludes effects that would matter for trading, and the
detector's accuracy against human labels makes measurement error an unlikely explanation.

The study has limits. Daily returns blur what happens during the keynote itself; minute-level
prices around keynotes held during market hours would sharpen the test. Applause is measured
on published videos whose audio mix varies across productions; within-company
standardisation and the official-channel robustness check address this but cannot remove
it. Caption tags are an imperfect ground truth (one Apple caption track does not line up with
its video at all). Some start times are uncertain, some event windows overlap for the same
company, and other news occasionally lands inside the window. Finally, the sample is nine of
the most liquid stocks in the world, where short-lived signals are least likely to survive;
smaller companies' launch events might behave differently.

## conclusion
Audience applause at tech keynotes is easy to measure accurately from audio, but it does not
predict how the company's stock reacts: louder keynotes are followed neither by higher nor by
lower abnormal returns, nor by larger moves or heavier trading. The market's reaction to
keynotes happens mostly before them. For practitioners the result is a useful negative:
the loudest clip of a keynote is marketing, not a signal.
