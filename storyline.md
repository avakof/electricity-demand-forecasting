# Presentation storyline

Audience: a head of trading (business manager), not a data scientist. No model
internals, no code, no library names, minimal jargon. Every number gets one
sentence of plain-language meaning next to it. Keep the whole deck to roughly
10 minutes of talking (plus 5 for Q&A), so each section below should map to a
small number of slides, not many.

## 1. The problem and the three questions

Open with the business situation, not the data: every day before the trading
desk's cutoff, we need tomorrow's hourly demand. Every forecast error gets
settled at a cost later, and the worst errors are on cold evenings. We were
asked three questions:

1. Can we beat RTE's own published forecast, and by how much?
2. How uncertain is our forecast, and when does it fail?
3. What should we deploy, and what's the risk?

One slide. State the three questions as the spine of the whole talk — tell
the audience we're going to answer each one now, then show our work, then
come back to them in more depth.

## 2. The headline answers (short form)

Answer all three immediately, in plain language, before any methodology:

- **Q1:** Yes. Our best model cuts the average forecast error roughly in
  half compared to RTE's own forecast (RTE ~1,439 MW average error; ours
  ~720 MW). That's about a 50% reduction.
- **Q2:** The forecast is noticeably less reliable on cold winter evenings —
  exactly the hours that cost the most when wrong. Our uncertainty ranges
  undershoot on those hours: we expect to cover the true outcome 90% of the
  time, but on cold evenings we're only actually covering it 44-64% of the
  time. We under-predict demand on about 3 out of 4 cold evenings.
- **Q3:** Don't replace the current process yet. Run this model alongside
  the existing one first (a "shadow" trial), not as a production
  replacement, until it's been tested on live data and a real customer
  portfolio. There's a short, named list of risks to close first.

One slide, three bullets, numbers only — no explanation of *how* yet. This
is the "if you remember nothing else" slide.

## 3. How we got there (the three-step process, quickly)

One slide, high level, no model names if avoidable (or name them once in
passing, not as the focus):

1. **Start simple:** build baseline forecasts and compare them honestly
   against RTE's own forecast, apples to apples, on the same held-out
   period.
2. **Add RTE's forecast as an input:** instead of competing with RTE from
   scratch, let the model learn from RTE's forecast and correct it —
   this is where the big accuracy jump happens.
3. **Add richer features and weight cold evenings more heavily:** expand
   what the model sees (weather, calendar, regional detail) and deliberately
   push it to pay more attention to the costly cold-evening hours.

Frame this as "each step bought us something specific" — don't dwell on
mechanics.

## 4. Walk through the process, showing where each answer came from

This is the longer middle section — now go back through the same three
steps, but this time show the evidence and connect each step to one of the
three questions:

- **Step 1 result → sets up Q1:** even simple methods already beat naive
  forecasting by a lot; RTE's own forecast is a strong baseline to clear.
  Show the baseline-vs-RTE comparison.
- **Step 2 result → answers most of Q1:** letting the model learn from RTE's
  forecast is where most of the improvement comes from (RTE ~1,439 MW →
  ~807 MW, roughly 44% better). Show the step-by-step improvement chart
  (this is a good place to reuse the leaderboard bar chart, relabeled in
  plain terms, with RTE and the naive baseline shaded so the audience has a
  fixed reference point).
- **Step 3 result → finishes Q1 and surfaces Q2:** the richer features push
  accuracy further (→ ~720 MW, ~50% better than RTE) — but this is also
  where the cold-evening weighting work happened, and it's what exposed the
  uncertainty problem. Show the overall-vs-cold-evening comparison, and the
  reliability/calibration chart (`09_jg_diagnostics.ipynb`, built and
  confirmed) that visualizes the undercoverage rather than just stating the
  percentage: calibration is good in the middle of the distribution, but
  both tails loosen, and cold-evening hours are far worse than the rest of
  the year at every level — e.g. the range meant to cover 97.5% of outcomes
  only actually covers about 70% of cold-evening outcomes.
- **What's actually driving the forecast:** one slide pulling the feature-
  importance chart (`09_jg_diagnostics.ipynb`) — in plain terms, the model's
  top signals are patterns in *how RTE's own forecast tends to be wrong*
  (how choppy RTE's forecast is that day, how it compares to the recent
  weekly pattern, how it's been drifting hour to hour) rather than raw
  weather or demand history. Plain-language framing: "the model's main job
  is learning RTE's blind spots, not re-predicting demand from scratch" —
  this also explains why it tracks RTE so closely most of the time but
  struggles exactly when RTE's own pattern breaks down (cold evenings).
- **Independent check:** mention once, briefly, that the final numbers were
  independently re-verified against the raw actuals before being trusted —
  this builds credibility without needing to explain the mechanics.

Keep technical detail here to "what we tried and what moved the number," not
"how the model works."

## 5. Answers again, in more depth

Close by returning to the three questions, now with enough depth to support
Q&A, still in business language:

- **Q1, in depth:** give the number with its honest range (confidence
  interval), and name what it's measured against (one full year, matched
  hour-by-hour against RTE's actual published forecast — not a cherry-picked
  period).
- **Q2, in depth:** be direct that this is the weak point, not a hidden one.
  Cold evenings are rare in the test data (66 hours in one winter), so the
  uncertainty estimate itself is less trustworthy exactly where it matters
  most — name this as a data-availability limitation, not just a model
  limitation. The calibration chart from section 4 is the strongest visual
  here: calibration is fine overall, but it degrades specifically and
  substantially on the hours that matter most financially.
- **Q3, in depth:** walk through the short risk list in plain terms (a
  sentence per risk, no jargon): things like — the comparison uses
  historical data that may not perfectly match what was available in real
  time; the model hasn't been tested on a live, real customer portfolio;
  cold-evening performance needs more data before we trust it; and any
  financial savings claim needs real prices and a trading decision rule
  layered on top, which hasn't been done yet. End on the concrete
  recommendation: shadow deployment, revisit after a winter of live data.

End on the recommendation, not on a caveat — the last thing the audience
hears should be the decision, with the risks already digested earlier in
this section.

## Notes for the slide-building agent

- Translate every MW figure into one plain sentence ("about half the
  error") in addition to the number — never show a bare number without a
  plain-language gloss nearby.
- No equations, no library/model-family jargon beyond naming "RTE's
  forecast" and "our model" — if a specific technique must be named, name it
  once, in passing, and move on.
- Reuse existing charts from `notebooks/10_jg_final_comparison.ipynb` and
  `notebooks/09_jg_diagnostics.ipynb` (both finished and verified) — don't
  regenerate numbers from scratch, pull the images/data directly from what
  these notebooks already produced and verified.

### Chart reference (which chart, from where, goes on which slide)

From `notebooks/10_jg_final_comparison.ipynb`:

1. **MAE leaderboard, horizontal bar chart** (baseline model shaded light
   grey) — use on section 2 (short answers, Q1) as the "if you remember
   nothing else" visual, and again in section 4 step 2/3 to show the
   step-by-step improvement. Relabel model names in plain terms for the
   business audience (e.g. "RTE's forecast," "our model," not internal
   family names).
2. **% improvement vs RTE, horizontal bar chart** — section 4, to show each
   step's incremental gain over the RTE benchmark.
3. **Actual-vs-forecast time series, sample week (5–11 Jan 2026)** — strong
   visual proof the model actually tracks real demand, not just a summary
   statistic. Good for section 2 or the opening of section 4, before the
   audience has seen any error numbers.
4. **Error distribution histogram** (best candidate vs RTE) — section 5,
   Q1 in depth (shows the error spread, not just the average) or Q2 (sets
   up the idea that errors aren't symmetric/uniform).
5. **Overall-vs-cold-evening MAE grouped bar chart** — section 4 step 3 and
   section 5 Q2 in depth; this is the chart that first shows the model's
   cold-evening weak spot in overall terms (pair it with chart 7 below for
   the fuller picture).

From `notebooks/09_jg_diagnostics.ipynb`:

6. **Feature-importance horizontal bar chart** (top 15–20 features for the
   winning model) — section 4 "what's actually driving the forecast," and
   can resurface in section 5 Q1 in depth to explain *why* the model works
   the way it does (it's correcting RTE's own forecast, not re-predicting
   demand from scratch).
7. **Reliability/calibration diagram — overall** — section 4 step 3, to
   show calibration is basically fine across the whole year.
8. **Reliability/calibration diagram — cold-evening only** — the single
   most important chart for Q2: pair directly with chart 7 (same chart
   type, restricted to the 66 cold-evening hours) so the audience can see
   the calibration gap, not just read a percentage. Use in section 4 step 3
   and again in section 5 Q2 in depth as the main supporting visual for
   "this is the honest weak point."
9. **ACF/PACF plots (SARIMAX residuals)** — optional, Q&A backup slide
   only, not in the main deck; pull out only if asked about residual
   diagnostics specifically.
10. **Ljung-Box test result** — not a chart (just the printed statistic/
    p-value) — mention only verbally in Q&A if asked, no slide needed.

General rule for the slide-building agent: prefer reusing more of these
charts rather than fewer — a business audience absorbs one honest chart
faster than a paragraph of caveats, so default to including charts 1-8 on
the main deck (not just a token 2-3) and reserve only 9-10 for backup.
- The three-question spine (section 1) should visually reappear at section 2
  (short answers) and section 5 (long answers) so the audience always knows
  which question is currently being answered.
- The project's course also expects a short local-vs-global modeling note
  (a global model was in fact built — see the neural experiment track) but
  this was not asked for in the business brief and does not need its own
  slide; one sentence in passing (e.g. in Q&A backup material) is enough if
  the instructor asks, not a dedicated section of the main deck.
