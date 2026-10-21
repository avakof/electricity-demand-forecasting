# Limitations

- Supplied CSV provenance, hourly averaging, interval labels, and revision vintages are unverified. Interval-start labels and one-hour mean-power values are explicit assumptions. Daily MWh is conditional on these assumptions.
- The RTE documentation describes quarter-hourly source values and subsequent consolidation/definitive revisions. No source-to-hourly conversion recipe was provided; matching timestamps do not prove matching intervals.
- No trustworthy archived RTE forecast issues, weather runs, or original demand observation vintages were supplied. Original observation availability is approximated as interval end plus two hours in retrospective runs only.
- Provided RTE values are an explicitly assumed-cutoff retrospective benchmark. Recorded future temperature is restricted to separately named oracle sensitivity experiments. No operational superiority is established.
- Earlier conversation and legacy EDA inspected the full history. The chronological test period is previously exposed, not pristine. Freezing this implementation now cannot undo that exposure; fresh prospective evaluation is required.
- Missing targets remain missing and are counted. Peaks and the country/regional discrepancy are retained and flagged; source verification is needed before correction.
- National consumption is a utility portfolio proxy. National public holidays omit school holidays and local exceptions. No customer portfolio, price, settlement or procurement data are provided.
- Forecast skill and shortage/surplus energy proxies are not euro savings. Hourly forecasting does not implement quarter-hour market or settlement alignment.
