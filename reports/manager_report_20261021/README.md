# Manager report — 21 October 2026

Read [the complete written report](complete_report.md) for the three manager questions, eight exportable figures, methods, risk register and linked evidence. Open [the interactive report](app/dist/index.html) for chart controls, full comparison tables and inspectable source records.

The interactive preview is served locally at <http://127.0.0.1:8768/>. To restart it from this directory:

```sh
python -m http.server 8768 --bind 127.0.0.1 --directory app/dist
```

The report uses frozen run `reports/rte_risk/20261020T213640021260Z`. It adds descriptive diagnostics and presentation files; no models were refitted, no selection was changed, and no production system was implemented.

## Main findings

- The development-selected LightGBM RTE correction reduces historical MAE from 1,438.98 to 719.63 MW, a 49.99% improvement, over 8,759 matched hours.
- Cold-evening MAE improves 35.00% versus RTE, but uncertainty remains inadequate. Central 90% coverage is 43.94% for the selected point model's residual intervals, 53.03% for the separate quantile selection and 63.64% for fixed CQR. Critical support is 66 hours on 24 days in one winter.
- The candidate warrants an unseen, vintage-correct comparison and then shadow evaluation. These results do not justify replacing procurement or reserve decisions. Historical issue-time uncertainty, prior test exposure and national-to-portfolio mismatch remain material.

## Contents and reproduction

| File or directory | Purpose |
|---|---|
| `complete_report.md` | Full written report with eight embedded figures |
| `app/dist/index.html` | Built interactive report |
| `figures/` | Eight PNG figures plus matching vector SVG files |
| `diagnostics/` | Additional daily, monthly, local-hour and failure analyses with provenance |
| `reviewed_snapshot.json` | Source-backed input snapshot for the report app |
| `claim_audit.md` | Independent audit of defensible conclusions |
| `report_review.md` | Independent numerical and wording review |
| `verification.json` | Final report checks and artifact hashes |

From the project root, reproduce the additional diagnostics and figures using the existing research Python environment:

```sh
python reports/manager_report_20261021/diagnostics/build_diagnostics.py
python reports/manager_report_20261021/render_research_figures.py
```

The report app source is in `app/`; its existing `npm run build` command checks the shared runtime before building. Do not edit its protected runtime to change report content.

The written report's business figure evaluates **quantile-based decisions**. The interactive business chart evaluates **unchanged point forecasts under hypothetical asymmetric penalties**. Their different scores reflect different experiments. Neither measures euro savings.

Charts and tables retain separate identities for the selected point model, the selected quantile model and the fixed conformal comparators. The report does not combine their favorable metrics into a fictitious single system.

## Verification scope

The independent report review checked 6,318 numeric/Boolean values, 55 model definitions and 34 local written-report links. Desktop and 390-pixel layout checks cover the rendered report and representative controls. Figure and diagnostic provenance records contain source hashes. These checks establish report consistency; they cannot validate unavailable historical forecast vintages or guarantee future performance.
