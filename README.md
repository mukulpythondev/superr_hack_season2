# superr-selector

Who delivers under pressure in T20 cricket? My Round 1 entry for SupeRR Selector Season 2
(Rajasthan Royals x Unstop). Unofficial.

Pressure is defined from match situations, not from a model: collapse, chase start, big chase and
finish for batters; death overs, a set batter, a small total and the ball after a boundary for
bowlers. Each ball is compared with what an average player does in the same state, and thin
samples are pulled toward the average. Details in [docs/METHOD.md](docs/METHOD.md).

## Top 10

| | Batters | Bowlers |
|---|---|---|
| 1 | Suryakumar Yadav | Jasprit Bumrah |
| 2 | Heinrich Klaasen | Varun Chakravarthy |
| 3 | Tim David | Wanindu Hasaranga |
| 4 | Sai Sudharsan | Sunil Narine |
| 5 | KL Rahul | Matheesha Pathirana |
| 6 | Virat Kohli | Nathan Ellis |
| 7 | Andre Russell | Kuldeep Yadav |
| 8 | Pathum Nissanka | Josh Hazlewood |
| 9 | Tilak Varma | Rashid Khan |
| 10 | Nicholas Pooran | Arshdeep Singh |

Full tables are in `outputs/open_pool/`. An interactive version is in `website/index.html`.

## Run

The data pack is not included. Put it in `data/raw/Data Pack/` or set `SUPERR_DATA`.

    pip install -r requirements.txt
    python -m superr_selector.run_pipeline      # about 6 minutes
    python evals/repeatability.py
    python website/export_data.py && python website/build.py

## Layout

    superr_selector/   pipeline, one module per step (see run_pipeline.py)
    evals/             repeatability, win-probability audit, venue check
    outputs/           rankings and diagnostics
    website/           single-page site built from the outputs
    docs/METHOD.md     method and limits
