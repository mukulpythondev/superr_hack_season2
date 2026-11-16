"""Write website/data.json from the pipeline outputs. Run from the repo root."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from superr_selector.config import INTERIM, OPEN_POOL

TOP_N = 30
BAT_SITS = {"Collapse": "A_collapse_level", "Big chase": "B_big_chase_level", "Finish": "C_finish_level", "Chase start": "I_chase_start_level"}
BOWL_SITS = {"Death": "E_death_level", "Set batter": "F_set_batter_level", "Small total": "G_small_total_level", "Bounce-back": "H_bounce_back_level"}


def r1(x):
    return None if pd.isna(x) else round(float(x), 1)


def chase_features(runs_needed, balls_left, wickets):
    d = pd.DataFrame({"runs_needed": runs_needed, "balls_left": balls_left, "wickets_in_hand": wickets})
    d["required_rr"] = (d.runs_needed / d.balls_left.clip(lower=1) * 6).clip(upper=36)
    d["wk_x_time"] = d.wickets_in_hand * d.balls_left / 120
    d["runs_per_wk"] = d.runs_needed / d.wickets_in_hand.clip(lower=1)
    d["sqrt_balls"] = np.sqrt(d.balls_left.clip(lower=0))
    return d


def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
    tt = bbb[bbb.top_tier]
    out = {}

    rb = pd.read_csv(os.path.join(OPEN_POOL, "ratings_batters.csv"))
    pid = rb.columns[0]
    n_bat = len(rb)
    rb = rb[rb.evidence_gate].head(TOP_N)
    p = tt[tt.S_bat_any & tt.is_ball_faced]
    st = p.groupby("striker_id").agg(balls=("runs", "size"), runs=("runs", "sum"), outs=("striker_out", "sum"),
                                     four=("is_four", "sum"), six=("is_six", "sum"), dots=("runs", lambda x: (x == 0).sum()))
    out["batters"] = []
    for _, r in rb.iterrows():
        s = st.loc[r[pid]]
        out["batters"].append({
            "name": r.player, "rank": int(r.cricket_rank), "role": r.role, "level": r1(r.pressure_level), "sd": r1(r.sd),
            "balls": int(r.pressure_balls), "sr": r1(s.runs / s.balls * 100), "avg": r1(s.runs / max(s.outs, 1)),
            "bnd": r1((s.four + s.six) / s.balls * 100), "dot": r1(s.dots / s.balls * 100),
            "pieces": {"boundary": r1(r.boundary), "rotation": r1(r.rotation), "wicket": r1(r.wicket)},
            "sit": {k: r1(r[c]) for k, c in BAT_SITS.items()}})
    out["bat_avg"] = {"sr": r1(p.runs.sum() / len(p) * 100), "avg": r1(p.runs.sum() / p.striker_out.sum()),
                      "bnd": r1((p.is_four | p.is_six).mean() * 100), "dot": r1((p.runs == 0).mean() * 100)}

    rw = pd.read_csv(os.path.join(OPEN_POOL, "ratings_bowlers.csv"))
    wid = rw.columns[0]
    n_bowl = len(rw)
    rw = rw[rw.evidence_gate].head(TOP_N)
    q = tt[tt.S_bowl_any]
    sw = q.groupby("bowler_id").agg(legal=("is_legal", "sum"), runs=("bowler_runs", "sum"), wk=("bowler_wicket", "sum"), n=("runs", "size"),
                                    dots=("bowler_runs", lambda x: (x == 0).sum()), four=("is_four", "sum"), six=("is_six", "sum"))
    out["bowlers"] = []
    for _, r in rw.iterrows():
        s = sw.loc[r[wid]]
        out["bowlers"].append({
            "name": r.player, "rank": int(r.cricket_rank), "role": r.role, "level": r1(r.pressure_level), "sd": r1(r.sd),
            "balls": int(r.pressure_balls), "econ": r1(s.runs / s.legal * 6), "bpw": r1(s.legal / max(s.wk, 1)), "wk": int(s.wk),
            "dot": r1(s.dots / s.n * 100), "bnd": r1((s.four + s.six) / s.n * 100),
            "pieces": {"boundary": r1(r.boundary), "rotation": r1(r.rotation), "wicket": r1(r.wicket)},
            "sit": {k: r1(r[c]) for k, c in BOWL_SITS.items()}})
    out["bowl_avg"] = {"econ": r1(q.bowler_runs.sum() / q.is_legal.sum() * 6), "bpw": r1(q.is_legal.sum() / q.bowler_wicket.sum()),
                       "dot": r1((q.bowler_runs == 0).mean() * 100), "bnd": r1((q.is_four | q.is_six).mean() * 100)}

    rep = pd.read_csv(os.path.join(ROOT, "evals", "out", "repeatability.csv"))
    out["repeat"] = [{"role": r.role, "stat": r.statistic, "r": float(r["repeat: cricket situations"])}
                     for _, r in rep[rep.kind == "raw"].iterrows()]

    # the chase win-probability model, refitted exactly as in superr_selector/pressure.py
    i2 = bbb[bbb.innings == 2]
    X = chase_features(i2.runs_needed.values, i2.balls_left.values, i2.wickets_in_hand.values)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=5000).fit(scaler.transform(X), i2.bat_team_won)
    out["wp"] = {"mean": [round(float(v), 5) for v in scaler.mean_], "scale": [round(float(v), 5) for v in scaler.scale_],
                 "coef": [round(float(v), 5) for v in model.coef_[0]], "intercept": round(float(model.intercept_[0]), 5)}
    out["meta"] = {"matches": int(bbb.match_id.nunique()), "balls": int(len(bbb)), "bat_eligible": n_bat, "bowl_eligible": n_bowl,
                   "pressure_cut": 0.137}
    with open(os.path.join(ROOT, "website", "data.json"), "w") as handle:
        json.dump(out, handle, separators=(",", ":"))
    print("wrote website/data.json")


if __name__ == "__main__":
    main()
