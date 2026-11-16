"""Pressure situations, defined from cricket rather than from the model.

Batting: A collapse (3+ down by over 10), B big chase (RRR 10+, <=10 overs left),
         C finish (overs 17-20), I chase start (overs 1-6 chasing 180+)
Bowling: E death (overs 17-20), F set batter (overs 7-16, striker on 20+ balls or stand of 40+),
         G small total (defending <=155), H bounce back (previous ball went for 4 or 6)
All except A need the match to be live. Knockout balls get extra weight."""
import os, numpy as np, pandas as pd
from superr_selector.config import DATA, DIAG, INTERIM

BIG_MATCH_WEIGHT = 1.5
CHASE_START_TARGET = 180          # "big target" for situation I; 170 and 190 are tested in checks.py
BAT_SITUATIONS  = {"A_collapse": "Rebuilder (collapse)", "B_big_chase": "Chaser (big chase)", "C_finish": "Finisher (overs 17-20)",
                   "I_chase_start": "Opener (chase start, big target)"}
BOWL_SITUATIONS = {"E_death": "Death bowler", "F_set_batter": "Versus set batter", "G_small_total": "Defending small total", "H_bounce_back": "Bounce-back after boundary"}

def add_situations(bbb, matches, big_weight=BIG_MATCH_WEIGHT, collapse_wkts=3, chase_start_target=CHASE_START_TARGET):
    bbb = bbb.drop(columns=[c for c in bbb.columns if c.startswith("S_") or c in ("stage", "knock_out", "w", "live", "big_match")], errors="ignore")
    bbb = bbb.merge(matches[["match_id", "stage", "knock_out"]], on="match_id", how="left")
    bbb = bbb.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    g = ["match_id", "innings"]
    bbb["striker_balls"] = bbb.groupby(g + ["striker_id"]).is_ball_faced.cumsum() - bbb.is_ball_faced
    wk_no = bbb.groupby(g).is_wicket.cumsum() - bbb.is_wicket
    bbb["partnership_runs"] = bbb.groupby(g + [wk_no]).total_runs.cumsum() - bbb.total_runs
    prev = (bbb.is_four | bbb.is_six).groupby([bbb.match_id, bbb.innings, bbb.overid, bbb.bowler_id]).shift(1)
    prev_boundary = prev.fillna(False).astype(bool)

    chase = bbb.innings == 2
    bbb["live"] = np.where(chase, bbb.win_prob.between(0.10, 0.90), bbb.win_prob.between(0.20, 0.80))
    live = bbb.live
    bbb["S_A_collapse"]    = (bbb.wickets_before >= collapse_wkts) & (bbb.overid <= 10)
    bbb["S_B_big_chase"]   = chase & (bbb.required_rr >= 10) & (bbb.balls_left <= 60) & live
    bbb["S_C_finish"]      = (bbb.overid >= 17) & live
    bbb["S_I_chase_start"] = chase & (bbb.overid <= 6) & (bbb.first_total + 1 >= chase_start_target) & live
    bbb["S_E_death"]       = bbb.S_C_finish
    bbb["S_F_set_batter"]  = bbb.overid.between(7, 16) & ((bbb.striker_balls >= 20) | (bbb.partnership_runs >= 40)) & live
    bbb["S_G_small_total"] = chase & (bbb.first_total <= 155) & live
    bbb["S_H_bounce_back"] = prev_boundary & live
    bbb["S_bat_any"]  = bbb[["S_" + s for s in BAT_SITUATIONS]].any(axis=1)
    bbb["S_bowl_any"] = bbb[["S_" + s for s in BOWL_SITUATIONS]].any(axis=1)
    bbb["big_match"] = (bbb.knock_out == 1) | bbb.stage.isin(["Final", "Qualifier", "Eliminator", "Semi Final"])
    bbb["w"] = np.where(bbb.big_match, big_weight, 1.0)
    return bbb

def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
    matches = pd.read_csv(os.path.join(DATA, "matches.csv"))
    bbb = add_situations(bbb, matches)
    tt = bbb[bbb.top_tier]
    rows = []
    for s in ["S_A_collapse", "S_B_big_chase", "S_C_finish", "S_I_chase_start", "S_F_set_batter", "S_G_small_total", "S_H_bounce_back", "S_bat_any", "S_bowl_any", "big_match"]:
        d = tt[tt[s]]
        rows.append({"situation": s.replace("S_", ""), "balls": len(d), "% of balls": len(d) / len(tt) * 100, "matches": d.match_id.nunique(),
                     "% in 1st innings": (d.innings == 1).mean() * 100, "run rate": d.total_runs.sum() / d.is_legal.sum() * 6,
                     "wkts per 100 balls": d.is_wicket.sum() / d.is_legal.sum() * 100,
                     "batters 60+ balls": (d[d.is_ball_faced].groupby("striker_id").size() >= 60).sum(),
                     "bowlers 60+ balls": (d.groupby("bowler_id").size() >= 60).sum()})
    res = pd.DataFrame(rows).round(1); print(res.to_string(index=False))
    res.to_csv(os.path.join(DIAG, "situations_summary.csv"), index=False)
    bbb.to_csv(os.path.join(INTERIM, "bbb_value.csv"), index=False)
    print("Step 4 done")


if __name__ == "__main__":
    main()
