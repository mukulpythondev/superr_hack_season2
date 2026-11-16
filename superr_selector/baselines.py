"""Expected runs and wicket chance for an average player on each ball, and value above that."""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from superr_selector.config import INTERIM


def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_pressure.csv"), low_memory=False)

    # team strength: runs per ball scored / conceded, relative to the league average
    lg_avg = bbb.groupby("league").total_runs.sum() / bbb.groupby("league").is_legal.sum()
    def strength(team_col):
        t = bbb.groupby(["league", team_col]).agg(r=("total_runs", "sum"), b=("is_legal", "sum")).reset_index()
        t["s"] = t.r / t.b - t.league.map(lg_avg)
        t.loc[t.b < 600, "s"] = t.loc[t.b < 600, "s"] * t.b / 600        # small samples pulled toward 0
        return t.set_index(["league", team_col]).s
    bat_s, bowl_s = strength("batting_team_id"), strength("bowling_team_id")
    bbb["bat_team_strength"]  = pd.MultiIndex.from_frame(bbb[["league", "batting_team_id"]]).map(bat_s).values
    bbb["bowl_team_strength"] = pd.MultiIndex.from_frame(bbb[["league", "bowling_team_id"]]).map(bowl_s).values

    bbb["league_code"] = bbb.league.map({"IPL": 0, "T20I": 1, "Other": 2})
    bbb["rrr_capped"]  = bbb.required_rr.clip(upper=36)
    bbb["is_spin"]     = (bbb.pace_or_spin == "Spin").astype(int)
    FEATS = ["league_code", "year", "innings", "overid", "wickets_before", "balls_left", "score_before",
             "rrr_capped", "runs_needed", "bat_team_strength", "bowl_team_strength", "is_spin"]

    def expected(target, rows, classifier=False):
        """Cross-fitted prediction: every match is predicted by a model that never saw that match."""
        d = bbb[rows]
        if classifier:
            m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=200, random_state=0)
            p = cross_val_predict(m, d[FEATS], d[target].astype(int), groups=d.match_id, cv=GroupKFold(5), method="predict_proba")[:, 1]
        else:
            m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=200, random_state=0)
            p = cross_val_predict(m, d[FEATS], d[target], groups=d.match_id, cv=GroupKFold(5))
        return pd.Series(p, index=d.index)

    faced = bbb.is_ball_faced
    allb  = pd.Series(True, index=bbb.index)
    bbb["x_bat_runs"]   = expected("runs", faced)
    bbb["x_bat_out"]    = expected("striker_out", faced, classifier=True)
    bbb["x_bowl_runs"]  = expected("bowler_runs", allb)
    bbb["x_bowl_wkt"]   = expected("bowler_wicket", allb, classifier=True)

    # how many runs is a wicket worth on this ball? (from the win-probability model)
    per_run = (bbb.wp_six - bbb.wp_dot) / 6
    raw_k   = (bbb.wp_dot - bbb.wp_wicket) / per_run.where(per_run > 1e-4)
    phase   = pd.cut(bbb.overid, [0, 6, 15, 20], labels=["PP", "MID", "DEATH"])
    bbb["phase"] = phase
    bbb["wicket_runs"] = raw_k.fillna(raw_k.groupby([bbb.innings, phase], observed=True).transform("median")).clip(3, 20)

    # run value above expectation
    bbb["bat_value"]  = (bbb.runs - bbb.x_bat_runs) - bbb.wicket_runs * (bbb.striker_out.astype(float) - bbb.x_bat_out)
    bbb["bowl_value"] = -(bbb.bowler_runs - bbb.x_bowl_runs) + bbb.wicket_runs * (bbb.bowler_wicket.astype(float) - bbb.x_bowl_wkt)
    bbb.loc[~faced, "bat_value"] = np.nan

    # split each value into three pieces: boundary runs | other runs | wickets
    bnd = bbb.runs.where(bbb.is_four | bbb.is_six, 0).astype(float)
    bbb["_bnd"] = bnd
    x_bnd_faced = expected("_bnd", faced)
    x_bnd_all   = expected("_bnd", allb)
    bbb["bat_p_boundary"]  = (bnd - x_bnd_faced).where(faced)
    bbb["bat_p_rotation"]  = ((bbb.runs - bbb.x_bat_runs) - (bnd - x_bnd_faced)).where(faced)
    bbb["bat_p_wicket"]    = (-bbb.wicket_runs * (bbb.striker_out.astype(float) - bbb.x_bat_out)).where(faced)
    bbb["bowl_p_boundary"] = -(bnd - x_bnd_all)
    bbb["bowl_p_rotation"] = -((bbb.bowler_runs - bbb.x_bowl_runs) - (bnd - x_bnd_all))
    bbb["bowl_p_wicket"]   = bbb.wicket_runs * (bbb.bowler_wicket.astype(float) - bbb.x_bowl_wkt)
    bbb = bbb.drop(columns="_bnd")

    print("average wicket value (runs):"); print(bbb.groupby(["innings", "phase"], observed=True).wicket_runs.mean().round(1))
    print("baseline check, mean value per ball should be ~0:")
    print(bbb.groupby("league")[["bat_value", "bowl_value"]].mean().round(3))
    print(bbb.groupby("high_pressure")[["bat_value", "bowl_value"]].mean().round(3))
    print("expected vs actual runs/ball by pressure:"); print(bbb[faced].groupby("high_pressure")[["runs", "x_bat_runs", "striker_out", "x_bat_out"]].mean().round(4))
    bbb.to_csv(os.path.join(INTERIM, "bbb_value.csv"), index=False)
    print("Step 3 done")


if __name__ == "__main__":
    main()
