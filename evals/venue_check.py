"""Does a venue factor improve the expected-runs baseline?"""
import os, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from _paths import EOUT
from superr_selector.config import DATA, INTERIM, OPEN_POOL

bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
m = pd.read_csv(os.path.join(DATA, "matches.csv"))[["match_id", "venue_id", "venue_country"]]
bbb = bbb.merge(m, on="match_id", how="left")

lg = bbb.groupby("league").total_runs.sum() / bbb.groupby("league").is_legal.sum()
mt = bbb.groupby(["match_id", "venue_id", "league"]).agg(r=("total_runs", "sum"), b=("is_legal", "sum")).reset_index()
vt = mt.groupby("venue_id").agg(R=("r", "sum"), B=("b", "sum"))
mt = mt.join(vt, on="venue_id")
mt["exp"] = mt.league.map(lg)
# leave this match out; compare with league average; shrink (600 balls of prior)
mt["venue_factor"] = ((mt.R - mt.r) - mt.exp * (mt.B - mt.b)) / ((mt.B - mt.b) + 600)
bbb = bbb.merge(mt[["match_id", "venue_factor"]], on="match_id")
bbb["in_india"] = (bbb.venue_country == "India").astype(int)

v = mt.drop_duplicates("venue_id").assign(rpo=lambda d: d.R / d.B * 6)
big = v[v.B >= 2400]
print(f"{len(v)} grounds; among the {len(big)} with 10+ matches, runs per over ranges {big.rpo.min():.1f} to {big.rpo.max():.1f}")
print("matches by country:", bbb.groupby("venue_country").match_id.nunique().sort_values(ascending=False).head(6).to_dict())

FEATS = ["league_code", "year", "innings", "overid", "wickets_before", "balls_left", "score_before",
         "rrr_capped", "runs_needed", "bat_team_strength", "bowl_team_strength", "is_spin"]
f = bbb[bbb.is_ball_faced]
def cv(feats):
    mdl = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=200, random_state=0)
    return cross_val_predict(mdl, f[feats], f.runs, groups=f.match_id, cv=GroupKFold(5))
p0, p1 = cv(FEATS), cv(FEATS + ["venue_factor", "in_india"])
base = ((f.runs - f.runs.mean()) ** 2).mean()
print("expected-runs model, share of ball-to-ball variation explained: without venue %.4f | with venue %.4f" % (
    1 - ((f.runs - p0) ** 2).mean() / base, 1 - ((f.runs - p1) ** 2).mean() / base))
# match-level: does the baseline get a whole innings right?
g = f.assign(p0=p0, p1=p1).groupby(["match_id", "innings"]).agg(a=("runs", "sum"), p0=("p0", "sum"), p1=("p1", "sum"))
print("average error in predicting an innings' bat runs: without venue %.1f | with venue %.1f" % ((g.a - g.p0).abs().mean(), (g.a - g.p1).abs().mean()))
# which top players' numbers were helped or hurt by their grounds?
f = f.assign(shift=(p0 - p1))          # positive = venue made scoring harder than the baseline assumed
hp = f[f.high_pressure & f.top_tier]
s = hp.groupby("striker_name").agg(balls=("runs", "size"), shift=("shift", "mean"), india=("in_india", "mean"))
s["value_change_per_100"] = s["shift"] * 100
top = pd.read_csv(os.path.join(OPEN_POOL, "top10_batters_direct.csv")).player
print("\neffect of the venue adjustment on our top 10 batters' pressure value (runs per 100 balls):")
print(s.loc[top, ["balls", "india", "value_change_per_100"]].round(2).to_string())
print("\nbiggest changes among batters with 120+ pressure balls:")
e = s[s.balls >= 120].sort_values("value_change_per_100")
print(pd.concat([e.head(5), e.tail(5)])[["balls", "india", "value_change_per_100"]].round(2).to_string())
pd.DataFrame([{"innings_error_without_venue": (g.a - g.p0).abs().mean(), "innings_error_with_venue": (g.a - g.p1).abs().mean()}]).round(2).to_csv(os.path.join(EOUT, "venue_check.csv"), index=False, lineterminator="\n")
