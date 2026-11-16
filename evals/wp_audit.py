"""Win-probability checks beyond AUC: variance budget and the cost of an early wicket."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import log_loss, roc_auc_score
from _paths import EOUT
from superr_selector.config import INTERIM

import os
bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
c = bbb[bbb.innings == 2].copy().reset_index(drop=True)
c["rrr"] = (c.runs_needed / c.balls_left.clip(lower=1) * 6).clip(upper=36)

def feats(r, b, w):
    X = pd.DataFrame({"runs_needed": r, "balls_left": b, "wk": w})
    X["rrr"] = (X.runs_needed / X.balls_left.clip(lower=1) * 6).clip(upper=36)
    X["frac_left"] = X.balls_left / 120
    X["wk_x_time"] = X.wk * X.frac_left               # wickets matter more when more balls remain
    X["runs_per_wk"] = X.runs_needed / X.wk.clip(lower=1)
    X["sqrt_balls"] = np.sqrt(X.balls_left.clip(lower=0))
    return X

BASE = ["runs_needed", "balls_left", "wk", "rrr"]
models = {
 "M0 plain logistic (first version)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), BASE),
 "M1 logistic + squares/interactions": (make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(max_iter=5000, C=0.3)), BASE),
 "M2 logistic + cricket features (in use)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), BASE + ["wk_x_time", "runs_per_wk", "sqrt_balls"]),
 "M3 boosting, monotonic": (HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=300, monotonic_cst=[-1, 0, 1, -1], random_state=0), BASE),
}
X = feats(c.runs_needed.values, c.balls_left.values, c.wickets_in_hand.values)

def terminal(p, r, b, w):
    p = np.where(r <= 0, 1.0, p)
    return np.where((r > 0) & ((b <= 0) | (w <= 0)), 0.0, p)

rows = []
for name, (m, cols) in models.items():
    p = cross_val_predict(m, X[cols], c.bat_team_won, groups=c.match_id, cv=GroupKFold(5), method="predict_proba")[:, 1]
    p = terminal(p, c.runs_needed.values, c.balls_left.values, c.wickets_in_hand.values)
    # variance budget: sum of squared moves per innings, including the final move to 0/1
    nxt = pd.Series(p).groupby(c.match_id).shift(-1)
    nxt = nxt.fillna(c.bat_team_won.astype(float))
    sq = ((nxt - p) ** 2).groupby(c.match_id).sum()
    p0 = pd.Series(p).groupby(c.match_id).first()
    m.fit(X[cols], c.bat_team_won)
    # early-wicket cost at a typical start: 170 needed, overs 1-3
    def wp(r, b, w):
        return terminal(m.predict_proba(feats(np.array(r), np.array(b), np.array(w))[cols])[:, 1], np.array(r), np.array(b), np.array(w))
    cost = wp([165.], [108.], [10.])[0] - wp([165.], [108.], [9.])[0]
    six  = wp([159.], [107.], [10.])[0] - wp([165.], [107.], [10.])[0]
    late = wp([12.], [6.], [5.])[0]
    rows.append({"model": name, "log_loss": log_loss(c.bat_team_won, np.clip(p, 1e-4, 1 - 1e-4)), "auc": roc_auc_score(c.bat_team_won, p),
                 "squared moves per innings": sq.mean(), "budget p0(1-p0)": (p0 * (1 - p0)).mean(),
                 "early wicket cost": cost, "early six gain": six, "12 off 6, 5 wkts": late})
res = pd.DataFrame(rows).set_index("model").round(3)
print(res.T.to_string())
res.to_csv(os.path.join(EOUT, "wp_audit.csv"), lineterminator="\n")

# what really happens after an early wicket? compare chases 1+ down vs 0 down after 3 overs, matched on runs needed
s = c[c.balls_left == 102].drop_duplicates("match_id")
s["band"] = pd.cut(s.runs_needed, [0, 130, 150, 170, 190, 400])
t = s.groupby(["band", s.wickets_in_hand == 10], observed=True).bat_team_won.agg(["mean", "size"]).unstack()
t.columns = ["win% 1+ down", "win% 0 down", "n 1+ down", "n 0 down"]
t["actual cost of wickets"] = t["win% 0 down"] - t["win% 1+ down"]
print("\nREAL DATA after 3 overs of a chase:"); print(t.round(2).to_string())
print("weighted actual cost: %.3f (avg wickets lost in the '1+ down' group: %.2f)" % (
    np.average(t["actual cost of wickets"], weights=t["n 1+ down"]), (10 - s[s.wickets_in_hand < 10].wickets_in_hand).mean()))
