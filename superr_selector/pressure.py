"""Win-probability models. Pressure of a ball = win chance after a six minus after a wicket."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss
from superr_selector.config import INTERIM


def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_state.csv"), low_memory=False)
    inn1, inn2 = bbb[bbb.innings == 1], bbb[bbb.innings == 2]

    PLAIN = ["runs_needed", "balls_left", "wickets_in_hand", "required_rr"]
    FULL  = PLAIN + ["wk_x_time", "runs_per_wk", "sqrt_balls"]

    def chase_X(runs_needed, balls_left, wk):
        X = pd.DataFrame({"runs_needed": runs_needed, "balls_left": balls_left, "wickets_in_hand": wk})
        X["required_rr"] = (X.runs_needed / X.balls_left.clip(lower=1) * 6).clip(upper=36)
        X["wk_x_time"]   = X.wickets_in_hand * X.balls_left / 120        # wickets matter more with more balls left
        X["runs_per_wk"] = X.runs_needed / X.wickets_in_hand.clip(lower=1)
        X["sqrt_balls"]  = np.sqrt(X.balls_left.clip(lower=0))
        return X

    def first_X(score, balls_left, wk):
        return pd.DataFrame({"score_before": score, "balls_left": balls_left, "wickets_in_hand": wk})

    model2       = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model2_plain = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model1 = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                           StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    X1 = first_X(inn1.score_before.values, inn1.balls_left.values, inn1.wickets_in_hand.values)
    X2 = chase_X(inn2.runs_needed.values, inn2.balls_left.values, inn2.wickets_in_hand.values)

    # honest test: 5-fold cross-validation, whole matches held out together
    for name, m, X, d in [("1st innings", model1, X1, inn1), ("chase", model2, X2[FULL], inn2), ("chase (plain)", model2_plain, X2[PLAIN], inn2)]:
        p = cross_val_predict(m, X, d.bat_team_won, groups=d.match_id, cv=GroupKFold(5), method="predict_proba")[:, 1]
        print(f"{name:14s} AUC={roc_auc_score(d.bat_team_won, p):.3f}  log_loss={log_loss(d.bat_team_won, p):.3f}")
    model1.fit(X1, inn1.bat_team_won); model2.fit(X2[FULL], inn2.bat_team_won); model2_plain.fit(X2[PLAIN], inn2.bat_team_won)

    def wp_chase(r, b, w, model=model2, cols=FULL):
        p = model.predict_proba(chase_X(r, b, w)[cols])[:, 1]
        p = np.where(r <= 0, 1.0, p)
        return np.where((r > 0) & ((b <= 0) | (w <= 0)), 0.0, p)

    def wp_first(s, b, w):
        return model1.predict_proba(first_X(s, b, w))[:, 1]

    s, b, w = inn1.score_before.values, inn1.balls_left.values, inn1.wickets_in_hand.values
    bbb.loc[inn1.index, "win_prob"]  = wp_first(s, b, w)
    bbb.loc[inn1.index, "wp_six"]    = wp_first(s + 6, b - 1, w)
    bbb.loc[inn1.index, "wp_dot"]    = wp_first(s, b - 1, w)
    bbb.loc[inn1.index, "wp_wicket"] = wp_first(s, b - 1, w - 1)
    r, b, w = inn2.runs_needed.values, inn2.balls_left.values, inn2.wickets_in_hand.values
    bbb.loc[inn2.index, "win_prob"]  = wp_chase(r, b, w)
    bbb.loc[inn2.index, "wp_six"]    = wp_chase(r - 6, b - 1, w)
    bbb.loc[inn2.index, "wp_dot"]    = wp_chase(r, b - 1, w)
    bbb.loc[inn2.index, "wp_wicket"] = wp_chase(r, b - 1, w - 1)

    bbb["pressure"] = (bbb.wp_six - bbb.wp_wicket).clip(lower=0)
    cut = bbb.pressure.quantile(0.75)
    bbb["high_pressure"] = bbb.pressure >= cut

    # alternative label from the plain chase model (robustness check only)
    alt = bbb.pressure.copy()
    alt.loc[inn2.index] = np.clip(wp_chase(r - 6, b - 1, w, model2_plain, PLAIN) - wp_chase(r, b - 1, w - 1, model2_plain, PLAIN), 0, None)
    bbb["high_pressure_alt"] = alt >= alt.quantile(0.75)

    print(f"high-pressure cut-off = {cut:.3f}")
    print(bbb.groupby("innings").high_pressure.mean().round(3))
    ex = pd.DataFrame({"runs_needed": [8, 5, 90, 60, 30, 12, 10], "balls_left": [1, 30, 60, 36, 18, 6, 6], "wickets_in_hand": [2, 8, 4, 7, 6, 5, 5]})
    rr, bb, ww = ex.runs_needed.values, ex.balls_left.values, ex.wickets_in_hand.values
    ex["win_prob"] = wp_chase(rr, bb, ww).round(2); ex["pressure"] = (wp_chase(rr - 6, bb - 1, ww) - wp_chase(rr, bb - 1, ww - 1)).round(3)
    print(ex.to_string(index=False))
    bbb.to_csv(os.path.join(INTERIM, "bbb_pressure.csv"), index=False)
    print("Step 2 done")


if __name__ == "__main__":
    main()
