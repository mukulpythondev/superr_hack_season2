"""Rating maths shared by the rankings: per-player values, skill spread, shrinkage."""
import numpy as np, pandas as pd

ROLE = {"bat": ("striker_id", "striker_name", "bat_value"),
        "bowl": ("bowler_id", "bowler_name", "bowl_value")}
PARTS = {"bat": ["bat_p_boundary", "bat_p_rotation", "bat_p_wicket"],
         "bowl": ["bowl_p_boundary", "bowl_p_rotation", "bowl_p_wicket"]}

def mean_and_se(d, pid, val):
    """Value per 100 balls per player. The standard error treats each match as one observation."""
    d = d.assign(_w=d["w"] if "w" in d.columns else 1.0)
    d = d.assign(_wv=d._w * d[val])
    pm = d.groupby([pid, "match_id"]).agg(A=("_wv", "sum"), B=("_w", "sum"), cnt=(val, "size")).reset_index()
    t = pm.groupby(pid).agg(A=("A", "sum"), W=("B", "sum"), n=("cnt", "sum"), m=("match_id", "size"))
    t["raw"] = t.A / t.W * 100
    pm = pm.merge(t[["raw"]], left_on=pid, right_index=True)
    pm["r2"] = (pm.A * 100 - pm.raw * pm.B) ** 2
    t["se"] = np.sqrt(pm.groupby(pid).r2.sum() / t.W ** 2 * t.m / (t.m - 1).clip(lower=1))
    return t[["raw", "se", "n", "m"]]

def player_table(bbb, role):
    pid, pname, val = ROLE[role]
    d = bbb[bbb[val].notna()]
    hp, us = mean_and_se(d[d.high_pressure], pid, val), mean_and_se(d[~d.high_pressure], pid, val)
    p = d.groupby(pid).agg(player=(pname, "last"), last_year=("year", "max"), matches=("match_id", "nunique"), balls=(val, "size"))
    p = p.join(hp.add_prefix("hp_")).join(us.add_prefix("usual_"))
    hm = d[d.high_pressure].groupby([pid, "match_id"])[val].sum()
    p["good_match_pct"] = (hm > 0).groupby(level=0).mean() * 100
    return p

def skill_spread(bbb, role, rng, reps=100, min_half=60):
    """Split-half estimate of how much real spread there is between players."""
    pid, _, val = ROLE[role]
    d = bbb[bbb[val].notna()]
    ids = d.match_id.unique(); acc = []
    for _ in range(reps):
        h = d.match_id.map(pd.Series(rng.integers(0, 2, len(ids)), index=ids)).values
        o = {}
        for k in (0, 1):
            dk = d[h == k]
            hp, us = mean_and_se(dk[dk.high_pressure], pid, val), mean_and_se(dk[~dk.high_pressure], pid, val)
            o[k] = pd.DataFrame({"hp": hp.raw, "n": hp.n, "us": us.raw})
        j = o[0].join(o[1], lsuffix="_a", rsuffix="_b").dropna()
        j = j[(j.n_a >= min_half) & (j.n_b >= min_half)]
        da, db = j.hp_a - j.us_a, j.hp_b - j.us_b
        acc.append([len(j), j.hp_a.corr(j.hp_b), j.us_a.corr(j.us_b), da.corr(db), (j.us_a.corr(j.hp_b) + j.us_b.corr(j.hp_a)) / 2,
                    j.hp_a.cov(j.hp_b), j.us_a.cov(j.us_b), (j.us_a.cov(j.hp_b) + j.us_b.cov(j.hp_a)) / 2])
    m = np.mean(acc, axis=0)
    # model: true pressure level = true usual level + a pressure-specific difference.
    # Both cov(usual A, usual B) and cov(usual A, pressure B) estimate the usual-level spread, so we average them.
    tu2 = max((m[6] + m[7]) / 2, 1.0)
    params = {"tau_usual": float(np.sqrt(tu2)), "slope": 1.0, "tau_pressure_specific": float(np.sqrt(max(m[5] - tu2, 0.0)))}
    corr = {"players compared": int(round(m[0])), "pressure half A vs half B": m[1], "usual half A vs half B": m[2],
            "clutch delta half A vs half B": m[3], "usual (A) predicts pressure (B)": m[4]}
    return params, corr

def two_source_posterior(mean, prior, observed, errors):
    """Combine a player's usual value and pressure value into one estimate of his pressure level."""
    present = np.isfinite(observed) & np.isfinite(errors)
    covariance = prior[np.ix_(present, present)] + np.diag(errors[present] ** 2)
    gain = np.linalg.solve(covariance, prior[present, :]).T
    post = mean + gain @ (observed[present] - mean[present])
    sd = np.sqrt(max((prior - gain @ prior[present, :])[1, 1], 0.0))
    return post, sd


def rate(bbb, role, params, min_hp, min_matches, recent_year):
    """Combine the pressure record and the usual level (two-source shrinkage)."""
    p = player_table(bbb, role)
    el = p[(p.hp_n >= min_hp) & (p.matches >= min_matches) & (p.last_year >= recent_year)].copy()
    tu2, b, tr2 = params["tau_usual"] ** 2, params["slope"], params["tau_pressure_specific"] ** 2
    mu_u = np.average(el.usual_raw, weights=1 / (el.usual_se ** 2 + tu2))
    mu_h = np.average(el.hp_raw - b * (el.usual_raw - mu_u), weights=1 / el.hp_se ** 2)
    prior_cov = np.array([[tu2, b * tu2], [b * tu2, b * b * tu2 + tr2]])
    mean = np.array([mu_u, mu_h])
    est, sd, ul = [], [], []
    for u, su, h, sh in zip(el.usual_raw, el.usual_se, el.hp_raw, el.hp_se):
        post, post_sd = two_source_posterior(mean, prior_cov, np.array([u, h]), np.array([su, sh]))
        ul.append(post[0]); est.append(post[1]); sd.append(post_sd)
    el["usual_level"], el["pressure_level"], el["sd"] = ul, est, sd
    el["rating"] = el.pressure_level - 1.28 * el.sd          # cautious (10th percentile) estimate
    el["clutch_delta"] = el.hp_raw - el.usual_raw
    return el

def rate_pieces(bbb, role, params, **kw):
    """Rate the three pieces separately and add them up."""
    val = ROLE[role][2]; out = None
    for part in PARTS[role]:
        r = rate(bbb.assign(**{val: bbb[part]}), role, params[part], **kw)
        name = part.split("_p_")[1]
        r = r[["pressure_level", "sd"]].rename(columns={"pressure_level": name, "sd": "sd_" + name})
        out = r if out is None else out.join(r, how="inner")
    out["pressure_level"] = out[["boundary", "rotation", "wicket"]].sum(axis=1)
    out["sd"] = np.sqrt((out[["sd_boundary", "sd_rotation", "sd_wicket"]] ** 2).sum(axis=1))
    out["rating"] = out.pressure_level - 1.28 * out.sd
    return out
