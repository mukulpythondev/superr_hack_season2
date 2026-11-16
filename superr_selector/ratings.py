"""Final ratings: direct top 10 and role-mix top 10 for batters and bowlers."""
import os, json, numpy as np, pandas as pd
from superr_selector.config import DIAG, INTERIM, OPEN_POOL
from superr_selector.rating_core import rate, rate_pieces
from superr_selector.roles import BATTING_ROLES, BOWLING_ROLES, load_roles
from superr_selector.situations import BAT_SITUATIONS, BOWL_SITUATIONS

ELIG     = dict(min_hp=120, min_matches=15, recent_year=2025)     # all situations together
ELIG_SIT = dict(min_hp=60,  min_matches=15, recent_year=2025)     # one situation
ROLE_MIX = {"bat": ["I_chase_start", "A_collapse", "B_big_chase", "C_finish"] * 2 + ["I_chase_start", "A_collapse"],
            "bowl": ["E_death", "F_set_batter", "G_small_total", "H_bounce_back"] * 2 + ["E_death", "F_set_batter"]}

def model_table(plain, role, params, elig=ELIG):
    A = rate(plain, role, params["total"], **elig)
    B = rate_pieces(plain, role, params, **elig)
    C = rate(plain.assign(high_pressure=plain.high_pressure_alt), role, params["total_alt"], **elig)
    t = A[["player", "hp_raw", "rating"]].join(B[["boundary", "rotation", "wicket", "rating"]].rename(columns={"rating": "r_pieces"}), how="inner")
    t = t.join(C[["rating"]].rename(columns={"rating": "r_alt"}), how="inner")
    t["model_avg_rank"] = pd.concat([t.rating.rank(ascending=False), t.r_pieces.rank(ascending=False), t.r_alt.rank(ascending=False)], axis=1).mean(axis=1)
    t = t[t.hp_raw > 0]
    t["model_rank"] = t.model_avg_rank.rank(method="first")
    return t

def cricket_tables(bbb, role, params, sits, any_col, elig=ELIG, elig_sit=ELIG_SIT):
    u = rate(bbb.assign(high_pressure=bbb[any_col]), role, params["cricket_pooled"], **elig)
    u = u[["player", "matches", "hp_n", "usual_raw", "hp_raw", "good_match_pct", "pressure_level", "sd", "rating"]]
    u = u.rename(columns={"hp_n": "pressure_balls", "hp_raw": "pressure_raw"})
    per = {}
    for s in sits:
        r = rate(bbb.assign(high_pressure=bbb["S_" + s]), role, params["cricket_pooled"], **elig_sit)
        per[s] = r[["player", "hp_n", "hp_raw", "pressure_level", "sd", "rating"]].rename(columns={"hp_n": "balls", "hp_raw": "raw"})
    return u, per

def role_mix(per, order, allowed):
    """Pick the best remaining player of each situation in turn."""
    picked, rows = set(), []
    for s in order:
        r = per[s]
        r = r[(r.raw > 0) & r.index.isin(allowed) & ~r.index.isin(picked)].sort_values("rating", ascending=False)
        if len(r):
            pid = r.index[0]; picked.add(pid)
            rows.append({"player_id": pid, "player": r.player.iloc[0], "situation": s, "balls": r.balls.iloc[0],
                         "raw": r.raw.iloc[0], "pressure_level": r.pressure_level.iloc[0], "rating": r.rating.iloc[0]})
    return pd.DataFrame(rows)

def build(bbb, role, params, sits, any_col, roles, **kw):
    """`roles` is the table from roles.player_roles; only players whose role fits the list are ranked."""
    plain = bbb.drop(columns="w")
    u, per = cricket_tables(bbb, role, params, sits, any_col, **kw)
    m = model_table(plain, role, params)
    wanted = BATTING_ROLES if role == "bat" else BOWLING_ROLES
    in_role = roles.index[roles.role.isin(wanted)]
    u = u[u.index.isin(in_role)]
    per = {s: r[r.index.isin(in_role)] for s, r in per.items()}
    m = m[m.index.isin(in_role)].copy()
    m["model_rank"] = m.model_avg_rank.rank(method="first")
    u = u.join(roles.role)
    t = u.join(m[["boundary", "rotation", "wicket", "model_rank"]], how="left")
    for s in sits:
        t = t.join(per[s][["balls", "pressure_level"]].rename(columns={"balls": s + "_balls", "pressure_level": s + "_level"}), how="left")
    t["evidence_gate"] = t.pressure_raw > 0
    t = t.sort_values("rating", ascending=False)
    t["cricket_rank"] = np.nan
    t.loc[t.evidence_gate, "cricket_rank"] = range(1, int(t.evidence_gate.sum()) + 1)
    return t, per, role_mix(per, ROLE_MIX[role], t.index[t.evidence_gate])

def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
    roles = load_roles(bbb)                      # roles use every match the player appears in
    roles.round(2).to_csv(os.path.join(DIAG, "player_roles.csv"), lineterminator="\n")
    bbb = bbb[bbb.top_tier]
    params = json.load(open(os.path.join(DIAG, "skill_spread.json")))
    for role, name, sits, any_col in [("bat", "batters", BAT_SITUATIONS, "S_bat_any"), ("bowl", "bowlers", BOWL_SITUATIONS, "S_bowl_any")]:
        t, per, mix = build(bbb, role, params[role], sits, any_col, roles)
        t.round(2).to_csv(os.path.join(OPEN_POOL, f"ratings_{name}.csv"))
        direct = t[t.evidence_gate].head(10)
        show = ["cricket_rank", "player", "role", "pressure_balls", "usual_raw", "pressure_raw", "pressure_level", "sd", "rating", "model_rank"] + [s + "_level" for s in sits]
        direct[show].round(2).to_csv(os.path.join(OPEN_POOL, f"top10_{name}_direct.csv"), index=False)
        mix.insert(0, "pick", range(1, len(mix) + 1))
        mix = mix.join(t[["cricket_rank", "model_rank"]], on="player_id")
        mix.drop(columns="player_id").round(2).to_csv(os.path.join(OPEN_POOL, f"top10_{name}_role_mix.csv"), index=False)
        print(f"\n=== {name.upper()}: {len(t)} eligible (all situations) ===")
        print("DIRECT TOP 10 (+ next 5)"); print(t[t.evidence_gate][show].head(15).round(1).to_string(index=False))
        print("ROLE-MIX TOP 10"); print(mix.drop(columns="player_id").round(1).to_string(index=False))
        for s in sits:
            r = per[s]; r = r[r.raw > 0].sort_values("rating", ascending=False)
            print(f"  best in {sits[s]}: " + ", ".join(f"{p} ({int(b)}b, {l:+.0f})" for p, b, l in zip(r.player.head(6), r.balls.head(6), r.pressure_level.head(6))))
    print("Step 6 done")


if __name__ == "__main__":
    main()
