"""Sanity checks on the final lists: plain stats, sensitivity, agreement with the model label."""
import os, json, pandas as pd
from superr_selector.config import DATA, DIAG, INTERIM, OPEN_POOL
from superr_selector.situations import add_situations, BAT_SITUATIONS, BOWL_SITUATIONS
from superr_selector.ratings import build, ELIG
from superr_selector.roles import load_roles


def main():
    full = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
    matches = pd.read_csv(os.path.join(DATA, "matches.csv"))
    params = json.load(open(os.path.join(DIAG, "skill_spread.json")))
    CFG = {"bat": (BAT_SITUATIONS, "S_bat_any"), "bowl": (BOWL_SITUATIONS, "S_bowl_any")}

    roles = load_roles(full)

    def lists(b, **kw):
        out = {}
        for role, (sits, any_col) in CFG.items():
            t, per, mix = build(b[b.top_tier], role, params[role], sits, any_col, roles, **kw)
            out[role] = (t, t[t.evidence_gate].head(10), mix)
        return out

    base = lists(full)
    bbb = full[full.top_tier]

    # (a) plain stats
    p = bbb[bbb.S_bat_any & bbb.is_ball_faced]
    bat = p.groupby("striker_id").agg(balls=("runs", "size"), runs=("runs", "sum"), outs=("striker_out", "sum"),
                                       bnd=("is_four", "sum"), six=("is_six", "sum"), dots=("runs", lambda x: (x == 0).sum()))
    bat["strike_rate"] = bat.runs / bat.balls * 100; bat["average"] = bat.runs / bat.outs.clip(lower=1)
    bat["boundary_pct"] = (bat.bnd + bat.six) / bat.balls * 100; bat["dot_pct"] = bat.dots / bat.balls * 100
    t = base["bat"][1][["player", "model_rank"]].join(bat[["balls", "runs", "strike_rate", "average", "boundary_pct", "dot_pct"]])
    t.insert(0, "rank", range(1, len(t) + 1))
    print(f"DIRECT TOP 10 BATTERS on cricket pressure balls | all players: strike rate {p.runs.sum()/len(p)*100:.0f}, average {p.runs.sum()/p.striker_out.sum():.1f}, "
          f"boundary {(p.is_four|p.is_six).mean()*100:.1f}%, dot {(p.runs==0).mean()*100:.1f}%")
    print(t.round(1).to_string(index=False)); t.round(2).to_csv(os.path.join(OPEN_POOL, "top10_batters_plain_stats.csv"), index=False)

    q = bbb[bbb.S_bowl_any]
    bw = q.groupby("bowler_id").agg(legal=("is_legal", "sum"), runs=("bowler_runs", "sum"), wkts=("bowler_wicket", "sum"), n=("runs", "size"),
                                    dots=("bowler_runs", lambda x: (x == 0).sum()), bnd=("is_four", "sum"), six=("is_six", "sum"))
    bw["economy"] = bw.runs / bw.legal * 6; bw["balls_per_wicket"] = bw.legal / bw.wkts.clip(lower=1)
    bw["dot_pct"] = bw.dots / bw.n * 100; bw["boundary_pct"] = (bw.bnd + bw.six) / bw.n * 100
    t = base["bowl"][1][["player", "model_rank"]].join(bw[["legal", "wkts", "economy", "balls_per_wicket", "dot_pct", "boundary_pct"]])
    t.insert(0, "rank", range(1, len(t) + 1))
    print(f"\nDIRECT TOP 10 BOWLERS on cricket pressure balls | all players: economy {q.bowler_runs.sum()/q.is_legal.sum()*6:.2f}, balls per wicket {q.is_legal.sum()/q.bowler_wicket.sum():.1f}, "
          f"dot {(q.bowler_runs==0).mean()*100:.1f}%, boundary {(q.is_four|q.is_six).mean()*100:.1f}%")
    print(t.round(1).to_string(index=False)); t.round(2).to_csv(os.path.join(OPEN_POOL, "top10_bowlers_plain_stats.csv"), index=False)

    # (b) sensitivity of the direct top 10
    def wk_const(b, wk):
        b = b.copy()
        b["bat_p_wicket"]  = (-wk * (b.striker_out.astype(float) - b.x_bat_out)).where(b.is_ball_faced)
        b["bowl_p_wicket"] = wk * (b.bowler_wicket.astype(float) - b.x_bowl_wkt)
        b["bat_value"]  = b.bat_p_boundary + b.bat_p_rotation + b.bat_p_wicket
        b["bowl_value"] = b.bowl_p_boundary + b.bowl_p_rotation + b.bowl_p_wicket
        return b
    variants = [("knockout balls: no extra weight", lambda: lists(add_situations(full, matches, big_weight=1.0))),
                ("knockout balls: double weight", lambda: lists(add_situations(full, matches, big_weight=2.0))),
                ("collapse = 2+ wickets down", lambda: lists(add_situations(full, matches, collapse_wkts=2))),
                ("chase start = target 170+", lambda: lists(add_situations(full, matches, chase_start_target=170))),
                ("chase start = target 190+", lambda: lists(add_situations(full, matches, chase_start_target=190))),
                ("wicket always worth 8 runs", lambda: lists(wk_const(full, 8))),
                ("wicket always worth 12 runs", lambda: lists(wk_const(full, 12))),
                ("minimum 200 pressure balls", lambda: lists(full, elig=dict(ELIG, min_hp=200)))]
    rows = []
    for label, fn in variants:
        v = fn()
        rows.append({"change": label, "batters kept (of 10)": len(set(v["bat"][1].player) & set(base["bat"][1].player)),
                     "bowlers kept (of 10)": len(set(v["bowl"][1].player) & set(base["bowl"][1].player))})
    s = pd.DataFrame(rows); print("\nSENSITIVITY of the direct top 10"); print(s.to_string(index=False))
    s.to_csv(os.path.join(DIAG, "sensitivity.csv"), index=False)

    # (c) agreement between lists
    for role, name in [("bat", "batters"), ("bowl", "bowlers")]:
        t, direct, mix = base[role]
        model10 = set(t[t.model_rank <= 10].player)
        print(f"\n{name}: direct & role-mix share {len(set(direct.player) & set(mix.player))} of 10 | direct & model cross-check share {len(set(direct.player) & model10)} of 10")
        print("  in direct top 10 but not in model top 10:", sorted(set(direct.player) - model10))
        print("  in model top 10 but not in direct top 10:", sorted(model10 - set(direct.player)))
    print("Step 7 done")


if __name__ == "__main__":
    main()
