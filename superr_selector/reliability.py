"""Split-half test: how much of the spread between players is real."""
import os, json, numpy as np, pandas as pd
from superr_selector.config import DIAG, INTERIM
from superr_selector.rating_core import ROLE, PARTS, skill_spread
from superr_selector.situations import BAT_SITUATIONS, BOWL_SITUATIONS


def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
    bbb = bbb[bbb.top_tier]              # rate players only on IPL and full-member T20Is
    plain = bbb.drop(columns="w")        # the model-based cross-check uses no big-match weight
    rng = np.random.default_rng(0)
    params, table = {}, {}
    for role, sits, any_col in [("bat", BAT_SITUATIONS, "S_bat_any"), ("bowl", BOWL_SITUATIONS, "S_bowl_any")]:
        val = ROLE[role][2]
        P = {}
        P["cricket_any"], table[(role, "cricket situations")] = skill_spread(bbb.assign(high_pressure=bbb[any_col]), role, rng, reps=80)
        for s in sits:
            P["cricket_" + s], _ = skill_spread(bbb.assign(high_pressure=bbb["S_" + s]), role, rng, reps=40, min_half=30)
        # Per-situation estimates rest on small samples and jump around, so the cricket ratings use one pooled
        # pair of numbers per discipline: the average real spread across the situations.
        keys = ["cricket_any"] + ["cricket_" + s for s in sits]
        P["cricket_pooled"] = {"tau_usual": float(np.sqrt(np.mean([P[k]["tau_usual"] ** 2 for k in keys]))), "slope": 1.0,
                               "tau_pressure_specific": float(np.sqrt(np.mean([P[k]["tau_pressure_specific"] ** 2 for k in keys])))}
        P["total"], table[(role, "model label")] = skill_spread(plain, role, rng, reps=80)
        for part in PARTS[role]:
            P[part], _ = skill_spread(plain.assign(**{val: plain[part]}), role, rng, reps=40)
        P["total_alt"], _ = skill_spread(plain.assign(high_pressure=plain.high_pressure_alt), role, rng, reps=40)
        params[role] = P

    res = pd.DataFrame(table).round(3)
    print(res.to_string())
    print("\nreal spread between players, runs per 100 balls:")
    print(pd.DataFrame({(r, k): v for r in params for k, v in params[r].items()}).round(2).T.to_string())
    res.to_csv(os.path.join(DIAG, "reliability.csv"))
    json.dump(params, open(os.path.join(DIAG, "skill_spread.json"), "w"), indent=1)
    print("Step 5 done")


if __name__ == "__main__":
    main()
