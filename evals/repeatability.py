"""Skill or luck: does a pressure stat from earlier seasons carry over to the next season?"""
import os

import numpy as np
import pandas as pd

from _paths import EOUT
from superr_selector.config import INTERIM

SPLITS = [(2023, 2024), (2024, 2025), (2025, 2026)]      # (last training year, test year)
MIN_TRAIN, MIN_TEST = 80, 40                             # pressure balls needed on each side

ATTACK = ["Slog", "Slog Sweep", "Pull shot", "Lofted Drive", "Hook", "Upper Cut", "Scoop", "Reverse Sweep", "Switch Hit", "Ramp"]
SLOWER = ["Off Cutter", "Leg Cutter", "Slower Ball", "Back of Hand", "Knuckle Ball"]


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def forward(d, pid, stat):
    """Pooled correlation between earlier-seasons value and next-season value."""
    parts = []
    for last, test_year in SPLITS:
        a = d[d.year <= last].groupby(pid)[stat].agg(["mean", "size"])
        b = d[d.year == test_year].groupby(pid)[stat].agg(["mean", "size"])
        j = a.join(b, lsuffix="_a", rsuffix="_b").dropna()
        j = j[(j.size_a >= MIN_TRAIN) & (j.size_b >= MIN_TEST)]
        if len(j) < 5:
            continue
        j["x"] = j.mean_a - np.average(j.mean_a, weights=j.size_b)       # centre within the season
        j["y"] = j.mean_b - np.average(j.mean_b, weights=j.size_b)
        parts.append(j)
    j = pd.concat(parts)
    return len(j), wcorr(j.x.values, j.y.values, j.size_b.values)


def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
    bbb = bbb[bbb.top_tier].copy()
    boundary = bbb.is_four | bbb.is_six
    bbb["boundary"] = boundary.astype(float)
    bbb["six"] = bbb.is_six.astype(float)
    bbb["dot_bat"] = (bbb.runs == 0).astype(float)
    bbb["out"] = bbb.striker_out.astype(float)
    bbb["attack_shot"] = bbb.stroke.isin(ATTACK).astype(float)
    bbb["dot_bowl"] = (bbb.bowler_runs == 0).astype(float)
    bbb["wicket"] = bbb.bowler_wicket.astype(float)
    bbb["extra_ball"] = (~bbb.is_legal).astype(float)
    bbb["yorker"] = (bbb.delivery_style == "Yorker").astype(float)
    bbb["slower"] = bbb.delivery_style.isin(SLOWER).astype(float)
    faced, pace = bbb.is_ball_faced, bbb.pace_or_spin == "Pace"
    everything = pd.Series(True, index=bbb.index)

    stats = [
        ("bat", "striker_id", faced, "boundary", "boundary %", "raw"),
        ("bat", "striker_id", faced, "six", "six %", "raw"),
        ("bat", "striker_id", faced, "attack_shot", "attacking-shot %", "raw"),
        ("bat", "striker_id", faced, "runs", "strike rate", "raw"),
        ("bat", "striker_id", faced, "dot_bat", "dot-ball %", "raw"),
        ("bat", "striker_id", faced, "out", "dismissal %", "raw"),
        ("bat", "striker_id", faced, "bat_p_boundary", "boundary runs above expected", "value"),
        ("bat", "striker_id", faced, "bat_p_rotation", "other runs above expected", "value"),
        ("bat", "striker_id", faced, "bat_p_wicket", "wicket kept, in runs", "value"),
        ("bat", "striker_id", faced, "bat_value", "total value above expected", "value"),
        ("bowl", "bowler_id", pace, "slower", "slower-ball / cutter % (pace)", "raw"),
        ("bowl", "bowler_id", everything, "dot_bowl", "dot-ball %", "raw"),
        ("bowl", "bowler_id", everything, "boundary", "boundary conceded %", "raw"),
        ("bowl", "bowler_id", everything, "extra_ball", "wides + no-balls %", "raw"),
        ("bowl", "bowler_id", pace, "yorker", "yorker % (pace)", "raw"),
        ("bowl", "bowler_id", everything, "bowler_runs", "economy", "raw"),
        ("bowl", "bowler_id", everything, "wicket", "wicket %", "raw"),
        ("bowl", "bowler_id", everything, "bowl_p_boundary", "boundary runs saved vs expected", "value"),
        ("bowl", "bowler_id", everything, "bowl_p_rotation", "other runs saved vs expected", "value"),
        ("bowl", "bowler_id", everything, "bowl_p_wicket", "wickets taken, in runs", "value"),
        ("bowl", "bowler_id", everything, "bowl_value", "total value above expected", "value"),
    ]
    labels = {"bat": [("cricket situations", "S_bat_any"), ("model label", "high_pressure")],
              "bowl": [("cricket situations", "S_bowl_any"), ("model label", "high_pressure")]}
    rows = []
    for role, pid, mask, stat, name, kind in stats:
        row = {"role": role, "statistic": name, "kind": kind}
        for label, col in labels[role]:
            n, r = forward(bbb[mask & bbb[col]], pid, stat)
            row[f"repeat: {label}"] = r
            row[f"player-seasons: {label}"] = n
        rows.append(row)
    res = pd.DataFrame(rows).round(2)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(EOUT, "repeatability.csv"), index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
