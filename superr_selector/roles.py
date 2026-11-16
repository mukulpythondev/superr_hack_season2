"""Batter / bowler / all-rounder, inferred from batting position, keeping and overs bowled."""
import os

import numpy as np
import pandas as pd

from superr_selector.config import DATA, DIAG, INTERIM

TOP_ORDER_MAX_POSITION = 7
KEEPER_SHARE = 0.30
BOWLS_REGULARLY_BALLS_PER_MATCH = 12
BATTING_ROLES = ("batter", "all-rounder")
BOWLING_ROLES = ("bowler", "all-rounder")


def player_roles(bbb, playing_eleven):
    xi = playing_eleven[playing_eleven.is_played.astype(bool)]
    played = xi.groupby("player_id").agg(matches=("match_id", "nunique"), keeper_share=("is_wk", "mean"))

    faced = bbb[bbb.is_ball_faced]
    first = faced.groupby(["match_id", "innings", "striker_id"]).innings_ball_num.min().reset_index()
    first["position"] = first.groupby(["match_id", "innings"]).innings_ball_num.rank(method="first")
    batting = first.groupby("striker_id").agg(innings_batted=("position", "size"), usual_position=("position", "median"))
    bowling = bbb.groupby("bowler_id").agg(balls_bowled=("runs", "size"), matches_bowled=("match_id", "nunique"))

    r = played.join(batting, how="outer").join(bowling, how="outer")
    r["matches"] = r.matches.fillna(r[["innings_batted", "matches_bowled"]].max(axis=1))
    r["balls_bowled_per_match"] = (r.balls_bowled / r.matches).fillna(0.0)
    top7 = r.usual_position <= TOP_ORDER_MAX_POSITION
    keeper = r.keeper_share.fillna(0) >= KEEPER_SHARE
    bats = top7 | keeper
    bowls = r.balls_bowled_per_match >= BOWLS_REGULARLY_BALLS_PER_MATCH
    r["role"] = np.select([bats & bowls, bats, bowls], ["all-rounder", "batter", "bowler"], default="other")
    r.index.name = "player_id"
    return r[["matches", "innings_batted", "usual_position", "keeper_share", "balls_bowled_per_match", "role"]]


def load_roles(bbb):
    xi = pd.read_csv(os.path.join(DATA, "playing_eleven.csv"))
    return player_roles(bbb, xi)


def main():
    bbb = pd.read_csv(os.path.join(INTERIM, "bbb_value.csv"), low_memory=False)
    roles = load_roles(bbb)
    roles.round(2).to_csv(os.path.join(DIAG, "player_roles.csv"), lineterminator="\n")
    print(roles.role.value_counts().to_string())


if __name__ == "__main__":
    main()
