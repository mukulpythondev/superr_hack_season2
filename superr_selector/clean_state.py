"""Clean the ball-by-ball data and rebuild the match state before every ball."""
import os, pandas as pd
from superr_selector.config import DATA, INTERIM


def main():
    bbb     = pd.read_csv(os.path.join(DATA, "ball_by_ball.csv"), low_memory=False)
    matches = pd.read_csv(os.path.join(DATA, "matches.csv"))
    comps   = pd.read_csv(os.path.join(DATA, "competitions.csv"))

    # drop super overs
    bbb = bbb[bbb.innings.isin([1, 2])].copy()

    # drop rain-affected matches (revised overs / targets) and matches without a winner
    rain_cols = ["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]
    bad = matches[matches[rain_cols].notna().any(axis=1) | matches.win_team_id.isna()].match_id
    bbb = bbb[~bbb.match_id.isin(bad)].copy()

    # helper columns
    NOBALL = ["Overstep", "other"]
    bbb["total_runs"]    = bbb.runs + bbb.extras1_runs + bbb.extras_2_runs
    bbb["is_legal"]      = ~bbb.extras_type.isin(["wide"] + NOBALL)
    bbb["is_ball_faced"] = bbb.extras_type != "wide"
    # runs charged to the bowler: everything except byes, leg-byes and penalties
    not_bowler = bbb.extras_type.isin(["byes", "legbyes", "Penalty"])
    bbb["bowler_runs"]   = bbb.total_runs - bbb.extras1_runs.where(not_bowler, 0)
    # wickets credited to the bowler (no run-outs etc.)
    bbb["bowler_wicket"] = bbb.is_wicket & bbb.dismissal_type.isin(["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"])
    # striker dismissed on this ball (a run-out of the non-striker is not the striker's fault)
    bbb["striker_out"]   = bbb.is_wicket & (bbb.dismissed_player_id == bbb.striker_id) & (bbb.dismissal_type != "Retire Out")

    # match situation before each ball
    bbb = bbb.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    g = bbb.groupby(["match_id", "innings"])
    bbb["score_before"]    = g.total_runs.cumsum() - bbb.total_runs
    bbb["wickets_before"]  = g.is_wicket.cumsum() - bbb.is_wicket
    bbb["balls_bowled"]    = g.is_legal.cumsum() - bbb.is_legal
    bbb["balls_left"]      = 120 - bbb.balls_bowled
    bbb["wickets_in_hand"] = 10 - bbb.wickets_before

    first_total = bbb[bbb.innings == 1].groupby("match_id").total_runs.sum().rename("first_total")
    bbb = bbb.merge(first_total, on="match_id", how="left")
    chase = bbb.innings == 2
    bbb["runs_needed"] = (bbb.first_total + 1 - bbb.score_before).where(chase)
    bbb["required_rr"] = (bbb.runs_needed / bbb.balls_left * 6).where(chase)

    # match labels
    info = matches[["match_id", "comp_id", "match_date", "win_team_id"]].merge(
        comps[["comp_id", "comp_name", "match_format", "season"]], on="comp_id")
    info["league"] = "Other"
    info.loc[info.match_format == "T20I", "league"] = "T20I"
    info.loc[info.comp_name.str.startswith("IPL"), "league"] = "IPL"
    info["year"] = pd.to_datetime(info.match_date).dt.year
    # top tier = IPL, or a T20I between two ICC full-member nations
    FULL = ["India", "Pakistan", "Australia", "England", "New Zealand", "South Africa", "West Indies",
            "Sri Lanka", "Bangladesh", "Afghanistan", "Zimbabwe", "Ireland"]
    mt = matches[["match_id", "home_team", "away_team"]]
    both_full = mt.home_team.isin(FULL) & mt.away_team.isin(FULL)
    info = info.merge(mt.assign(both_full=both_full)[["match_id", "both_full"]], on="match_id")
    info["top_tier"] = (info.league == "IPL") | ((info.league == "T20I") & info.both_full)
    bbb = bbb.merge(info[["match_id", "league", "year", "win_team_id", "top_tier"]], on="match_id")
    bbb["bat_team_won"] = (bbb.batting_team_id == bbb.win_team_id).astype(int)

    bbb.to_csv(os.path.join(INTERIM, "bbb_state.csv"), index=False)
    print(f"Step 1 done: {len(bbb):,} balls, {bbb.match_id.nunique()} matches")
    print(bbb.groupby(["league", "top_tier"]).match_id.nunique())


if __name__ == "__main__":
    main()
