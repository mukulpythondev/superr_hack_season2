# Method

Data: IPL and T20I ball-by-ball, 2022 to Nov 2026. After dropping super overs, rain-affected
matches and no-results: 818 matches, 195,124 balls. The data has no running score, so score,
wickets, balls left, target and required rate are rebuilt ball by ball (`clean_state.py`).

## Pressure

| | Situation | Definition |
|---|---|---|
| Bat | Collapse | 3+ wickets down, up to over 10 |
| | Chase start | chasing 180+, overs 1-6 |
| | Big chase | required rate 10+, 10 overs or fewer left |
| | Finish | overs 17-20 |
| Bowl | Death | overs 17-20 |
| | Set batter | overs 7-16, striker on 20+ balls or partnership 40+ |
| | Small total | defending 155 or fewer |
| | Bounce-back | previous ball of the over went for 4 or 6 |

All except collapse need the match to be live: win probability 10-90% in a chase, 20-80% batting
first. Knockout balls count 1.5 times.

I first tried letting the model define pressure (win chance after a six minus after a wicket, top
25% of balls). It put half of all chase balls under pressure and almost no first-innings balls,
and gave openers more pressure balls than finishers. That label is kept only as a cross-check.

The win-probability model is a logistic regression on runs needed, balls left, wickets in hand,
required rate and three interactions. 5-fold by match: AUC 0.895 for chases, 0.682 batting first.

## Value

Two gradient-boosted models predict the runs and the wicket chance of an average player on each
ball, from league, year, over, wickets, score, required rate, pace or spin and team strengths.
Each match is predicted by a model that did not see it.

    batter value = (runs - expected) - wicket_value * (out - expected)
    bowler value = -(conceded - expected) + wicket_value * (wicket - expected)

`wicket_value` comes from the win-probability model: roughly 13-16 runs in the powerplay, 10 in
the middle, 4-8 at the death.

## Fair comparison

- Batting lists rank batters and all-rounders, bowling lists bowlers and all-rounders. Roles come
  from the data: usual position 7 or higher or keeper = batter, 12+ balls bowled per match = bowler.
- Only IPL and T20Is between full members count towards a rating.
- Minimum 120 pressure balls, 15 matches, and a match in 2025 or later.
- A player's rating mixes his pressure record with his usual level. The weights come from a
  split-half test, not from hand-set numbers (`rating_core.py`).
- Ranking uses estimate minus 1.28 sd, so a small sample cannot buy a high rank.

## What repeats

Earlier seasons against the next season, on pressure balls (`evals/repeatability.py`):

| Batters | r | Bowlers | r |
|---|---|---|---|
| Boundary % | 0.42 | Slower balls and cutters % | 0.74 |
| Dot % | 0.33 | Wides and no-balls % | 0.51 |
| Six % | 0.30 | Yorker % | 0.50 |
| Strike rate | 0.28 | Dot % | 0.35 |
| Dismissal % | 0.13 | Boundary conceded % | 0.34 |
| | | Economy | 0.22 |
| | | Wicket % | 0.09 |

Boundary hitting, dot balls and execution carry over. Wickets under pressure mostly do not.

## Limits

- Pressure value as a whole repeats weakly (split-half r about 0.05), so neighbouring ranks are
  not real differences.
- The rating has not been shown to predict next-season pressure numbers better than plain strike
  rate or economy.
- Baselines and team strengths were fitted on all seasons, so the forward tests are not strictly
  past-only.
- The batting list moves most with the wicket value: fixing it at 8 runs keeps 5 of the 10.
- Opposition is adjusted by team, not by the bowler or batter faced.
