from typing import Mapping, Optional

import pandas as pd


SCORE_WEIGHTS = {
    "performance_score": 0.35,
    "development_score": 0.25,
    "playing_time_score": 0.15,
    "league_context_score": 0.15,
    "age_advantage_score": 0.10,
}


def calculate_talent_scores(
    players: pd.DataFrame,
    weights: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Calculate a weighted Talent Score between 0 and 100."""
    scored = players.copy()
    score_weights = dict(weights or SCORE_WEIGHTS)

    missing_columns = [
        column for column in score_weights if column not in scored.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing score component columns: {missing_columns}")

    scored["talent_score"] = 0.0
    for column, weight in score_weights.items():
        scored["talent_score"] += scored[column].fillna(0) * weight

    scored["talent_score"] = scored["talent_score"].clip(0, 100).round(2)
    return scored


def rank_players(players: pd.DataFrame) -> pd.DataFrame:
    """Sort players by Talent Score and stable identity fields."""
    return (
        players.sort_values(
            by=["talent_score", "performance_score", "development_score", "player_name"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
