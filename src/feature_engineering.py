import numpy as np
import pandas as pd


def normalize_to_100(series: pd.Series) -> pd.Series:
    """Scale a numeric series to a 0 to 100 range."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = numeric.min()
    maximum = numeric.max()
    if np.isclose(maximum, minimum):
        return pd.Series(50.0, index=series.index)
    return ((numeric - minimum) / (maximum - minimum) * 100).clip(0, 100)


def _per_90(counts: pd.Series, minutes: pd.Series) -> pd.Series:
    safe_minutes = pd.to_numeric(minutes, errors="coerce").fillna(0)
    safe_counts = pd.to_numeric(counts, errors="coerce").fillna(0)
    return pd.Series(
        np.where(safe_minutes > 0, (safe_counts * 90) / safe_minutes, 0),
        index=counts.index,
    )


def engineer_features(players: pd.DataFrame) -> pd.DataFrame:
    """Create scouting features used by ranking and recommendation models."""
    featured = players.copy()

    featured["goals_per_90"] = _per_90(featured["goals"], featured["minutes_played"])
    featured["assists_per_90"] = _per_90(
        featured["assists"],
        featured["minutes_played"],
    )
    featured["contribution_per_90"] = featured["goals_per_90"] + featured["assists_per_90"]

    featured["minutes_score"] = normalize_to_100(featured["minutes_played"])
    featured["market_value_score"] = normalize_to_100(featured["market_value_eur"])
    featured["market_growth_score"] = normalize_to_100(
        featured["market_value_growth_pct"]
    )
    featured["availability_score_normalized"] = normalize_to_100(
        featured["availability_score"]
    )
    featured["league_level_score_normalized"] = normalize_to_100(
        featured["league_level_score"]
    )

    contribution_score = normalize_to_100(featured["contribution_per_90"])
    goals_score = normalize_to_100(featured["goals_per_90"])
    assists_score = normalize_to_100(featured["assists_per_90"])
    featured["performance_score"] = (
        0.55 * contribution_score + 0.25 * goals_score + 0.20 * assists_score
    ).round(2)

    featured["development_score"] = (
        0.60 * featured["market_growth_score"]
        + 0.40 * featured["market_value_score"]
    ).round(2)

    featured["playing_time_score"] = (
        0.70 * featured["minutes_score"]
        + 0.30 * featured["availability_score_normalized"]
    ).round(2)

    featured["league_context_score"] = featured["league_level_score_normalized"].round(2)

    age_advantage = ((18 - featured["age"]) / 3) * 100
    featured["age_advantage_score"] = age_advantage.clip(0, 100).round(2)

    return featured
