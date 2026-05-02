from typing import List

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


SIMILARITY_FEATURES: List[str] = [
    "age",
    "height_cm",
    "weight_kg",
    "goals_per_90",
    "assists_per_90",
    "contribution_per_90",
    "minutes_played",
    "market_value_eur",
    "availability_score",
    "talent_score",
]


def recommend_similar_players(
    players: pd.DataFrame,
    target_player: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend similar U18 players using scaled cosine similarity."""
    if players.empty:
        raise ValueError("Cannot recommend similar players from an empty DataFrame.")

    if target_player not in set(players["player_name"]):
        raise ValueError(f"Target player '{target_player}' was not found.")

    available_features = [
        feature for feature in SIMILARITY_FEATURES if feature in players.columns
    ]
    if not available_features:
        raise ValueError("No similarity features are available.")

    feature_matrix = (
        players[available_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    scaled_features = StandardScaler().fit_transform(feature_matrix)
    similarity_matrix = cosine_similarity(scaled_features)

    target_index = players.index[players["player_name"] == target_player][0]
    similarity_scores = similarity_matrix[target_index]

    recommendations = players.copy()
    recommendations["raw_similarity"] = similarity_scores
    recommendations = recommendations[recommendations.index != target_index].copy()
    recommendations = recommendations.sort_values("raw_similarity", ascending=False)
    recommendations = recommendations.head(top_n)

    recommendations["similarity_score"] = (
        ((recommendations["raw_similarity"] + 1) / 2).clip(0, 1).round(4)
    )
    recommendations["target_player"] = target_player
    recommendations["similar_player"] = recommendations["player_name"]

    return recommendations[
        [
            "target_player",
            "similar_player",
            "similarity_score",
            "position",
            "club",
            "age",
            "talent_score",
        ]
    ].reset_index(drop=True)
