import re
from typing import Dict

import pandas as pd


TEXT_COLUMNS = [
    "player_id",
    "player_name",
    "nationality",
    "position",
    "club",
    "league",
    "preferred_foot",
]

NUMERIC_DEFAULTS: Dict[str, float] = {
    "age": 18,
    "height_cm": 175,
    "weight_kg": 70,
    "appearances": 0,
    "minutes_played": 0,
    "goals": 0,
    "assists": 0,
    "market_value_eur": 0,
    "market_value_growth_pct": 0,
    "availability_score": 70,
    "league_level_score": 50,
}


def _standardize_column_name(column_name: str) -> str:
    cleaned = column_name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def clean_player_data(players: pd.DataFrame) -> pd.DataFrame:
    """Clean raw player data and return a validated DataFrame."""
    if players.empty:
        raise ValueError("The input player dataset is empty.")

    cleaned = players.copy()
    cleaned.columns = [_standardize_column_name(column) for column in cleaned.columns]

    if "player_id" not in cleaned.columns:
        cleaned["player_id"] = [
            f"GEN_{index + 1:04d}" for index in range(len(cleaned))
        ]

    for column in TEXT_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = "Unknown"

    for column, default_value in NUMERIC_DEFAULTS.items():
        if column not in cleaned.columns:
            cleaned[column] = default_value

    for column, default_value in NUMERIC_DEFAULTS.items():
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if column == "age":
            fallback = cleaned[column].median()
            if pd.isna(fallback):
                fallback = default_value
            cleaned[column] = cleaned[column].fillna(fallback)
        else:
            cleaned[column] = cleaned[column].fillna(default_value)

    for column in TEXT_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    missing_ids = cleaned["player_id"].eq("Unknown")
    if missing_ids.any():
        generated_ids = [
            f"GEN_{position + 1:04d}" for position in range(missing_ids.sum())
        ]
        cleaned.loc[missing_ids, "player_id"] = generated_ids

    cleaned = cleaned.drop_duplicates(subset=["player_id"], keep="first")
    cleaned = cleaned.drop_duplicates(
        subset=["player_name", "club", "age"], keep="first"
    )

    cleaned = cleaned[cleaned["age"].between(10, 45)].copy()
    cleaned["age"] = cleaned["age"].round().astype(int)

    non_negative_columns = [
        "height_cm",
        "weight_kg",
        "appearances",
        "minutes_played",
        "goals",
        "assists",
        "market_value_eur",
    ]
    for column in non_negative_columns:
        cleaned[column] = cleaned[column].clip(lower=0)

    cleaned["availability_score"] = cleaned["availability_score"].clip(lower=0, upper=100)
    cleaned["league_level_score"] = cleaned["league_level_score"].clip(lower=0, upper=100)
    cleaned["market_value_growth_pct"] = cleaned["market_value_growth_pct"].clip(
        lower=-100,
        upper=500,
    )

    return cleaned.reset_index(drop=True)
