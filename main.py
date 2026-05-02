from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_player_data
from src.feature_engineering import engineer_features
from src.recommendation import recommend_similar_players
from src.scoring_model import calculate_talent_scores, rank_players
from src.visualization import create_visualizations


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_players.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


RANKED_OUTPUT_COLUMNS = [
    "player_id",
    "player_name",
    "age",
    "nationality",
    "position",
    "club",
    "league",
    "talent_score",
    "performance_score",
    "development_score",
    "playing_time_score",
    "league_context_score",
    "age_advantage_score",
    "availability_score",
]


def main() -> None:
    """Run the ScoutAI sample scouting pipeline from raw data to outputs."""
    print("=" * 72)
    print("ScoutAI: U18 Football Talent Scouting Decision-Support Pipeline")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_players = pd.read_csv(DATA_PATH)
    clean_players = clean_player_data(raw_players)
    u18_players = clean_players[clean_players["age"] < 18].copy()

    if u18_players.empty:
        raise ValueError("No U18 players were found after cleaning the dataset.")

    featured_players = engineer_features(u18_players)
    scored_players = calculate_talent_scores(featured_players)
    ranked_players = rank_players(scored_players)

    ranked_output = ranked_players[RANKED_OUTPUT_COLUMNS].copy()
    top_players_path = OUTPUT_DIR / "top_u18_players.csv"
    ranked_output.to_csv(top_players_path, index=False)

    target_player = ranked_players.iloc[0]["player_name"]
    similar_players = recommend_similar_players(
        ranked_players,
        target_player=target_player,
        top_n=5,
    )
    similar_players_path = OUTPUT_DIR / "similar_players.csv"
    similar_players.to_csv(similar_players_path, index=False)

    created_figures = create_visualizations(ranked_players, FIGURES_DIR)

    print(f"Loaded sample rows: {len(raw_players)}")
    print(f"Clean rows after validation: {len(clean_players)}")
    print(f"U18 players ranked: {len(ranked_players)}")
    print()
    print("Top 5 U18 players by Talent Score:")
    print(
        ranked_output[
            ["player_name", "age", "position", "club", "talent_score"]
        ]
        .head(5)
        .to_string(index=False)
    )
    print()
    print(f"Similar-player target: {target_player}")
    print(
        similar_players[
            ["similar_player", "similarity_score", "position", "club", "talent_score"]
        ].to_string(index=False)
    )
    print()
    print("Saved outputs:")
    print(f"- {top_players_path.relative_to(PROJECT_ROOT)}")
    print(f"- {similar_players_path.relative_to(PROJECT_ROOT)}")
    for figure_path in created_figures:
        print(f"- {figure_path.relative_to(PROJECT_ROOT)}")
    print()
    print("Note: ScoutAI is a decision-support tool. Human scouting review remains essential.")


if __name__ == "__main__":
    main()
