from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def create_visualizations(players: pd.DataFrame, figures_dir: Path) -> List[Path]:
    """Create and save all ScoutAI visualizations."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_top_talent_scores(players, figures_dir),
        _plot_score_breakdown(players, figures_dir),
        _plot_position_distribution(players, figures_dir),
    ]
    return paths


def _plot_top_talent_scores(players: pd.DataFrame, figures_dir: Path) -> Path:
    top_players = players.nlargest(10, "talent_score").sort_values("talent_score")
    output_path = figures_dir / "top_talent_scores.png"

    plt.figure(figsize=(10, 6))
    plt.barh(top_players["player_name"], top_players["talent_score"], color="#2E7D6B")
    plt.xlabel("Talent Score")
    plt.title("Top 10 U18 Players by Talent Score")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_score_breakdown(players: pd.DataFrame, figures_dir: Path) -> Path:
    top_players = players.nlargest(5, "talent_score")
    weighted_components = [
        ("performance_score", 0.35, "Performance"),
        ("development_score", 0.25, "Development"),
        ("playing_time_score", 0.15, "Playing Time"),
        ("league_context_score", 0.15, "League Context"),
        ("age_advantage_score", 0.10, "Age Advantage"),
    ]
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD", "#D62728"]
    output_path = figures_dir / "score_breakdown.png"

    plt.figure(figsize=(11, 6))
    bottom = [0] * len(top_players)
    for (component, weight, label), color in zip(weighted_components, colors):
        values = top_players[component] * weight
        plt.bar(top_players["player_name"], values, bottom=bottom, label=label, color=color)
        bottom = [current + value for current, value in zip(bottom, values)]

    plt.ylabel("Weighted Contribution to Talent Score")
    plt.title("Score Component Breakdown for Top U18 Players")
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_position_distribution(players: pd.DataFrame, figures_dir: Path) -> Path:
    position_counts = players["position"].value_counts().sort_values(ascending=False)
    output_path = figures_dir / "position_distribution.png"

    plt.figure(figsize=(9, 5))
    plt.bar(position_counts.index, position_counts.values, color="#4C78A8")
    plt.xlabel("Position")
    plt.ylabel("Number of U18 Players")
    plt.title("Position Distribution of U18 Players")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
