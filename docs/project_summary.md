# ScoutAI Project Summary

## Problem

Football clubs often need to identify promising U18 players from incomplete public information. Manual scouting remains essential, but structured data can help scouts prioritize which players to review.

## Solution

ScoutAI is a Python-based decision-support system that ranks U18 football players and recommends similar players using public football-style data. It includes both a command-line pipeline and a Streamlit scouting dashboard. It does not claim to predict future stars automatically.

## Data

The repository uses `data/sample_players.csv`, a small fictional demonstration dataset with player age, position, club, league, playing time, goals, assists, market value, market value growth, availability, and league context. Full public datasets are not uploaded because of file size, licensing, and usage restrictions.

## Model

The pipeline cleans the data, filters U18 players, engineers per-90 and normalized scoring features, calculates a weighted Talent Score from 0 to 100, and uses cosine similarity to recommend players with similar profiles.

## Outputs

Running `python main.py` creates:

- `outputs/top_u18_players.csv`
- `outputs/similar_players.csv`
- `outputs/figures/top_talent_scores.png`
- `outputs/figures/score_breakdown.png`
- `outputs/figures/position_distribution.png`

Running `streamlit run streamlit_app.py` opens the interactive dashboard with overview, data explorer, ranking, player profile, recommendation, regional map, model explanation, visual analytics, ethics, and project information sections. The dashboard includes a Barca-inspired visual theme, football imagery, keyword player search, high-contrast player dossier panels, image-based player cards, motion windows, selectable country maps, and six-axis radar charts for player profiles.

## Ethics

U18 players are minors. ScoutAI should use only legally accessible public data, should not collect private medical records, and should treat injury-related information only as public availability indicators. Results should support human scouts, not replace them.

## Limitations

The project uses a sample dataset, public youth football data is often incomplete, and there is no perfect label for future success. The Talent Score formula is simplified, position differences are not fully modeled, and public data may favor visible clubs and leagues.

## Future Improvements

Future work could integrate real Transfermarkt data, add position-specific scoring, predict market value growth, analyze development over time, build a Streamlit dashboard, improve similar-player search, add league strength adjustment, support club-specific filters, evaluate fairness, and include event-level football data.
