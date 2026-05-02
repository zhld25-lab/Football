# ScoutAI: A Data-Driven Youth Football Scouting System for Identifying U18 Football Talent

## 1. Project Title

**ScoutAI: A Data-Driven Youth Football Scouting System for Identifying U18 Football Talent**

## 2. Project Overview

ScoutAI is a Python data science project that helps football scouts review promising U18 players using structured public football-style data. The system cleans player data, engineers scouting features, calculates a weighted Talent Score, ranks U18 players, recommends similar players, and generates CSV outputs and charts.

The project is designed for a university data science course. It uses a small sample dataset by default so the repository can run immediately with:

```bash
python main.py
```

ScoutAI is not an automatic talent prediction system. It is a decision-support tool that helps scouts organize evidence and prioritize review. Human scouts, coaches, analysts, and recruitment staff should make the final decisions.

## 3. Problem Statement

Youth football scouting is difficult because clubs often evaluate many players with incomplete information. Public data can include age, position, nationality, club, league, playing time, goals, assists, market value, and availability indicators, but these values are spread across different sources and are not always directly comparable.

This project asks:

> How can public football data be transformed into a simple, transparent scoring and recommendation system that supports U18 scouting decisions?

## 4. Motivation

Football clubs want to identify talented young players early, but they must avoid over-relying on reputation, club visibility, or raw goal totals. A transparent data pipeline can help scouts:

- Compare players using consistent feature definitions.
- Balance performance, development, playing time, league context, and age.
- Find similar players for deeper review.
- Understand why a player receives a high score.

The motivation is to support better scouting conversations, not to replace expert judgement.

## 5. Proposed Solution

ScoutAI provides an end-to-end Python pipeline:

1. Load `data/sample_players.csv`.
2. Clean and validate the dataset.
3. Filter players under 18 years old.
4. Engineer performance and development features.
5. Calculate a weighted Talent Score from 0 to 100.
6. Rank U18 players.
7. Recommend similar players using cosine similarity.
8. Save CSV outputs.
9. Generate charts with matplotlib.
10. Print a clean terminal summary.

## 6. Data Sources

The repository runs with a small fictional demonstration dataset:

| File | Purpose |
| --- | --- |
| `data/sample_players.csv` | Small sample dataset used by default |
| `data/readme_data.txt` | Notes about sample data, full sources, and responsible use |

Recommended real-world sources for future expansion:

| Source | Link | Possible Use |
| --- | --- | --- |
| Transfermarkt datasets | <https://github.com/dcaribou/transfermarkt-datasets> | Player identity, clubs, competitions, appearances, market values |
| Football Manager 2023 dataset | <https://www.kaggle.com/datasets/siddhrajthakor/football-manager-2023-dataset> | Supplementary technical and physical attributes |

The sample data is fictional and exists only to demonstrate the method.

## 7. Why Full Datasets Are Not Uploaded

Full public football datasets are not uploaded to this repository because:

- They can be too large for a small coursework repository.
- Their licenses and terms of use may restrict redistribution.
- Data providers may update records over time.
- Responsible data use requires users to check the original source terms.

Users who want to work with full datasets should download them directly from the original links and adapt the data-loading step.

## 8. Repository Structure

```text
Football/
|-- README.md
|-- requirements.txt
|-- main.py
|-- data/
|   |-- sample_players.csv
|   `-- readme_data.txt
|-- src/
|   |-- data_cleaning.py
|   |-- feature_engineering.py
|   |-- scoring_model.py
|   |-- recommendation.py
|   `-- visualization.py
|-- outputs/
|   |-- top_u18_players.csv
|   |-- similar_players.csv
|   `-- figures/
|       |-- top_talent_scores.png
|       |-- score_breakdown.png
|       `-- position_distribution.png
`-- docs/
    `-- project_summary.md
```

## 9. Methodology

ScoutAI uses a transparent scoring pipeline rather than a black-box prediction model. This is appropriate for a course project because the logic can be inspected, explained, and challenged.

| Stage | Description |
| --- | --- |
| Data loading | Reads the sample CSV file |
| Cleaning | Standardizes columns, converts numeric fields, fills missing values, validates age |
| Filtering | Keeps only players younger than 18 |
| Feature engineering | Creates per-90, normalized, performance, development, playing time, context, and age scores |
| Ranking | Calculates a weighted Talent Score |
| Recommendation | Uses cosine similarity to find players with similar profiles |
| Visualization | Saves charts for interpretation and presentation |

## 10. Data Cleaning

Implemented in `src/data_cleaning.py`.

The cleaning step:

- Standardizes column names.
- Removes duplicate players.
- Converts numeric columns safely.
- Fills missing numeric values with reasonable defaults.
- Fills missing text values with `"Unknown"`.
- Validates age values.
- Clips availability and league scores to the 0 to 100 range.

## 11. Feature Engineering

Implemented in `src/feature_engineering.py`.

The engineered features include:

| Feature | Meaning |
| --- | --- |
| `goals_per_90` | Goals adjusted to 90 minutes |
| `assists_per_90` | Assists adjusted to 90 minutes |
| `contribution_per_90` | Goals plus assists adjusted to 90 minutes |
| `minutes_score` | Normalized playing time score |
| `performance_score` | Combined attacking contribution score |
| `development_score` | Market value and market growth signal |
| `playing_time_score` | Minutes and availability signal |
| `league_context_score` | Normalized league level signal |
| `age_advantage_score` | Rewards younger players within the U18 group |
| `market_value_score` | Normalized market value |
| `availability_score_normalized` | Normalized public availability indicator |
| `league_level_score_normalized` | Normalized league level |

The code avoids division by zero when calculating per-90 statistics.

## 12. Models and Algorithms

ScoutAI uses two simple and explainable methods:

| Task | Method |
| --- | --- |
| Talent ranking | Weighted scoring model |
| Similar-player recommendation | Scaled cosine similarity |

The project uses `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

## 13. Talent Score Formula

Implemented in `src/scoring_model.py`.

The Talent Score is calculated from 0 to 100:

```text
Talent Score =
35% Performance Score
+ 25% Development / Market Value Score
+ 15% Playing Time Stability Score
+ 15% League / Club Context Score
+ 10% Age Advantage Score
```

The implementation uses normalized component scores. The result is a transparent ranking signal, not a guarantee of future success.

## 14. Similar Player Recommendation

Implemented in `src/recommendation.py`.

The recommendation model selects the highest-ranked player as the target and recommends the top 5 most similar U18 players using cosine similarity. The similarity features include:

- Age
- Height
- Weight
- Goals per 90
- Assists per 90
- Contribution per 90
- Minutes played
- Market value
- Availability score
- Talent Score

The output is saved to `outputs/similar_players.csv`.

## 15. Visualizations

Implemented in `src/visualization.py` using matplotlib only.

The pipeline generates:

| Chart | Output Path |
| --- | --- |
| Top 10 U18 Players by Talent Score | `outputs/figures/top_talent_scores.png` |
| Score Component Breakdown for Top Players | `outputs/figures/score_breakdown.png` |
| Position Distribution of U18 Players | `outputs/figures/position_distribution.png` |

## 16. How to Run the Project

Clone the repository:

```bash
git clone https://github.com/zhld25-lab/Football.git
cd Football
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py
```

## 17. Example Outputs

After running the project, the generated files are:

| Output | Description |
| --- | --- |
| `outputs/top_u18_players.csv` | Ranked U18 players with score components |
| `outputs/similar_players.csv` | Similar players for the top-ranked target player |
| `outputs/figures/top_talent_scores.png` | Bar chart of top U18 Talent Scores |
| `outputs/figures/score_breakdown.png` | Component breakdown chart |
| `outputs/figures/position_distribution.png` | Position distribution chart |

The terminal also prints the number of loaded rows, U18 rows, the top 5 ranked players, the similar-player target, and saved output paths.

## 18. Ethical Considerations

U18 football players are minors, so the project must be used carefully and responsibly.

- The system should use only legally accessible public data.
- The system should not collect private medical records.
- Injury-related information should be treated only as public availability indicators, such as appearances, minutes played, missed matches, or publicly reported absence information.
- The model should support human decision-making, not replace scouts.
- Public football data may be biased toward famous clubs and visible leagues.
- The system may overlook players from smaller academies or less visible regions.
- The results should not be used as final recruitment decisions.
- Human scouts should review match footage, training context, personality, coaching reports, and live performance before making any recruitment recommendation.

ScoutAI should be used to organize evidence and encourage structured discussion, not to label young players permanently.

## 19. Limitations

This project has important limitations:

- The repository uses a sample dataset.
- Youth football data is often incomplete.
- There is no perfect label for future success.
- The Talent Score formula is simplified.
- Different positions require different evaluation standards.
- Public data may be biased.
- A real club would need official data partnerships and human scouting review.

The current version is best understood as a reproducible prototype for learning data science workflow design.

## 20. Future Improvements

Possible future improvements include:

- Real Transfermarkt dataset integration.
- Position-specific scoring models.
- Market value growth prediction.
- Time-series player development analysis.
- Streamlit dashboard.
- More advanced similar-player search.
- League strength adjustment.
- Club-specific scouting filters.
- Fairness and bias evaluation.
- Event-level football data integration.

## 21. References

- Transfermarkt datasets: <https://github.com/dcaribou/transfermarkt-datasets>
- Football Manager 2023 dataset: <https://www.kaggle.com/datasets/siddhrajthakor/football-manager-2023-dataset>
- pandas documentation: <https://pandas.pydata.org/docs/>
- NumPy documentation: <https://numpy.org/doc/>
- scikit-learn documentation: <https://scikit-learn.org/stable/>
- matplotlib documentation: <https://matplotlib.org/stable/>
