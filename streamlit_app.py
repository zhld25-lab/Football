from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_cleaning import clean_player_data
from src.feature_engineering import engineer_features
from src.recommendation import recommend_similar_players
from src.scoring_model import SCORE_WEIGHTS, calculate_talent_scores, rank_players
from src.visualization import create_visualizations


st.set_page_config(
    page_title="ScoutAI | Youth Football Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_players.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

REQUIRED_UPLOAD_COLUMNS = [
    "player_name",
    "age",
    "position",
    "club",
    "league",
    "appearances",
    "minutes_played",
    "goals",
    "assists",
]

OPTIONAL_UPLOAD_COLUMNS = [
    "player_id",
    "nationality",
    "height_cm",
    "weight_kg",
    "preferred_foot",
    "market_value_eur",
    "market_value_growth_pct",
    "availability_score",
    "league_level_score",
]

SCORE_COMPONENTS = [
    "performance_score",
    "development_score",
    "playing_time_score",
    "league_context_score",
    "age_advantage_score",
]

COMPONENT_LABELS = {
    "performance_score": "Performance",
    "development_score": "Development",
    "playing_time_score": "Playing Time",
    "league_context_score": "League Context",
    "age_advantage_score": "Age Advantage",
}

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

PROFILE_COLUMNS = [
    "player_name",
    "age",
    "nationality",
    "position",
    "club",
    "league",
    "height_cm",
    "weight_kg",
    "preferred_foot",
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "market_value_eur",
    "availability_score",
    "talent_score",
]

ANALYTIC_COLUMNS = [
    "goals_per_90",
    "assists_per_90",
    "contribution_per_90",
    "minutes_score",
    "market_value_score",
    "market_growth_score",
    "availability_score_normalized",
    "league_level_score_normalized",
    "performance_score",
    "development_score",
    "playing_time_score",
    "league_context_score",
    "age_advantage_score",
    "talent_score",
]


def standardize_column_name(column_name: str) -> str:
    cleaned = column_name.strip().lower()
    for character in [" ", "-", "/", ".", "(", ")"]:
        cleaned = cleaned.replace(character, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def init_session_state() -> None:
    st.session_state.setdefault("selected_player", None)
    st.session_state.setdefault("uploaded_dataset_name", None)
    for component, weight in SCORE_WEIGHTS.items():
        st.session_state.setdefault(weight_key(component), int(round(weight * 100)))
        st.session_state.setdefault(
            model_weight_key(component),
            int(round(weight * 100)),
        )


def weight_key(component: str) -> str:
    return f"weight_{component}"


def model_weight_key(component: str) -> str:
    return f"model_weight_{component}"


def sync_model_weight(component: str) -> None:
    st.session_state[weight_key(component)] = st.session_state[model_weight_key(component)]


def reset_weights() -> None:
    for component, weight in SCORE_WEIGHTS.items():
        st.session_state[weight_key(component)] = int(round(weight * 100))
        st.session_state[model_weight_key(component)] = int(round(weight * 100))


def get_weight_percentages() -> Dict[str, int]:
    return {
        component: int(st.session_state.get(weight_key(component), weight * 100))
        for component, weight in SCORE_WEIGHTS.items()
    }


def get_normalized_weights() -> Dict[str, float]:
    percentages = get_weight_percentages()
    total = sum(percentages.values())
    if total <= 0:
        return dict(SCORE_WEIGHTS)
    return {
        component: percentage / total
        for component, percentage in percentages.items()
    }


def get_weight_signature() -> Tuple[Tuple[str, float], ...]:
    normalized = get_normalized_weights()
    return tuple((component, normalized[component]) for component in SCORE_COMPONENTS)


def format_money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"EUR {value:,.0f}"


def format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_uploaded_data(uploaded_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(uploaded_bytes))


def validate_columns(dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
    standardized_columns = {
        standardize_column_name(column) for column in dataframe.columns
    }
    missing_required = [
        column for column in REQUIRED_UPLOAD_COLUMNS if column not in standardized_columns
    ]
    missing_optional = [
        column for column in OPTIONAL_UPLOAD_COLUMNS if column not in standardized_columns
    ]
    return missing_required, missing_optional


@st.cache_data(show_spinner="Preparing scouting data...")
def run_pipeline(
    raw_players: pd.DataFrame,
    weight_signature: Tuple[Tuple[str, float], ...],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_players = clean_player_data(raw_players)
    u18_players = clean_players[clean_players["age"] < 18].copy()

    if u18_players.empty:
        dashboard_players = clean_players.copy()
        for column in ANALYTIC_COLUMNS:
            dashboard_players[column] = np.nan
        return clean_players, dashboard_players, pd.DataFrame()

    featured_players = engineer_features(u18_players)
    scored_players = calculate_talent_scores(
        featured_players,
        weights=dict(weight_signature),
    )
    ranked_players = rank_players(scored_players)

    dashboard_players = clean_players.merge(
        ranked_players[["player_id"] + ANALYTIC_COLUMNS],
        on="player_id",
        how="left",
    )
    return clean_players, dashboard_players, ranked_players


def ensure_output_files(ranked_players: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if ranked_players.empty:
        return

    output_columns = [
        column for column in RANKED_OUTPUT_COLUMNS if column in ranked_players.columns
    ]
    ranked_players[output_columns].to_csv(
        OUTPUT_DIR / "top_u18_players.csv",
        index=False,
    )

    target_player = ranked_players.iloc[0]["player_name"]
    similar_players = recommend_similar_players(
        ranked_players,
        target_player=target_player,
        top_n=5,
    )
    similar_players.to_csv(OUTPUT_DIR / "similar_players.csv", index=False)
    create_visualizations(ranked_players, FIGURES_DIR)


def render_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }
        .small-muted {
            color: #64748b;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_controls() -> Tuple[pd.DataFrame, str]:
    st.sidebar.markdown("## ⚽ ScoutAI")
    st.sidebar.caption("Data-driven U18 football scouting dashboard")
    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader(
        "Upload a scouting CSV",
        type=["csv"],
        help="If uploaded, a valid CSV replaces the sample dataset for this session.",
    )

    if uploaded_file is None:
        st.session_state.uploaded_dataset_name = None
        st.sidebar.info("Using the built-in sample dataset.")
        return load_default_data(), "Sample dataset"

    try:
        uploaded_data = load_uploaded_data(uploaded_file.getvalue())
    except Exception as exc:
        st.sidebar.error(f"Could not read uploaded CSV: {exc}")
        st.sidebar.warning("Falling back to the sample dataset.")
        return load_default_data(), "Sample dataset"

    missing_required, missing_optional = validate_columns(uploaded_data)
    if missing_required:
        st.sidebar.error(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing_required)
        )
        st.sidebar.warning("Falling back to the sample dataset.")
        return load_default_data(), "Sample dataset"

    st.session_state.uploaded_dataset_name = uploaded_file.name
    st.sidebar.success(f"Using uploaded dataset: {uploaded_file.name}")
    if missing_optional:
        st.sidebar.info(
            "Optional columns missing; reasonable defaults will be created for: "
            + ", ".join(missing_optional)
        )
    return uploaded_data, uploaded_file.name


def render_weight_controls() -> None:
    with st.sidebar.expander("Scoring Weights", expanded=False):
        st.caption("Default formula is preserved unless you change these sliders.")
        if st.button("Reset to default weights", use_container_width=True):
            reset_weights()

        for component in SCORE_COMPONENTS:
            st.slider(
                COMPONENT_LABELS[component],
                min_value=0,
                max_value=100,
                value=int(st.session_state[weight_key(component)]),
                step=1,
                key=weight_key(component),
            )

        total = sum(get_weight_percentages().values())
        st.write(f"Current total: **{total}%**")
        if total == 100:
            st.success("Weights sum to 100%.")
        elif total > 0:
            st.warning("Weights will be normalized automatically for scoring.")
        else:
            st.error("All weights are zero; default weights will be used.")


def safe_options(dataframe: pd.DataFrame, column: str) -> List[str]:
    if column not in dataframe.columns:
        return []
    return sorted(dataframe[column].dropna().astype(str).unique().tolist())


def safe_multiselect_default(key: str, options: List[str]) -> List[str]:
    current = st.session_state.get(key, options)
    valid = [value for value in current if value in options]
    return valid if valid else options


def render_filter_controls(dataframe: pd.DataFrame) -> Dict[str, object]:
    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    min_age = int(dataframe["age"].min()) if not dataframe.empty else 0
    max_age = int(dataframe["age"].max()) if not dataframe.empty else 30
    age_range = st.sidebar.slider(
        "Age range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        key="filter_age_range",
    )

    positions = safe_options(dataframe, "position")
    leagues = safe_options(dataframe, "league")
    clubs = safe_options(dataframe, "club")
    nationalities = safe_options(dataframe, "nationality")
    preferred_feet = safe_options(dataframe, "preferred_foot")

    selected_positions = st.sidebar.multiselect(
        "Position",
        options=positions,
        default=safe_multiselect_default("filter_positions", positions),
        key="filter_positions",
    )
    selected_leagues = st.sidebar.multiselect(
        "League",
        options=leagues,
        default=safe_multiselect_default("filter_leagues", leagues),
        key="filter_leagues",
    )
    selected_clubs = st.sidebar.multiselect(
        "Club",
        options=clubs,
        default=safe_multiselect_default("filter_clubs", clubs),
        key="filter_clubs",
    )
    selected_nationalities = st.sidebar.multiselect(
        "Nationality",
        options=nationalities,
        default=safe_multiselect_default("filter_nationalities", nationalities),
        key="filter_nationalities",
    )
    selected_feet = st.sidebar.multiselect(
        "Preferred foot",
        options=preferred_feet,
        default=safe_multiselect_default("filter_preferred_feet", preferred_feet),
        key="filter_preferred_feet",
    )

    max_minutes = int(dataframe["minutes_played"].max()) if not dataframe.empty else 0
    max_appearances = int(dataframe["appearances"].max()) if not dataframe.empty else 0
    min_minutes = st.sidebar.slider(
        "Minimum minutes played",
        min_value=0,
        max_value=max_minutes,
        value=0,
        key="filter_min_minutes",
    )
    min_appearances = st.sidebar.slider(
        "Minimum appearances",
        min_value=0,
        max_value=max_appearances,
        value=0,
        key="filter_min_appearances",
    )

    min_talent = 0
    if "talent_score" in dataframe.columns:
        min_talent = st.sidebar.slider(
            "Minimum Talent Score",
            min_value=0,
            max_value=100,
            value=0,
            key="filter_min_talent",
        )

    search_text = st.sidebar.text_input(
        "Search player name",
        value="",
        key="filter_search_text",
    )

    st.sidebar.divider()
    st.sidebar.caption("Built for a Data Science course project")

    return {
        "age_range": age_range,
        "positions": selected_positions,
        "leagues": selected_leagues,
        "clubs": selected_clubs,
        "nationalities": selected_nationalities,
        "preferred_feet": selected_feet,
        "min_minutes": min_minutes,
        "min_appearances": min_appearances,
        "min_talent": min_talent,
        "search_text": search_text,
    }


def apply_filters(dataframe: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    filtered = dataframe.copy()
    if filtered.empty:
        return filtered

    age_min, age_max = filters["age_range"]
    filtered = filtered[filtered["age"].between(age_min, age_max)]

    for filter_key, column in [
        ("positions", "position"),
        ("leagues", "league"),
        ("clubs", "club"),
        ("nationalities", "nationality"),
        ("preferred_feet", "preferred_foot"),
    ]:
        selected_values = filters.get(filter_key, [])
        if selected_values:
            filtered = filtered[filtered[column].astype(str).isin(selected_values)]

    filtered = filtered[filtered["minutes_played"] >= filters["min_minutes"]]
    filtered = filtered[filtered["appearances"] >= filters["min_appearances"]]

    if filters.get("min_talent", 0) > 0 and "talent_score" in filtered.columns:
        filtered = filtered[filtered["talent_score"] >= filters["min_talent"]]

    search_text = str(filters.get("search_text", "")).strip()
    if search_text:
        filtered = filtered[
            filtered["player_name"].str.contains(search_text, case=False, na=False)
        ]

    return filtered.reset_index(drop=True)


def make_kpi_cards(clean_players: pd.DataFrame, ranked_players: pd.DataFrame) -> None:
    total_players = len(clean_players)
    total_u18 = len(ranked_players)
    total_clubs = clean_players["club"].nunique()
    total_leagues = clean_players["league"].nunique()
    average_score = ranked_players["talent_score"].mean() if total_u18 else np.nan
    highest_score = ranked_players["talent_score"].max() if total_u18 else np.nan

    columns = st.columns(6)
    columns[0].metric("Total Players", f"{total_players:,}")
    columns[1].metric("U18 Players", f"{total_u18:,}")
    columns[2].metric("Clubs", f"{total_clubs:,}")
    columns[3].metric("Leagues", f"{total_leagues:,}")
    columns[4].metric("Avg Talent Score", format_number(average_score, 1))
    columns[5].metric("Highest Score", format_number(highest_score, 1))


def render_overview(clean_players: pd.DataFrame, ranked_players: pd.DataFrame) -> None:
    st.header("ScoutAI Overview")
    st.markdown(
        """
        ScoutAI is a data-driven scouting assistant that helps football clubs create
        a smarter shortlist of promising U18 players using public football data. It
        does not automatically decide who will become a star. Instead, it organizes
        player information, calculates interpretable scouting scores, and helps
        human scouts focus on the most relevant candidates.
        """
    )

    make_kpi_cards(clean_players, ranked_players)
    st.divider()

    st.subheader("Problem Statement")
    st.write(
        "Youth scouting teams often review many players with incomplete public data. "
        "ScoutAI turns basic football information into a transparent ranking and "
        "similar-player workflow that supports, rather than replaces, expert review."
    )

    st.subheader("Data Science Workflow")
    workflow_columns = st.columns(7)
    workflow_steps = [
        "Public Football Data",
        "Data Cleaning",
        "U18 Filtering",
        "Feature Engineering",
        "Talent Score",
        "Similar Player Recommendation",
        "Scout Shortlist",
    ]
    for index, step in enumerate(workflow_steps):
        workflow_columns[index].markdown(f"**{step}**")
        if index < len(workflow_steps) - 1:
            workflow_columns[index].caption("then")

    st.info(
        "Key idea: public data can help scouts prioritize review, but human scouts "
        "still make the final decision after video, context, coaching, and ethical checks."
    )


def render_data_explorer(
    raw_players: pd.DataFrame,
    dashboard_players: pd.DataFrame,
    filtered_players: pd.DataFrame,
) -> None:
    st.header("Data Explorer")
    st.write(
        "This page helps scouts narrow the search space before reviewing individual players."
    )

    columns = st.columns(4)
    columns[0].metric("Raw Rows", f"{len(raw_players):,}")
    columns[1].metric("Filtered Rows", f"{len(filtered_players):,}")
    columns[2].metric("Columns", f"{dashboard_players.shape[1]:,}")
    columns[3].metric("Missing Values", f"{int(dashboard_players.isna().sum().sum()):,}")

    st.subheader("Full Dataset Preview")
    st.dataframe(
        dashboard_players.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Filtered Dataset Preview")
    st.dataframe(
        filtered_players,
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download filtered dataset as CSV",
        data=dataframe_to_csv_bytes(filtered_players),
        file_name="filtered_scoutai_players.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Data shape and columns", expanded=False):
        st.write(f"Dataset shape: **{dashboard_players.shape[0]} rows x {dashboard_players.shape[1]} columns**")
        st.write(", ".join(dashboard_players.columns.tolist()))

    with st.expander("Missing value summary", expanded=False):
        missing_summary = (
            dashboard_players.isna().sum().reset_index()
        )
        missing_summary.columns = ["column", "missing_values"]
        st.dataframe(missing_summary, use_container_width=True, hide_index=True)

    with st.expander("Basic descriptive statistics", expanded=False):
        numeric_summary = dashboard_players.describe(include=[np.number]).T
        st.dataframe(numeric_summary, use_container_width=True)


def make_top_players_chart(ranking_data: pd.DataFrame, limit: int = 10) -> go.Figure:
    chart_data = ranking_data.nlargest(limit, "talent_score").sort_values("talent_score")
    return px.bar(
        chart_data,
        x="talent_score",
        y="player_name",
        orientation="h",
        color="position",
        title=f"Top {min(limit, len(chart_data))} U18 Players by Talent Score",
        labels={"talent_score": "Talent Score", "player_name": "Player"},
        hover_data=["club", "league", "age"],
    )


def make_component_breakdown_chart(ranking_data: pd.DataFrame, limit: int = 10) -> go.Figure:
    chart_data = ranking_data.nlargest(limit, "talent_score").copy()
    weights = get_normalized_weights()
    fig = go.Figure()
    for component in SCORE_COMPONENTS:
        fig.add_bar(
            x=chart_data["player_name"],
            y=chart_data[component] * weights[component],
            name=COMPONENT_LABELS[component],
        )
    fig.update_layout(
        barmode="stack",
        title="Weighted Score Component Comparison",
        xaxis_title="Player",
        yaxis_title="Weighted contribution to Talent Score",
        legend_title="Component",
    )
    return fig


def build_ranking_view(ranked_players: pd.DataFrame) -> pd.DataFrame:
    ranking_view = ranked_players.copy().reset_index(drop=True)
    ranking_view.insert(0, "rank", range(1, len(ranking_view) + 1))
    display_columns = ["rank"] + [
        column for column in RANKED_OUTPUT_COLUMNS if column in ranking_view.columns
    ]
    return ranking_view[display_columns]


def render_talent_ranking(
    ranked_players: pd.DataFrame,
    sidebar_filters: Dict[str, object],
) -> None:
    st.header("Talent Ranking")
    st.write(
        "The Talent Score is not a final recruitment decision. It is an interpretable "
        "ranking tool that helps scouts prioritize players for deeper review."
    )

    if ranked_players.empty:
        st.warning("No U18 players are available for ranking after cleaning/filtering.")
        return

    ranking_data = apply_filters(ranked_players, sidebar_filters)

    control_columns = st.columns(4)
    display_count = control_columns[0].slider(
        "Number of players to display",
        min_value=5,
        max_value=max(5, min(30, len(ranking_data))),
        value=min(10, max(5, len(ranking_data))),
    )
    position_options = safe_options(ranking_data, "position")
    selected_positions = control_columns[1].multiselect(
        "Ranking position filter",
        options=position_options,
        default=position_options,
    )
    min_minutes = control_columns[2].number_input(
        "Minimum minutes",
        min_value=0,
        value=0,
        step=100,
    )
    min_availability = control_columns[3].slider(
        "Minimum availability score",
        min_value=0,
        max_value=100,
        value=0,
    )

    sort_by = st.selectbox(
        "Sort ranking by",
        options=[
            "talent_score",
            "performance_score",
            "development_score",
            "availability_score",
        ],
        index=0,
    )

    if selected_positions:
        ranking_data = ranking_data[ranking_data["position"].isin(selected_positions)]
    ranking_data = ranking_data[ranking_data["minutes_played"] >= min_minutes]
    ranking_data = ranking_data[ranking_data["availability_score"] >= min_availability]
    ranking_data = ranking_data.sort_values(sort_by, ascending=False).reset_index(drop=True)

    if ranking_data.empty:
        st.warning("No players match the current ranking controls.")
        return

    visible_ranking = build_ranking_view(ranking_data).head(display_count)
    st.dataframe(
        visible_ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "talent_score": st.column_config.ProgressColumn(
                "talent_score",
                min_value=0,
                max_value=100,
                format="%.2f",
            ),
            "performance_score": st.column_config.NumberColumn(format="%.2f"),
            "development_score": st.column_config.NumberColumn(format="%.2f"),
            "playing_time_score": st.column_config.NumberColumn(format="%.2f"),
            "league_context_score": st.column_config.NumberColumn(format="%.2f"),
            "age_advantage_score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.download_button(
        "Download ranked U18 players as CSV",
        data=dataframe_to_csv_bytes(build_ranking_view(ranking_data)),
        file_name="ranked_u18_players.csv",
        mime="text/csv",
        use_container_width=True,
    )

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.plotly_chart(
            make_top_players_chart(ranking_data, limit=min(10, display_count)),
            use_container_width=True,
        )
    with chart_columns[1]:
        st.plotly_chart(
            make_component_breakdown_chart(ranking_data, limit=min(8, display_count)),
            use_container_width=True,
        )


def render_player_profile(ranked_players: pd.DataFrame) -> None:
    st.header("Player Profile")
    if ranked_players.empty:
        st.warning("No U18 players are available for profile review.")
        return

    player_names = ranked_players["player_name"].tolist()
    if st.session_state.selected_player not in player_names:
        st.session_state.selected_player = player_names[0]

    selected_player = st.selectbox(
        "Select player by name",
        options=player_names,
        index=player_names.index(st.session_state.selected_player),
        key="selected_player",
    )
    player = ranked_players[ranked_players["player_name"] == selected_player].iloc[0]

    st.subheader(player["player_name"])
    identity_columns = st.columns(6)
    identity_columns[0].metric("Age", int(player["age"]))
    identity_columns[1].metric("Nationality", player["nationality"])
    identity_columns[2].metric("Position", player["position"])
    identity_columns[3].metric("Club", player["club"])
    identity_columns[4].metric("League", player["league"])
    identity_columns[5].metric("Talent Score", format_number(player["talent_score"], 2))

    st.divider()
    profile_columns = st.columns(4)
    profile_columns[0].metric("Height", f"{player['height_cm']:.0f} cm")
    profile_columns[1].metric("Weight", f"{player['weight_kg']:.0f} kg")
    profile_columns[2].metric("Preferred Foot", player["preferred_foot"])
    profile_columns[3].metric("Market Value", format_money(player["market_value_eur"]))

    match_columns = st.columns(5)
    match_columns[0].metric("Appearances", int(player["appearances"]))
    match_columns[1].metric("Minutes", f"{int(player['minutes_played']):,}")
    match_columns[2].metric("Goals", int(player["goals"]))
    match_columns[3].metric("Assists", int(player["assists"]))
    match_columns[4].metric("Availability", format_number(player["availability_score"], 0))

    metric_columns = st.columns(6)
    metric_columns[0].metric("Goals / 90", format_number(player["goals_per_90"]))
    metric_columns[1].metric("Assists / 90", format_number(player["assists_per_90"]))
    metric_columns[2].metric(
        "Contribution / 90",
        format_number(player["contribution_per_90"]),
    )
    metric_columns[3].metric("Performance", format_number(player["performance_score"]))
    metric_columns[4].metric("Development", format_number(player["development_score"]))
    metric_columns[5].metric("Playing Time", format_number(player["playing_time_score"]))

    score_data = pd.DataFrame(
        {
            "component": [COMPONENT_LABELS[column] for column in SCORE_COMPONENTS],
            "score": [player[column] for column in SCORE_COMPONENTS],
        }
    )
    fig = px.bar(
        score_data,
        x="component",
        y="score",
        title=f"Score Breakdown for {player['player_name']}",
        labels={"component": "Score Component", "score": "Score"},
        text_auto=".1f",
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    strongest_component = max(SCORE_COMPONENTS, key=lambda column: player[column])
    weakest_component = min(SCORE_COMPONENTS, key=lambda column: player[column])
    interpretation = (
        f"{player['player_name']} has a Talent Score of "
        f"{player['talent_score']:.2f}. The profile is strongest in "
        f"{COMPONENT_LABELS[strongest_component].lower()}, while "
        f"{COMPONENT_LABELS[weakest_component].lower()} is the area that should be "
        "reviewed more carefully."
    )
    if player["availability_score"] < 80:
        interpretation += (
            " Availability is below the strongest range, so scouts should review "
            "public playing-time context and absence history carefully."
        )
    elif player["performance_score"] >= 75 and player["playing_time_score"] >= 50:
        interpretation += (
            " The player combines strong output with meaningful playing time, which "
            "makes the profile useful for deeper video and tactical review."
        )
    else:
        interpretation += (
            " Scouts should still review match video, tactical fit, development "
            "context, and non-quantitative factors before making any decision."
        )
    st.info(interpretation)

    with st.expander("Full selected player row", expanded=False):
        st.dataframe(
            pd.DataFrame([player[PROFILE_COLUMNS]]),
            use_container_width=True,
            hide_index=True,
        )


def render_similar_player_finder(ranked_players: pd.DataFrame) -> None:
    st.header("Similar Player Finder")
    st.write(
        "This feature helps answer the scouting question: if a club likes one player, "
        "which other U18 players have similar statistical profiles?"
    )

    if ranked_players.empty:
        st.warning("No U18 players are available for similar-player search.")
        return

    player_names = ranked_players["player_name"].tolist()
    default_player = st.session_state.selected_player
    if default_player not in player_names:
        default_player = player_names[0]

    control_columns = st.columns(3)
    target_player = control_columns[0].selectbox(
        "Target player",
        options=player_names,
        index=player_names.index(default_player),
    )
    top_n = control_columns[1].selectbox(
        "Number of similar players",
        options=[3, 5, 10],
        index=1,
    )
    same_position_only = control_columns[2].checkbox(
        "Filter to same position",
        value=False,
    )

    target_row = ranked_players[ranked_players["player_name"] == target_player].iloc[0]
    candidate_players = ranked_players.copy()
    if same_position_only:
        candidate_players = candidate_players[
            candidate_players["position"] == target_row["position"]
        ].copy()

    if len(candidate_players) <= 1:
        st.warning("Not enough players are available for this similarity search.")
        return

    recommendations = recommend_similar_players(
        candidate_players,
        target_player=target_player,
        top_n=top_n,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(OUTPUT_DIR / "similar_players.csv", index=False)

    target_columns = st.columns(5)
    target_columns[0].metric("Target", target_row["player_name"])
    target_columns[1].metric("Position", target_row["position"])
    target_columns[2].metric("Club", target_row["club"])
    target_columns[3].metric("Age", int(target_row["age"]))
    target_columns[4].metric("Talent Score", format_number(target_row["talent_score"]))

    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.download_button(
        "Download similar player recommendations",
        data=dataframe_to_csv_bytes(recommendations),
        file_name="similar_players.csv",
        mime="text/csv",
        use_container_width=True,
    )

    comparison_data = recommendations.rename(columns={"similar_player": "player"})
    fig = px.bar(
        comparison_data,
        x="player",
        y="similarity_score",
        color="position",
        title=f"Most Similar Players to {target_player}",
        labels={"similarity_score": "Similarity Score", "player": "Similar Player"},
        hover_data=["club", "age", "talent_score"],
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def render_score_model_explanation(ranked_players: pd.DataFrame) -> None:
    st.header("Score Model Explanation")
    st.markdown(
        """
        The weighted scoring model is intentionally simple and interpretable. It is
        suitable for early-stage scouting because scouts can understand why a player
        is ranked highly.
        """
    )

    st.code(
        """
Talent Score =
35% Performance Score
+ 25% Development / Market Value Score
+ 15% Playing Time Stability Score
+ 15% League / Club Context Score
+ 10% Age Advantage Score
        """.strip(),
        language="text",
    )

    component_table = pd.DataFrame(
        [
            {
                "Component": "Performance Score",
                "Default Weight": "35%",
                "Meaning": "Goals, assists, and contribution per 90",
            },
            {
                "Component": "Development Score",
                "Default Weight": "25%",
                "Meaning": "Market value and growth indicators",
            },
            {
                "Component": "Playing Time Score",
                "Default Weight": "15%",
                "Meaning": "Minutes and appearances",
            },
            {
                "Component": "League Context Score",
                "Default Weight": "15%",
                "Meaning": "League or club strength indicator",
            },
            {
                "Component": "Age Advantage Score",
                "Default Weight": "10%",
                "Meaning": "Younger players with strong performance receive credit",
            },
        ]
    )
    st.dataframe(component_table, use_container_width=True, hide_index=True)

    st.subheader("Interactive Weight Controls")
    st.write(
        "Adjust the sliders below or in the sidebar. The app normalizes non-zero "
        "weights automatically and recalculates Talent Scores on rerun."
    )

    slider_columns = st.columns(5)
    for index, component in enumerate(SCORE_COMPONENTS):
        st.session_state[model_weight_key(component)] = st.session_state[
            weight_key(component)
        ]
        with slider_columns[index]:
            st.slider(
                COMPONENT_LABELS[component],
                min_value=0,
                max_value=100,
                value=int(st.session_state[weight_key(component)]),
                step=1,
                key=model_weight_key(component),
                on_change=sync_model_weight,
                args=(component,),
            )

    total = sum(get_weight_percentages().values())
    if total == 100:
        st.success("Current weights sum to 100%.")
    elif total > 0:
        st.warning(
            f"Current weights sum to {total}%. The app normalizes them for scoring."
        )
    else:
        st.error("All custom weights are zero. Default weights are used instead.")

    current_weights = pd.DataFrame(
        [
            {
                "Component": COMPONENT_LABELS[component],
                "Input Weight (%)": get_weight_percentages()[component],
                "Normalized Weight (%)": round(get_normalized_weights()[component] * 100, 2),
            }
            for component in SCORE_COMPONENTS
        ]
    )
    st.dataframe(current_weights, use_container_width=True, hide_index=True)

    if not ranked_players.empty:
        st.subheader("Current Top Players Under Active Weights")
        st.dataframe(
            build_ranking_view(ranked_players).head(10),
            use_container_width=True,
            hide_index=True,
        )


def render_visual_analytics(ranked_players: pd.DataFrame) -> None:
    st.header("Visual Analytics")
    if ranked_players.empty:
        st.warning("No U18 players are available for visual analytics.")
        return

    chart = make_top_players_chart(ranked_players, limit=10)
    st.plotly_chart(chart, use_container_width=True)
    st.caption("This chart shows which U18 players are currently highest on the shortlist.")

    chart_columns = st.columns(2)

    with chart_columns[0]:
        position_counts = ranked_players["position"].value_counts().reset_index()
        position_counts.columns = ["position", "players"]
        fig = px.bar(
            position_counts,
            x="position",
            y="players",
            title="Position Distribution",
            labels={"position": "Position", "players": "Number of Players"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart helps scouts see whether the dataset is balanced by role.")

    with chart_columns[1]:
        fig = px.histogram(
            ranked_players,
            x="talent_score",
            nbins=12,
            title="Talent Score Distribution",
            labels={"talent_score": "Talent Score"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart shows whether scores are clustered or spread out.")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        fig = px.scatter(
            ranked_players,
            x="market_value_eur",
            y="talent_score",
            color="position",
            hover_name="player_name",
            title="Market Value vs Talent Score",
            labels={
                "market_value_eur": "Market Value (EUR)",
                "talent_score": "Talent Score",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart highlights players whose model score differs from market value.")

    with chart_columns[1]:
        fig = px.scatter(
            ranked_players,
            x="age",
            y="talent_score",
            color="position",
            hover_name="player_name",
            title="Age vs Talent Score",
            labels={"age": "Age", "talent_score": "Talent Score"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart shows how younger players compare within the U18 group.")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        fig = px.scatter(
            ranked_players,
            x="minutes_played",
            y="performance_score",
            color="position",
            hover_name="player_name",
            title="Minutes Played vs Performance Score",
            labels={
                "minutes_played": "Minutes Played",
                "performance_score": "Performance Score",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart helps separate small-sample output from stable playing time.")

    with chart_columns[1]:
        position_scores = (
            ranked_players.groupby("position", as_index=False)["talent_score"]
            .mean()
            .sort_values("talent_score", ascending=False)
        )
        fig = px.bar(
            position_scores,
            x="position",
            y="talent_score",
            title="Average Talent Score by Position",
            labels={"position": "Position", "talent_score": "Average Talent Score"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart helps scouts identify position groups that score highly.")

    league_scores = (
        ranked_players.groupby("league", as_index=False)["talent_score"]
        .mean()
        .sort_values("talent_score", ascending=False)
    )
    fig = px.bar(
        league_scores,
        x="league",
        y="talent_score",
        title="Average Talent Score by League",
        labels={"league": "League", "talent_score": "Average Talent Score"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This chart helps scouts identify whether high-scoring players are concentrated "
        "in specific positions or leagues."
    )


def render_ethics_page() -> None:
    st.header("Ethics and Limitations")
    st.warning(
        "This app is a decision-support prototype. It should not be used to make "
        "final recruitment decisions about minors."
    )

    st.subheader("Ethical Use")
    st.markdown(
        """
        1. U18 players are minors.
        2. The system should only use legally accessible public data.
        3. The system should not collect private medical records.
        4. Injury-related information should be treated only as public availability
           indicators, not medical diagnosis.
        5. The system should support human scouts, not replace them.
        6. Public football data may be biased toward famous clubs and visible leagues.
        7. Players from smaller academies may be underrepresented.
        8. The Talent Score should not be used as a final recruitment decision.
        9. Clubs should combine data analysis with video review, coach reports,
           medical review, and ethical scouting practices.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The repository uses a small sample dataset by default.
        - Youth football data is often incomplete and uneven across regions.
        - There is no perfect label for future football success.
        - The Talent Score formula is simplified.
        - Different positions require different evaluation standards.
        - Public data may favor well-known academies and visible leagues.
        - Real clubs would need official data partnerships and human scouting review.
        """
    )


def render_about_page(dataset_name: str) -> None:
    st.header("About This Project")
    st.subheader("Project Context")
    st.write(
        "ScoutAI is a university data science course project designed for a final "
        "English presentation. The app frames a problem-solution workflow for youth "
        "football scouting and demonstrates how public football data can support a "
        "transparent scouting dashboard."
    )

    st.subheader("Current Dataset")
    st.write(f"Active dataset: **{dataset_name}**")

    st.subheader("Technologies Used")
    st.markdown(
        """
        - Python
        - pandas
        - numpy
        - scikit-learn
        - Streamlit
        - matplotlib
        - plotly
        - GitHub
        """
    )

    st.subheader("Repository Structure")
    st.code(
        """
Football/
|-- README.md
|-- requirements.txt
|-- main.py
|-- streamlit_app.py
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
|-- docs/
|   `-- project_summary.md
`-- .streamlit/
    `-- config.toml
        """.strip(),
        language="text",
    )

    st.subheader("How to Run Locally")
    st.code(
        """
pip install -r requirements.txt
python main.py
streamlit run streamlit_app.py
        """.strip(),
        language="bash",
    )

    st.subheader("How to Deploy on Streamlit Community Cloud")
    st.markdown(
        """
        1. Push the repository to GitHub.
        2. Make sure `streamlit_app.py` exists in the root directory.
        3. Make sure `requirements.txt` includes all dependencies.
        4. Go to Streamlit Community Cloud.
        5. Select the GitHub repository.
        6. Set the main file path to `streamlit_app.py`.
        7. Deploy the app.
        """
    )

    st.subheader("References")
    st.markdown(
        """
        - Transfermarkt datasets: https://github.com/dcaribou/transfermarkt-datasets
        - Football Manager 2023 dataset: https://www.kaggle.com/datasets/siddhrajthakor/football-manager-2023-dataset
        - pandas documentation: https://pandas.pydata.org/docs/
        - scikit-learn documentation: https://scikit-learn.org/stable/
        - Streamlit documentation: https://docs.streamlit.io/
        - Plotly documentation: https://plotly.com/python/
        """
    )


def render_header(dataset_name: str) -> None:
    st.title("ScoutAI: Youth Football Scouting Dashboard")
    st.markdown(
        f"""
        <p class="small-muted">
        Data-driven U18 football scouting dashboard for ranking players,
        explaining score components, and finding similar player profiles.
        Active dataset: <strong>{dataset_name}</strong>.
        </p>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_session_state()
    render_custom_css()

    raw_players, dataset_name = render_dataset_controls()
    render_weight_controls()

    try:
        clean_players, dashboard_players, ranked_players = run_pipeline(
            raw_players,
            get_weight_signature(),
        )
    except Exception as exc:
        st.error(f"ScoutAI could not prepare the dataset: {exc}")
        st.stop()

    try:
        ensure_output_files(ranked_players)
    except Exception as exc:
        st.warning(f"Outputs could not be refreshed automatically: {exc}")

    if ranked_players.empty:
        st.warning("No U18 players were found after cleaning the active dataset.")

    filters = render_filter_controls(dashboard_players)
    filtered_players = apply_filters(dashboard_players, filters)

    render_header(dataset_name)

    tabs = st.tabs(
        [
            "Overview",
            "Data Explorer",
            "Talent Ranking",
            "Player Profile",
            "Similar Player Finder",
            "Score Model Explanation",
            "Visual Analytics",
            "Ethics and Limitations",
            "About This Project",
        ]
    )

    with tabs[0]:
        render_overview(clean_players, ranked_players)
    with tabs[1]:
        render_data_explorer(raw_players, dashboard_players, filtered_players)
    with tabs[2]:
        render_talent_ranking(ranked_players, filters)
    with tabs[3]:
        render_player_profile(ranked_players)
    with tabs[4]:
        render_similar_player_finder(ranked_players)
    with tabs[5]:
        render_score_model_explanation(ranked_players)
    with tabs[6]:
        render_visual_analytics(ranked_players)
    with tabs[7]:
        render_ethics_page()
    with tabs[8]:
        render_about_page(dataset_name)


if __name__ == "__main__":
    main()
