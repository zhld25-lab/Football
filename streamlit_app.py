from html import escape
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

THEMES = {
    "Barca Inspired": {
        "name": "Barca Inspired",
        "primary": "#A50044",
        "secondary": "#004D98",
        "gold": "#EDBB00",
        "surface": "rgba(8, 14, 34, 0.82)",
        "surface_light": "rgba(255, 255, 255, 0.10)",
        "text": "#F8FAFC",
        "muted": "#CBD5E1",
        "background": "https://images.unsplash.com/photo-1522778119026-d647f0596c20?auto=format&fit=crop&w=2200&q=80",
    },
    "Classic Light": {
        "name": "Classic Light",
        "primary": "#1F77B4",
        "secondary": "#2E7D6B",
        "gold": "#F59E0B",
        "surface": "rgba(255, 255, 255, 0.92)",
        "surface_light": "rgba(15, 23, 42, 0.06)",
        "text": "#0F172A",
        "muted": "#475569",
        "background": "https://images.unsplash.com/photo-1431324155629-1a6deb1dec8d?auto=format&fit=crop&w=2200&q=80",
    },
}

FOOTBALL_IMAGES = [
    "https://images.unsplash.com/photo-1517927033932-b3d18e61fb3a?auto=format&fit=crop&w=900&q=80",
    "https://images.unsplash.com/photo-1526232761682-d26e03ac148e?auto=format&fit=crop&w=900&q=80",
    "https://images.unsplash.com/photo-1574629810360-7efbbe195018?auto=format&fit=crop&w=900&q=80",
    "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?auto=format&fit=crop&w=900&q=80",
    "https://images.unsplash.com/photo-1524015368236-bbf6f72545b6?auto=format&fit=crop&w=900&q=80",
    "https://images.unsplash.com/photo-1529900748604-07564a03e7a6?auto=format&fit=crop&w=900&q=80",
]

POSITION_IMAGES = {
    "GK": "https://images.unsplash.com/photo-1551958219-acbc608c6377?auto=format&fit=crop&w=900&q=80",
    "CB": "https://images.unsplash.com/photo-1626248801379-51a0748a5f96?auto=format&fit=crop&w=900&q=80",
    "FB": "https://images.unsplash.com/photo-1553778263-73a83bab9b0c?auto=format&fit=crop&w=900&q=80",
    "DM": "https://images.unsplash.com/photo-1531415074968-036ba1b575da?auto=format&fit=crop&w=900&q=80",
    "CM": "https://images.unsplash.com/photo-1579952363873-27f3bade9f55?auto=format&fit=crop&w=900&q=80",
    "AM": "https://images.unsplash.com/photo-1517466787929-bc90951d0974?auto=format&fit=crop&w=900&q=80",
    "Winger": "https://images.unsplash.com/photo-1518604666860-9ed391f76460?auto=format&fit=crop&w=900&q=80",
    "ST": "https://images.unsplash.com/photo-1556056504-5c7696c4c28d?auto=format&fit=crop&w=900&q=80",
}

MEDIA_STORIES = [
    {
        "title": "Academy Watchlist",
        "tag": "U18 Radar",
        "copy": "用数据先筛出值得看录像的球员，再交给球探判断比赛气质和战术适配。",
        "image": "https://images.unsplash.com/photo-1522778034537-20a2486be803?auto=format&fit=crop&w=900&q=80",
    },
    {
        "title": "Blaugrana Theme",
        "tag": "Presentation Mode",
        "copy": "红蓝配色、球场背景和深色玻璃卡片，让最终展示更像真实俱乐部看板。",
        "image": "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?auto=format&fit=crop&w=900&q=80",
    },
    {
        "title": "Similar Player Search",
        "tag": "Recommendation",
        "copy": "如果俱乐部喜欢某一类球员，可以快速找到数据画像相近的其他候选人。",
        "image": "https://images.unsplash.com/photo-1577223625816-7546f13df25d?auto=format&fit=crop&w=900&q=80",
    },
]

SEARCH_EXAMPLE_NAMES = [
    "Lamine Yamal",
    "Philipp Lahm",
    "Lamar Hunt",
    "Lampros Choutos",
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
    st.session_state.setdefault("visual_theme", "Barca Inspired")
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
    theme = THEMES.get(st.session_state.get("visual_theme"), THEMES["Barca Inspired"])
    px.defaults.template = "plotly_dark" if theme["name"] == "Barca Inspired" else "plotly_white"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(120deg, rgba(0, 77, 152, 0.90), rgba(165, 0, 68, 0.86)),
                url('{theme["background"]}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: {theme["text"]};
        }}
        .stApp, .stMarkdown, .stText, p, span, label, h1, h2, h3, h4, h5, h6 {{
            color: {theme["text"]} !important;
        }}
        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background: linear-gradient(180deg, rgba(3, 7, 18, 0.96), rgba(15, 23, 42, 0.92));
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }}
        [data-testid="stSidebar"] * {{
            color: #F8FAFC !important;
        }}
        [data-testid="stMetric"] {{
            background: {theme["surface"]};
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.18);
        }}
        .small-muted {{
            color: #F1F5F9;
            font-size: 0.95rem;
            font-weight: 650;
        }}
        .hero-shell {{
            min-height: 360px;
            border-radius: 22px;
            padding: 2rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(90deg, rgba(3, 7, 18, 0.84), rgba(0, 77, 152, 0.42), rgba(165, 0, 68, 0.58)),
                url('https://images.unsplash.com/photo-1522778119026-d647f0596c20?auto=format&fit=crop&w=2000&q=80');
            background-size: cover;
            background-position: center;
            border: 1px solid rgba(255, 255, 255, 0.16);
            box-shadow: 0 28px 70px rgba(2, 6, 23, 0.38);
            display: flex;
            align-items: flex-end;
        }}
        .hero-kicker {{
            color: {theme["gold"]};
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .hero-title {{
            color: white;
            font-size: 3rem;
            line-height: 1.05;
            font-weight: 900;
            margin: 0.35rem 0;
        }}
        .hero-copy {{
            color: #E5E7EB;
            max-width: 740px;
            font-size: 1.04rem;
        }}
        .glass-card, .player-card, .media-card {{
            background: {theme["surface"]};
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 16px;
            box-shadow: 0 18px 42px rgba(2, 6, 23, 0.24);
            overflow: hidden;
        }}
        .glass-card {{
            padding: 1rem;
        }}
        .media-card img, .player-card img {{
            width: 100%;
            height: 170px;
            object-fit: cover;
            display: block;
        }}
        .media-body, .player-body {{
            padding: 0.95rem;
        }}
        .tag-pill {{
            display: inline-block;
            border: 1px solid rgba(237, 187, 0, 0.55);
            color: {theme["gold"]};
            border-radius: 999px;
            padding: 0.12rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}
        .player-name {{
            color: white;
            font-size: 1rem;
            font-weight: 850;
            margin-bottom: 0.25rem;
        }}
        .player-meta {{
            color: #E2E8F0;
            font-size: 0.85rem;
            font-weight: 650;
        }}
        .score-badge {{
            margin-top: 0.6rem;
            color: #020617;
            background: linear-gradient(90deg, {theme["gold"]}, #FDE68A);
            padding: 0.35rem 0.55rem;
            border-radius: 10px;
            display: inline-block;
            font-weight: 900;
        }}
        .section-kicker {{
            color: {theme["gold"]};
            font-size: 0.8rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .section-title {{
            color: white;
            font-size: 1.7rem;
            font-weight: 900;
            margin: 0.1rem 0 0.8rem;
        }}
        .workflow-chip {{
            background: {theme["surface"]};
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 14px;
            padding: 0.8rem;
            min-height: 88px;
        }}
        .theme-note {{
            color: #E2E8F0;
            font-size: 0.82rem;
        }}
        .fact-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
            margin: 1rem 0;
        }}
        .fact-tile {{
            background: rgba(2, 6, 23, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-left: 4px solid {theme["gold"]};
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            min-height: 92px;
            box-shadow: 0 14px 34px rgba(2, 6, 23, 0.24);
        }}
        .fact-label {{
            color: #CBD5E1;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}
        .fact-value {{
            color: #FFFFFF;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1.12;
            white-space: normal;
            overflow-wrap: anywhere;
        }}
        .fact-sub {{
            color: #E2E8F0;
            font-size: 0.82rem;
            font-weight: 650;
            margin-top: 0.25rem;
        }}
        .search-panel {{
            background: rgba(2, 6, 23, 0.76);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 18px;
            padding: 1rem;
            margin: 0.7rem 0 1rem;
            box-shadow: 0 16px 38px rgba(2, 6, 23, 0.26);
        }}
        .motion-window {{
            position: relative;
            height: 210px;
            border-radius: 16px;
            overflow: hidden;
            background:
                linear-gradient(90deg, rgba(10, 88, 47, 0.92), rgba(16, 122, 63, 0.84)),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.08) 0 2px, transparent 2px 80px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: inset 0 0 0 2px rgba(255,255,255,0.08), 0 16px 38px rgba(2,6,23,0.28);
        }}
        .pitch-line {{
            position: absolute;
            inset: 20px;
            border: 2px solid rgba(255,255,255,0.40);
            border-radius: 10px;
        }}
        .moving-ball {{
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 999px;
            background: #FFFFFF;
            box-shadow: 0 0 18px rgba(255,255,255,0.9);
            animation: ballmove 5.5s infinite ease-in-out;
        }}
        .moving-dot {{
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 999px;
            background: {theme["gold"]};
            box-shadow: 0 0 18px rgba(237,187,0,0.75);
            animation: pressmove 4.8s infinite ease-in-out;
        }}
        .moving-dot.second {{
            background: {theme["primary"]};
            animation-delay: 0.8s;
        }}
        .motion-label {{
            position: absolute;
            left: 1rem;
            bottom: 0.85rem;
            color: white;
            font-weight: 900;
            text-shadow: 0 2px 10px rgba(0,0,0,0.6);
        }}
        @keyframes ballmove {{
            0% {{ left: 16%; top: 68%; }}
            32% {{ left: 44%; top: 35%; }}
            65% {{ left: 74%; top: 48%; }}
            100% {{ left: 16%; top: 68%; }}
        }}
        @keyframes pressmove {{
            0% {{ left: 22%; top: 34%; }}
            50% {{ left: 60%; top: 28%; }}
            100% {{ left: 22%; top: 34%; }}
        }}
        .stDataFrame {{
            background: rgba(2, 6, 23, 0.78) !important;
            border-radius: 14px;
        }}
        input, textarea {{
            color: #FFFFFF !important;
        }}
        [data-baseweb="input"], [data-baseweb="select"] > div {{
            background: rgba(15, 23, 42, 0.92) !important;
            border-color: rgba(255, 255, 255, 0.24) !important;
        }}
        [data-baseweb="select"] span {{
            color: #FFFFFF !important;
        }}
        .stButton button {{
            background: linear-gradient(90deg, {theme["primary"]}, {theme["secondary"]});
            color: #FFFFFF !important;
            border: 1px solid rgba(255,255,255,0.20);
            font-weight: 850;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_active_theme() -> Dict[str, str]:
    return THEMES.get(st.session_state.get("visual_theme"), THEMES["Barca Inspired"])


def stable_index(value: object, length: int) -> int:
    if length <= 0:
        return 0
    text = str(value)
    return sum(ord(character) for character in text) % length


def player_image_url(player: pd.Series) -> str:
    position = str(player.get("position", ""))
    if position in POSITION_IMAGES:
        return POSITION_IMAGES[position]
    return FOOTBALL_IMAGES[stable_index(player.get("player_name", ""), len(FOOTBALL_IMAGES))]


def render_section_heading(kicker: str, title: str, copy: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-kicker">{escape(kicker)}</div>
        <div class="section-title">{escape(title)}</div>
        """,
        unsafe_allow_html=True,
    )
    if copy:
        st.markdown(f'<p class="small-muted">{escape(copy)}</p>', unsafe_allow_html=True)


def render_hero(ranked_players: pd.DataFrame) -> None:
    top_name = "No player ranked yet"
    top_score = "N/A"
    if not ranked_players.empty:
        top_name = str(ranked_players.iloc[0]["player_name"])
        top_score = f'{ranked_players.iloc[0]["talent_score"]:.2f}'

    st.markdown(
        f"""
        <div class="hero-shell">
            <div>
                <div class="hero-kicker">Barca-inspired scouting command center</div>
                <div class="hero-title">ScoutAI Youth Talent Lab</div>
                <div class="hero-copy">
                    一个红蓝主题的 U18 球探数据网站：综合球员数据、能力六边形、
                    相似球员搜索、图片栏目和可视化看板，帮助球探做更聪明的 shortlist。
                </div>
                <div class="score-badge">Current No.1: {escape(top_name)} | Talent Score {top_score}</div>
                <div class="theme-note">Theme note: Barca-inspired visual style only. No official club affiliation.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_media_wall() -> None:
    render_section_heading(
        "Visual Columns",
        "球探媒体栏目",
        "用图片和短文把数据项目讲成一个完整的足球故事，而不是只有表格。",
    )
    columns = st.columns(3)
    for column, story in zip(columns, MEDIA_STORIES):
        with column:
            st.markdown(
                f"""
                <div class="media-card">
                    <img src="{story['image']}" alt="{escape(story['title'])}">
                    <div class="media-body">
                        <div class="tag-pill">{escape(story['tag'])}</div>
                        <div class="player-name">{escape(story['title'])}</div>
                        <div class="player-meta">{escape(story['copy'])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_search_candidates(query: str, player_names: List[str]) -> List[str]:
    cleaned_query = query.strip().lower()
    if not cleaned_query:
        return player_names[:12]

    all_names = player_names + [
        name for name in SEARCH_EXAMPLE_NAMES if name not in player_names
    ]
    startswith_matches = [
        name for name in all_names if name.lower().startswith(cleaned_query)
    ]
    contains_matches = [
        name
        for name in all_names
        if cleaned_query in name.lower() and name not in startswith_matches
    ]
    return (startswith_matches + contains_matches)[:12]


def render_global_player_search(ranked_players: pd.DataFrame) -> None:
    if ranked_players.empty:
        return

    player_names = ranked_players["player_name"].tolist()
    with st.container():
        st.markdown('<div class="search-panel">', unsafe_allow_html=True)
        render_section_heading(
            "Quick Search",
            "球员查询窗口",
            "输入名字或首字母后，下方会自动给出候选项。例如输入 lam 可以看到 Lamine Yamal 或 Lahm 这类候选。",
        )
        search_columns = st.columns([1, 1.25, 0.75])
        query = search_columns[0].text_input(
            "输入球员姓名 / 首字母",
            value="",
            placeholder="例如：lam, luc, diego",
            key="global_player_query",
        )
        candidates = build_search_candidates(query, player_names)
        if not candidates:
            search_columns[1].warning("没有匹配候选，请换一个关键词。")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        selected_candidate = search_columns[1].selectbox(
            "候选球员",
            options=candidates,
            key="global_player_candidate",
        )
        if search_columns[2].button("打开球员档案", use_container_width=True):
            if selected_candidate in player_names:
                st.session_state.selected_player = selected_candidate
                st.success(f"已选择 {selected_candidate}，请进入 Player Profile 查看。")
            else:
                st.info(
                    f"{selected_candidate} 是搜索示例名，不在当前 sample dataset 中。"
                    "上传完整球员 CSV 后可以查询真实数据。"
                )
        st.markdown("</div>", unsafe_allow_html=True)


def render_motion_windows() -> None:
    render_section_heading(
        "Motion Windows",
        "视频 / 动图小窗口",
        "用动态战术板模拟比赛片段，让页面更像球探汇报界面。",
    )
    motion_columns = st.columns(2)
    with motion_columns[0]:
        st.markdown(
            """
            <div class="motion-window">
                <div class="pitch-line"></div>
                <div class="moving-ball"></div>
                <div class="moving-dot"></div>
                <div class="moving-dot second"></div>
                <div class="motion-label">Animated pressing pattern</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with motion_columns[1]:
        st.video(
            "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
        )
        st.caption(
            "这里是视频窗口示例。后续可以替换成公开视频 URL、训练片段或自己上传的视频素材。"
        )


def render_player_gallery(players: pd.DataFrame, title: str, limit: int = 6) -> None:
    if players.empty:
        return

    render_section_heading(
        "Player Gallery",
        title,
        "球员卡片墙用于展示 shortlist，不只是把名字放进表格。",
    )
    visible_players = players.head(limit).reset_index(drop=True)
    columns = st.columns(min(3, len(visible_players)))
    for index, (_, player) in enumerate(visible_players.iterrows()):
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="player-card">
                    <img src="{player_image_url(player)}" alt="{escape(player['player_name'])}">
                    <div class="player-body">
                        <div class="tag-pill">{escape(str(player['position']))} | Age {int(player['age'])}</div>
                        <div class="player-name">{escape(str(player['player_name']))}</div>
                        <div class="player-meta">{escape(str(player['club']))}</div>
                        <div class="player-meta">{escape(str(player['league']))}</div>
                        <div class="score-badge">Talent Score {player['talent_score']:.2f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def make_player_radar_chart(player: pd.Series) -> go.Figure:
    theme = get_active_theme()
    categories = [
        "Talent",
        "Performance",
        "Development",
        "Playing Time",
        "League Context",
        "Availability",
    ]
    values = [
        player["talent_score"],
        player["performance_score"],
        player["development_score"],
        player["playing_time_score"],
        player["league_context_score"],
        player["availability_score"],
    ]
    closed_categories = categories + [categories[0]]
    closed_values = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_categories,
            fill="toself",
            name=str(player["player_name"]),
            line=dict(color=theme["gold"], width=3),
            fillcolor="rgba(237, 187, 0, 0.28)",
        )
    )
    fig.update_layout(
        title=f"{player['player_name']} 能力六边形",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["text"]),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.22)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.22)"),
        ),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def make_radar_comparison_chart(players: pd.DataFrame, limit: int = 3) -> go.Figure:
    theme = get_active_theme()
    categories = [
        "Talent",
        "Performance",
        "Development",
        "Playing Time",
        "League Context",
        "Availability",
    ]
    fig = go.Figure()
    colors = [theme["gold"], theme["primary"], theme["secondary"]]
    for index, (_, player) in enumerate(players.head(limit).iterrows()):
        values = [
            player["talent_score"],
            player["performance_score"],
            player["development_score"],
            player["playing_time_score"],
            player["league_context_score"],
            player["availability_score"],
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=str(player["player_name"]),
                line=dict(color=colors[index % len(colors)], width=2),
                opacity=0.75,
            )
        )
    fig.update_layout(
        title="Top Players 能力六边形对比",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["text"]),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return fig


def render_recommendation_cards(recommendations: pd.DataFrame, ranked_players: pd.DataFrame) -> None:
    if recommendations.empty:
        return
    render_section_heading(
        "Similar Profiles",
        "相似球员图片推荐",
        "卡片保留数据表的精度，同时让展示更像球探部门内部 briefing。",
    )
    merged = recommendations.merge(
        ranked_players,
        left_on="similar_player",
        right_on="player_name",
        how="left",
        suffixes=("", "_source"),
    )
    columns = st.columns(min(3, len(merged)))
    for index, (_, row) in enumerate(merged.iterrows()):
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="player-card">
                    <img src="{player_image_url(row)}" alt="{escape(row['similar_player'])}">
                    <div class="player-body">
                        <div class="tag-pill">Similarity {row['similarity_score']:.2f}</div>
                        <div class="player-name">{escape(str(row['similar_player']))}</div>
                        <div class="player-meta">{escape(str(row['position']))} | {escape(str(row['club']))}</div>
                        <div class="score-badge">Talent Score {row['talent_score']:.2f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_player_fact_panel(player: pd.Series) -> None:
    facts = [
        ("Age", int(player["age"]), "development age"),
        ("Nationality", player["nationality"], "background"),
        ("Position", player["position"], "role"),
        ("Club", player["club"], "current team"),
        ("League", player["league"], "context"),
        ("Talent Score", f"{player['talent_score']:.2f}", "model score"),
        ("Height", f"{player['height_cm']:.0f} cm", "profile"),
        ("Weight", f"{player['weight_kg']:.0f} kg", "profile"),
        ("Foot", player["preferred_foot"], "preferred foot"),
        ("Market Value", format_money(player["market_value_eur"]), "public signal"),
        ("Minutes", f"{int(player['minutes_played']):,}", "playing time"),
        ("Availability", f"{player['availability_score']:.0f}", "public indicator"),
    ]
    html = ['<div class="fact-grid">']
    for label, value, subtext in facts:
        html.append(
            f"""
            <div class="fact-tile">
                <div class="fact-label">{escape(str(label))}</div>
                <div class="fact-value">{escape(str(value))}</div>
                <div class="fact-sub">{escape(str(subtext))}</div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def make_talent_gauge(player: pd.Series) -> go.Figure:
    theme = get_active_theme()
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(player["talent_score"]),
            number={"font": {"size": 44, "color": theme["text"]}},
            title={"text": "Talent Score", "font": {"color": theme["text"], "size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": theme["text"]},
                "bar": {"color": theme["gold"]},
                "bgcolor": "rgba(255,255,255,0.08)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.20)",
                "steps": [
                    {"range": [0, 45], "color": "rgba(148, 163, 184, 0.25)"},
                    {"range": [45, 70], "color": "rgba(0, 77, 152, 0.36)"},
                    {"range": [70, 100], "color": "rgba(165, 0, 68, 0.42)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=310,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text"]},
        margin=dict(l=25, r=25, t=45, b=20),
    )
    return fig


def make_player_metric_bars(player: pd.Series) -> go.Figure:
    theme = get_active_theme()
    chart_data = pd.DataFrame(
        {
            "metric": [
                "Performance",
                "Development",
                "Playing Time",
                "League Context",
                "Age Advantage",
                "Availability",
            ],
            "score": [
                player["performance_score"],
                player["development_score"],
                player["playing_time_score"],
                player["league_context_score"],
                player["age_advantage_score"],
                player["availability_score"],
            ],
        }
    ).sort_values("score")
    fig = px.bar(
        chart_data,
        x="score",
        y="metric",
        orientation="h",
        text="score",
        color="score",
        color_continuous_scale=[[0, theme["secondary"]], [0.6, theme["primary"]], [1, theme["gold"]]],
        range_x=[0, 100],
        title="能力维度条形图",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text"]},
        coloraxis_showscale=False,
        margin=dict(l=20, r=40, t=55, b=20),
    )
    return fig


def make_player_output_chart(player: pd.Series) -> go.Figure:
    theme = get_active_theme()
    chart_data = pd.DataFrame(
        {
            "metric": ["Goals", "Assists", "Goals / 90", "Assists / 90", "Contribution / 90"],
            "value": [
                player["goals"],
                player["assists"],
                player["goals_per_90"],
                player["assists_per_90"],
                player["contribution_per_90"],
            ],
        }
    )
    fig = px.bar(
        chart_data,
        x="metric",
        y="value",
        text="value",
        title="进攻贡献图",
        color="metric",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        height=330,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text"]},
        showlegend=False,
        margin=dict(l=30, r=20, t=55, b=20),
    )
    return fig


def render_target_summary_panel(player: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="fact-grid">
            <div class="fact-tile"><div class="fact-label">Target</div><div class="fact-value">{escape(str(player['player_name']))}</div><div class="fact-sub">selected player</div></div>
            <div class="fact-tile"><div class="fact-label">Position</div><div class="fact-value">{escape(str(player['position']))}</div><div class="fact-sub">role</div></div>
            <div class="fact-tile"><div class="fact-label">Club</div><div class="fact-value">{escape(str(player['club']))}</div><div class="fact-sub">current team</div></div>
            <div class="fact-tile"><div class="fact-label">Talent</div><div class="fact-value">{player['talent_score']:.2f}</div><div class="fact-sub">model score</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_controls() -> Tuple[pd.DataFrame, str]:
    st.sidebar.markdown("## ScoutAI")
    st.sidebar.caption("U18 football scouting dashboard")
    st.sidebar.radio(
        "视觉主题",
        options=list(THEMES.keys()),
        key="visual_theme",
        help="Barca Inspired 使用红蓝配色和球场背景，Classic Light 更适合简洁报告展示。",
    )
    st.sidebar.caption(
        "Barca-inspired theme is a visual style only; this project has no official club affiliation."
    )
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
    render_hero(ranked_players)
    render_section_heading(
        "Overview",
        "项目总览",
        "红蓝主题球探数据中心：图片、图表、六边形能力图和候选球员搜索集中在一个页面体验里。",
    )
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
    render_media_wall()
    st.divider()
    render_motion_windows()
    st.divider()

    st.subheader("Problem Statement / 问题陈述")
    st.write(
        "Youth scouting teams often review many players with incomplete public data. "
        "ScoutAI turns basic football information into a transparent ranking and "
        "similar-player workflow that supports, rather than replaces, expert review."
    )

    st.subheader("Data Science Workflow / 数据科学流程")
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
        workflow_columns[index].markdown(
            f'<div class="workflow-chip"><strong>{escape(step)}</strong><br><span class="small-muted">Step {index + 1}</span></div>',
            unsafe_allow_html=True,
        )

    st.info(
        "Key idea: public data can help scouts prioritize review, but human scouts "
        "still make the final decision after video, context, coaching, and ethical checks."
    )

    if not ranked_players.empty:
        st.divider()
        render_player_gallery(ranked_players, "首页重点球员卡片", limit=6)


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
            key="ranking_top_players_chart",
        )
    with chart_columns[1]:
        st.plotly_chart(
            make_component_breakdown_chart(ranking_data, limit=min(8, display_count)),
            use_container_width=True,
            key="ranking_component_breakdown_chart",
        )

    st.divider()
    radar_columns = st.columns([1.05, 1])
    with radar_columns[0]:
        st.plotly_chart(
            make_radar_comparison_chart(ranking_data, limit=min(3, len(ranking_data))),
            use_container_width=True,
            key="ranking_radar_comparison_chart",
        )
    with radar_columns[1]:
        render_section_heading(
            "Hexagon Comparison",
            "懂球帝风格能力六边形",
            "用六个维度同时比较球员画像：Talent、Performance、Development、Playing Time、League Context、Availability。",
        )
        st.write(
            "六边形图比单一分数更适合课堂展示，因为它能说明一个球员到底是靠进攻产出、发展潜力、出场稳定性，还是联赛环境拿到高分。"
        )

    st.divider()
    render_player_gallery(ranking_data, "Top Shortlist 图片球员墙", limit=min(6, display_count))


def render_player_profile(ranked_players: pd.DataFrame) -> None:
    st.header("Player Profile")
    if ranked_players.empty:
        st.warning("No U18 players are available for profile review.")
        return

    player_names = ranked_players["player_name"].tolist()
    if st.session_state.selected_player not in player_names:
        st.session_state.selected_player = player_names[0]

    search_columns = st.columns([1, 1.4])
    profile_query = search_columns[0].text_input(
        "球员搜索",
        value="",
        placeholder="输入首字母或名字，例如 lam / luc / diego",
        key="profile_player_query",
    )
    profile_candidates = build_search_candidates(profile_query, player_names)
    dataset_candidates = [name for name in profile_candidates if name in player_names]
    if not dataset_candidates:
        search_columns[1].warning("当前 sample dataset 没有这个球员，可上传完整 CSV 后查询。")
        selected_player = st.session_state.selected_player
    else:
        selected_player = search_columns[1].selectbox(
            "候选球员",
            options=dataset_candidates,
            index=dataset_candidates.index(st.session_state.selected_player)
            if st.session_state.selected_player in dataset_candidates
            else 0,
            key="selected_player",
        )
    player = ranked_players[ranked_players["player_name"] == selected_player].iloc[0]

    spotlight_columns = st.columns([0.9, 1.6])
    with spotlight_columns[0]:
        st.markdown(
            f"""
            <div class="player-card">
                <img src="{player_image_url(player)}" alt="{escape(player['player_name'])}">
                <div class="player-body">
                    <div class="tag-pill">Player Spotlight</div>
                    <div class="player-name">{escape(str(player['player_name']))}</div>
                    <div class="player-meta">{escape(str(player['position']))} | {escape(str(player['club']))}</div>
                    <div class="score-badge">Talent Score {player['talent_score']:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with spotlight_columns[1]:
        st.plotly_chart(
            make_player_radar_chart(player),
            use_container_width=True,
            key=f"profile_radar_{player['player_id']}",
        )

    render_section_heading("Player Dossier", str(player["player_name"]))
    render_player_fact_panel(player)

    visual_columns = st.columns([0.8, 1.1, 1.1])
    with visual_columns[0]:
        st.plotly_chart(
            make_talent_gauge(player),
            use_container_width=True,
            key=f"profile_gauge_{player['player_id']}",
        )
    with visual_columns[1]:
        st.plotly_chart(
            make_player_metric_bars(player),
            use_container_width=True,
            key=f"profile_metric_bars_{player['player_id']}",
        )
    with visual_columns[2]:
        st.plotly_chart(
            make_player_output_chart(player),
            use_container_width=True,
            key=f"profile_output_chart_{player['player_id']}",
        )

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

    control_columns = st.columns([1, 1.2, 0.75, 0.75])
    similar_query = control_columns[0].text_input(
        "搜索目标球员",
        value="",
        placeholder="输入首字母，例如 lam / jo / noah",
        key="similar_player_query",
    )
    similar_candidates = [
        name for name in build_search_candidates(similar_query, player_names)
        if name in player_names
    ]
    if not similar_candidates:
        control_columns[1].warning("当前数据没有匹配球员。")
        return
    target_player = control_columns[1].selectbox(
        "候选目标",
        options=similar_candidates,
        index=similar_candidates.index(default_player)
        if default_player in similar_candidates
        else 0,
    )
    top_n = control_columns[2].selectbox(
        "Number of similar players",
        options=[3, 5, 10],
        index=1,
    )
    same_position_only = control_columns[3].checkbox(
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

    render_target_summary_panel(target_row)

    fig_similarity = px.scatter(
        recommendations,
        x="similarity_score",
        y="talent_score",
        size="talent_score",
        color="position",
        hover_name="similar_player",
        title="Similarity vs Talent Score",
        labels={
            "similarity_score": "Similarity",
            "talent_score": "Talent Score",
        },
        range_x=[0, 1],
    )
    st.plotly_chart(
        fig_similarity,
        use_container_width=True,
        key=f"similarity_scatter_{target_row['player_id']}",
    )

    with st.expander("查看推荐 CSV 数据", expanded=False):
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.download_button(
        "Download similar player recommendations",
        data=dataframe_to_csv_bytes(recommendations),
        file_name="similar_players.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    render_recommendation_cards(recommendations, ranked_players)

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
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"similarity_bar_{target_row['player_id']}",
    )


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
    render_section_heading(
        "Visual Analytics",
        "数据可视化战情室",
        "把排名、市场价值、年龄、出场时间和联赛分布放在同一页，方便展示时讲清楚模型结果。",
    )
    if ranked_players.empty:
        st.warning("No U18 players are available for visual analytics.")
        return

    image_columns = st.columns(3)
    for index, column in enumerate(image_columns):
        with column:
            st.image(FOOTBALL_IMAGES[index], use_container_width=True)

    chart = make_top_players_chart(ranked_players, limit=10)
    st.plotly_chart(chart, use_container_width=True, key="analytics_top_players_chart")
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
        st.plotly_chart(fig, use_container_width=True, key="analytics_position_distribution")
        st.caption("This chart helps scouts see whether the dataset is balanced by role.")

    with chart_columns[1]:
        fig = px.histogram(
            ranked_players,
            x="talent_score",
            nbins=12,
            title="Talent Score Distribution",
            labels={"talent_score": "Talent Score"},
        )
        st.plotly_chart(fig, use_container_width=True, key="analytics_talent_distribution")
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
        st.plotly_chart(fig, use_container_width=True, key="analytics_market_vs_talent")
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
        st.plotly_chart(fig, use_container_width=True, key="analytics_age_vs_talent")
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
        st.plotly_chart(fig, use_container_width=True, key="analytics_minutes_vs_performance")
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
        st.plotly_chart(fig, use_container_width=True, key="analytics_position_average")
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
    st.plotly_chart(fig, use_container_width=True, key="analytics_league_average")
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
    st.title("ScoutAI: 红蓝主题青年球探数据中心")
    st.markdown(
        f"""
        <p class="small-muted">
        Barca-inspired visual dashboard for ranking U18 players,
        comparing hexagon profiles, and building a scout shortlist.
        Active dataset: <strong>{escape(dataset_name)}</strong>.
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
    render_global_player_search(ranked_players)

    tabs = st.tabs(
        [
            "首页 Overview",
            "数据 Data Explorer",
            "排名 Talent Ranking",
            "球员 Player Profile",
            "相似 Similar Finder",
            "模型 Score Model",
            "图表 Visual Analytics",
            "伦理 Ethics",
            "项目 About",
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
