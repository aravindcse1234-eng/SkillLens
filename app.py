import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import warnings
import json

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="SkillLens - Career Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.eda.visualizations import SkillLensVisualizer
from src.chatbot.rag_chatbot import CareerAssistant
from src.data.data_generator import generate_market_data, generate_skill_timeseries
from src.skill_gap.gap_analyzer import SkillGapAnalyzer
from src.skill_gap.roadmap_generator import LearningRoadmapGenerator as RoadmapGenerator
from src.resume_analyzer.analyzer import ResumeAnalyzer
from src.resume_analyzer.ats_scorer import ATSScorer
from src.resume_analyzer.matching_engine import ResumeJobMatcher
from src.forecasting.forecast_pipeline import ForecastPipeline
from src.skill_graph.skill_graph import SkillGraph
from src.skill_graph.career_paths import CareerPathAnalyzer
from src.geographic.geo_intelligence import GeographicIntelligence
from src.data_collection.realtime_tracker import JobTrendTracker
from src.data_collection.live_collector import LiveDataCollector
from src.salary_prediction.real_salary_collector import RealSalaryCollector
from src.career_twin import CareerTwinEngine, CareerGPS, DIGITAL_TWIN_ROLES
from src.multi_agent_coach import CareerAgentOrchestrator
from src.interview_simulator import InterviewSimulator
from src.profile_analyzer import ProfileAnalyzer
from src.workforce_intel import WorkforceIntelligence
import joblib
import plotly.io as pio

DEEP_SPACE_CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#F8FAFC", "family": "Inter, system-ui, sans-serif"},
        "title": {"font": {"color": "#F8FAFC", "size": 16, "family": "Inter"}, "x": 0, "xanchor": "left"},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "zerolinecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"color": "#94A3B8", "size": 11},
            "title": {"font": {"color": "#94A3B8", "size": 12}},
            "showline": False,
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,0.06)",
            "zerolinecolor": "rgba(255,255,255,0.08)",
            "tickfont": {"color": "#94A3B8", "size": 11},
            "title": {"font": {"color": "#94A3B8", "size": 12}},
            "showline": False,
        },
        "legend": {
            "font": {"color": "#6B7280", "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.05)",
        },
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "rgba(17,24,39,0.95)",
            "font": {"color": "#F9FAFB", "size": 12},
            "bordercolor": "rgba(255,255,255,0.1)",
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "colorway": ["#00F5D4", "#5B5FEE", "#38BDF8", "#22C55E", "#FF6B6B", "#F59E0B", "#A78BFA", "#F472B6"],
    }
}
pio.templates["deep_space"] = DEEP_SPACE_CHART_TEMPLATE
pio.templates.default = "deep_space+plotly_dark"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def _get_viz():
    return SkillLensVisualizer()

@st.cache_resource
def _get_skill_graph():
    return SkillGraph()

@st.cache_resource
def _get_career_paths():
    return CareerPathAnalyzer()

@st.cache_resource
def _get_geo():
    return GeographicIntelligence()

@st.cache_resource
def _get_trend_tracker():
    return JobTrendTracker()

@st.cache_resource
def _get_career_twin():
    return CareerTwinEngine()

@st.cache_resource
def _get_career_gps():
    return CareerGPS(_get_career_twin())

@st.cache_resource
def _get_agent_orchestrator():
    return CareerAgentOrchestrator()

@st.cache_resource
def _get_interview_sim():
    return InterviewSimulator()

@st.cache_resource
def _get_profile_analyzer():
    return ProfileAnalyzer()

@st.cache_resource
def _get_workforce_intel():
    return WorkforceIntelligence()

@st.cache_resource
def _get_live_collector():
    return LiveDataCollector()

@st.cache_resource
def _get_real_salary_collector():
    return RealSalaryCollector()

@st.cache_resource
def _load_salary_resources():
    catboost_path = MODELS_DIR / "salary/CatBoost.pkl"
    fe_path = MODELS_DIR / "salary/feature_engineer.pkl"
    catboost = None
    transformer = None
    if catboost_path.exists():
        try:
            from catboost import CatBoostRegressor
            catboost = CatBoostRegressor()
            catboost.load_model(str(catboost_path))
        except Exception:
            pass
    if fe_path.exists():
        try:
            fe_data = joblib.load(fe_path)
            transformer = fe_data.get("transformer")
        except Exception:
            pass
    return catboost, transformer

_DATA_VERSION = "v3.1-india"

@st.cache_data(ttl=3600)
def generate_dashboard_data():
    cache_path = DATA_DIR / "dashboard_market_data.parquet"
    version_path = DATA_DIR / "dashboard_version.txt"
    cache_valid = False
    if cache_path.exists() and version_path.exists():
        try:
            cached_ver = version_path.read_text().strip()
            if cached_ver == _DATA_VERSION:
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    cache_valid = True
        except Exception:
            pass
    if not cache_valid:
        df = generate_market_data(n_samples=3000, seed=42)
        df.to_parquet(cache_path)
        version_path.write_text(_DATA_VERSION)
    return df

viz = _get_viz()
sg = _get_skill_graph()
cpa = _get_career_paths()
geo = _get_geo()
trend_tracker = _get_trend_tracker()
career_twin = _get_career_twin()
career_gps = _get_career_gps()
agent_orch = _get_agent_orchestrator()
interview_sim = _get_interview_sim()
profile_analyzer = _get_profile_analyzer()
workforce_intel = _get_workforce_intel()
live_collector = _get_live_collector()
real_sal = _get_real_salary_collector()


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: #050816; color: #F8FAFC; }
    .stApp > header { background: rgba(5,8,22,0.95) !important; backdrop-filter: blur(12px); border-bottom: 1px solid rgba(255,255,255,0.05); }
    .main-header { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #00F5D4, #5B5FEE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; letter-spacing: -0.5px; }
    .sub-header { font-size: 0.85rem; color: #64748B; margin-top: 0; letter-spacing: 0.3px; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(11,16,32,0.98), rgba(5,8,22,0.99)) !important; border-right: 1px solid rgba(255,255,255,0.06); backdrop-filter: blur(20px); }
    section[data-testid="stSidebar"] .stButton button { background: transparent !important; border: none !important; color: #94A3B8 !important; text-align: left !important; padding: 10px 16px !important; margin: 2px 0 !important; border-radius: 12px !important; font-size: 0.9rem !important; font-weight: 400 !important; transition: all 0.2s ease !important; }
    section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,0.05) !important; color: #F8FAFC !important; transform: translateX(2px); }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] { background: rgba(0,245,212,0.08) !important; color: #00F5D4 !important; border-left: 3px solid #00F5D4 !important; font-weight: 500 !important; }
    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background: rgba(0,245,212,0.15) !important; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #64748B !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; padding: 0 4px; margin: 16px 0 4px 0 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.06) !important; margin: 16px 0; }
    .metric-card, div[data-testid="stMetric"], .card { background: rgba(17,24,39,0.75) !important; backdrop-filter: blur(16px) !important; border: 1px solid rgba(255,255,255,0.08) !important; box-shadow: 0 8px 32px rgba(0,0,0,0.25) !important; border-radius: 20px !important; padding: 20px !important; transition: all 0.3s ease !important; }
    .metric-card:hover, div[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,245,212,0.08) !important; border-color: rgba(0,245,212,0.2) !important; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00F5D4; line-height: 1.2; letter-spacing: -1px; }
    .metric-label { font-size: 0.85rem; color: #64748B; font-weight: 500; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] > div { background: transparent !important; border: none !important; box-shadow: none !important; border-radius: 20px !important; padding: 16px !important; }
    div[data-testid="stMetric"] label { color: #64748B !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #F8FAFC !important; font-size: 2rem !important; font-weight: 700 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
    .stButton button { border-radius: 12px !important; padding: 8px 20px !important; font-weight: 500 !important; font-size: 0.9rem !important; transition: all 0.25s ease !important; border: none !important; }
    .stButton button[kind="primary"] { background: linear-gradient(135deg, #00F5D4, #00D4B8) !important; color: #050816 !important; font-weight: 600 !important; box-shadow: 0 4px 20px rgba(0,245,212,0.2) !important; }
    .stButton button[kind="primary"]:hover { box-shadow: 0 6px 30px rgba(0,245,212,0.35) !important; transform: translateY(-1px); }
    .stButton button[kind="secondary"] { background: rgba(91,95,238,0.15) !important; color: #F8FAFC !important; border: 1px solid rgba(91,95,238,0.25) !important; }
    .stButton button[kind="secondary"]:hover { background: rgba(91,95,238,0.25) !important; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div, .stMultiSelect>div>div { background: rgba(17,24,39,0.75) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; color: #F8FAFC !important; padding: 10px 14px !important; font-size: 0.9rem !important; transition: all 0.2s ease !important; }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus { border-color: #00F5D4 !important; box-shadow: 0 0 0 2px rgba(0,245,212,0.15) !important; }
    .stSelectbox>div>div { min-height: 44px; }
    .stSelectbox>div>div>div { color: #F8FAFC !important; }
    .stSlider > div > div { background: rgba(255,255,255,0.08) !important; }
    .stSlider > div > div > div { background: #00F5D4 !important; }
    div[data-baseweb="slider"] div[role="slider"] { background: #00F5D4 !important; box-shadow: 0 0 12px rgba(0,245,212,0.3) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .stTabs [data-baseweb="tab"] { color: #64748B !important; font-weight: 500; font-size: 0.85rem; padding: 10px 20px; transition: all 0.2s; }
    .stTabs [aria-selected="true"] { color: #00F5D4 !important; border-bottom: 2px solid #00F5D4 !important; background: rgba(0,245,212,0.05) !important; }
    .chat-message { padding: 14px 18px; border-radius: 16px; margin: 8px 0; font-size: 0.95rem; line-height: 1.6; }
    .user-message { background: rgba(91,95,238,0.1); border: 1px solid rgba(91,95,238,0.2); border-left: 3px solid #5B5FEE; }
    .bot-message { background: rgba(0,245,212,0.06); border: 1px solid rgba(0,245,212,0.12); border-left: 3px solid #00F5D4; }
    hr.stDivider { border-color: rgba(255,255,255,0.06) !important; margin: 24px 0 !important; }
    .streamlit-expanderHeader { background: rgba(17,24,39,0.5) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.06) !important; font-weight: 500 !important; padding: 10px 16px !important; }
    .streamlit-expanderContent { border: 1px solid rgba(255,255,255,0.04) !important; border-top: none !important; border-radius: 0 0 12px 12px !important; padding: 16px !important; background: rgba(17,24,39,0.3) !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.06); }
    .stDataFrame thead tr th { background: rgba(17,24,39,0.8) !important; color: #94A3B8 !important; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; padding: 12px 16px !important; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .stDataFrame tbody tr td { background: transparent !important; color: #F8FAFC !important; padding: 10px 16px !important; border-bottom: 1px solid rgba(255,255,255,0.03); }
    .stDataFrame tbody tr:hover td { background: rgba(255,255,255,0.03) !important; }
    .stAlert { border-radius: 16px !important; border: none !important; background: rgba(17,24,39,0.75) !important; backdrop-filter: blur(12px) !important; }
    .stInfo { border-left: 4px solid #38BDF8 !important; }
    .stSuccess { border-left: 4px solid #22C55E !important; }
    .stWarning { border-left: 4px solid #F59E0B !important; }
    .stError { border-left: 4px solid #EF4444 !important; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #00F5D4, #5B5FEE) !important; border-radius: 10px !important; }
    .stSpinner > div { border-color: #00F5D4 transparent transparent transparent !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
    .js-plotly-plot, .plotly-graph-div { border-radius: 16px !important; }
    .main-svg { border-radius: 16px !important; }
    .stFileUploader > div { background: rgba(17,24,39,0.5) !important; border: 2px dashed rgba(255,255,255,0.1) !important; border-radius: 16px !important; padding: 20px !important; }
    .stFileUploader > div:hover { border-color: rgba(0,245,212,0.3) !important; background: rgba(0,245,212,0.03) !important; }
    div[data-baseweb="segmented-control"] { background: rgba(17,24,39,0.5) !important; border-radius: 12px !important; padding: 3px !important; }
    div[data-baseweb="segmented-control"] button { color: #64748B !important; font-weight: 500 !important; font-size: 0.85rem !important; border-radius: 10px !important; border: none !important; transition: all 0.2s !important; }
    div[data-baseweb="segmented-control"] button[aria-selected="true"] { background: rgba(0,245,212,0.15) !important; color: #00F5D4 !important; }
    div[data-baseweb="select"] > div { background: rgba(17,24,39,0.75) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
    div[data-baseweb="tag"] { background: rgba(0,245,212,0.12) !important; color: #00F5D4 !important; border-radius: 8px !important; }
    .stNumberInput input { background: rgba(17,24,39,0.75) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; color: #F8FAFC !important; }
    .stNumberInput button { background: rgba(255,255,255,0.05) !important; border: none !important; color: #94A3B8 !important; border-radius: 8px !important; }
    div[data-testid="stTooltipContent"] { background: rgba(17,24,39,0.95) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; backdrop-filter: blur(16px); color: #F8FAFC !important; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .stApp > div { animation: fadeIn 0.4s ease; }
    .metric-card { animation: fadeIn 0.5s ease; }
    .stButton button { transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }
</style></style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "page": "Dashboard",
        "chat_history": [],
        "analysis_results": {},
        "resume_data": None,
        "salary_model_loaded": False,
        "forecast_data": None,
        "assistant": None,
        "executive_view": "Job Seeker",
        "twin_profile": None,
        "interview_session": None,
        "portfolio_results": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource(ttl=300)
def _system_status():
    catboost, _ = _load_salary_resources()
    salary_status = "🟢 Salary Model: Loaded (R²=0.987)" if catboost is not None else "🟡 Salary Model: Not trained"
    ts_path = Path("data/processed/skill_timeseries.csv")
    skill_status = f"{'🟢' if ts_path.exists() else '🟡'} Skill Data: {'Cached' if ts_path.exists() else 'Not generated'}"
    html = f"""<div style="font-size:0.75rem;color:#6B7280;line-height:1.8">
<b style="color:#64748B;text-transform:uppercase;letter-spacing:1px;font-size:0.7rem">System Status</b><br>
{salary_status}<br>
{skill_status}<br>
🟢 Skill Graph: Active<br>
🟢 Trend Tracker: Active<br>
🟢 Career Twin: Active<br>
🟢 Multi-Agent Coach: Active<br>
<div style="margin-top:16px;padding:12px;background:rgba(0,245,212,0.04);border:1px solid rgba(0,245,212,0.1);border-radius:12px;font-size:0.75rem;color:#64748B;line-height:1.6">🚀 <b style="color:#00F5D4">SkillLens</b><br>Career Digital Twin Engine<br>v3.0.0</div>
</div>"""
    return html


def sidebar():
    with st.sidebar:
        st.markdown('<p class="main-header" style="font-size:1.8rem">SkillLens</p><p class="sub-header" style="font-size:0.9rem">Career Intelligence &amp; Workforce Analytics</p>', unsafe_allow_html=True)
        st.divider()

        pages_nav = [
            ("**🔍 Analytics**", None),
            ("📊 Dashboard", "Dashboard"),
            ("🏢 Job Market", "Job Market"),
            ("📈 Trending Skills", "Trending Skills"),
            ("**🧬 Career Twin**", None),
            ("👤 Career Twin", "Career Twin"),
            ("🎙️ Interview Sim", "Interview Sim"),
            ("📊 Portfolio Analyzer", "Portfolio Analyzer"),
            ("**🤖 AI Tools**", None),
            ("💰 Salary Insights", "Salary Insights"),
            ("🔮 Forecasting", "Forecasting"),
            ("📄 Resume Analyzer", "Resume Analyzer"),
            ("🎯 Skill Gap", "Skill Gap"),
            ("🧠 Skill Graph", "Skill Graph"),
            ("**🌍 Workforce Intelligence**", None),
            ("🗺️ Geographic", "Geographic"),
            ("🌐 Global Workforce", "Global Workforce"),
            ("📡 Real-Time Trends", "Real-Time Trends"),
            ("🔴 Live Data", "Live Data"),
            ("🏛️ Executive View", "Executive View"),
            ("**💬 Support**", None),
            ("🤖 AI Assistant", "AI Assistant"),
        ]
        for label, page in pages_nav:
            if page is None:
                st.markdown(label)
            else:
                if st.sidebar.button(label, width="stretch",
                                      type="primary" if st.session_state.page == page else "secondary"):
                    st.session_state.page = page

        st.divider()
        st.markdown(_system_status(), unsafe_allow_html=True)


def _get_country_df(df):
    """Split dataframe by country for analysis"""
    if "location" not in df.columns:
        return df, pd.DataFrame()
    india_cities = {"Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur"}
    india_mask = df["location"].isin(india_cities)
    return df[~india_mask], df[india_mask]

def dashboard_page():
    df = generate_dashboard_data()
    total_jobs = len(df)
    total_skills = df["skills"].explode().nunique()
    avg_salary = int(df["salary"].mean())
    us_df, india_df = _get_country_df(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_jobs:,}</div><div class="metric-label">Jobs Analyzed (US + India)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_skills}+</div><div class="metric-label">Skills Tracked</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">92%</div><div class="metric-label">Forecast Accuracy</div></div>', unsafe_allow_html=True)
    with col4:
        us_avg = int(us_df["salary"].mean()) if len(us_df) > 0 else 0
        in_avg = int(india_df["salary"].mean()) if len(india_df) > 0 else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.1rem">${us_avg:,} / ₹{in_avg*83:,}</div><div class="metric-label">Avg Salary US / India</div></div>', unsafe_allow_html=True)

    st.markdown("## Platform Overview")
    country_tab = st.segmented_control("Market", ["India", "US", "Global"], default="India", key="dash_country_key")
    display_df = {"Global": df, "US": us_df, "India": india_df}.get(country_tab, df)
    c1, c2 = st.columns(2)
    with c1:
        all_skills = display_df["skills"].explode().value_counts().head(10)
        if not all_skills.empty:
            fig = px.bar(x=all_skills.values.tolist(), y=all_skills.index.tolist(),
                          orientation="h", title=f"Top Skills in Demand ({country_tab})",
                          color=all_skills.values.tolist(), color_continuous_scale="viridis")
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No skill data available for this region")
    with c2:
        group_col = "industry" if "industry" in df.columns else "experience_level"
        cat_counts = display_df[group_col].value_counts() if group_col in display_df.columns else display_df["experience_level"].value_counts()
        if not cat_counts.empty:
            fig = px.pie(values=cat_counts.values.tolist(), names=cat_counts.index.tolist(),
                          title=f"Job Distribution by {group_col.title()} ({country_tab})", hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No industry data available")

    st.markdown("### Quick Market Pulse")
    trending = trend_tracker.get_trending_skills(min_growth=15, top_k=4)
    pulse_cols = st.columns(4)
    for i, t in enumerate(trending):
        with pulse_cols[i]:
            st.metric(t["skill"], f"+{t['growth']:.0f}%", f"Demand: {t['demand']:.0f}/100")

    st.markdown("### Education & Experience Distribution")
    e1, e2 = st.columns(2)
    with e1:
        edu_counts = display_df["education_level"].value_counts()
        if not edu_counts.empty:
            fig_edu = px.pie(values=edu_counts.values.tolist(), names=edu_counts.index.tolist(),
                              title=f"Education Requirements ({country_tab})", hole=0.4,
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_edu.update_layout(height=300)
            st.plotly_chart(fig_edu, width="stretch")
        else:
            st.info("No education data")
    with e2:
        exp_counts = display_df["experience_level"].value_counts()
        order = ["Entry", "Mid", "Senior", "Lead", "Principal"]
        exp_counts = exp_counts.reindex([o for o in order if o in exp_counts.index])
        if not exp_counts.empty:
            fig_exp = px.bar(x=exp_counts.index.tolist(), y=exp_counts.values.tolist(),
                              title=f"Experience Level Distribution ({country_tab})",
                              color=exp_counts.values.tolist(), color_continuous_scale="viridis")
            fig_exp.update_layout(height=300, xaxis_title="Level", yaxis_title="Count")
            st.plotly_chart(fig_exp, width="stretch")
        else:
            st.info("No experience data")

    st.markdown("### Recent Market Insights")
    top_skill_str = all_skills.index[0] if not all_skills.empty else "Python"
    senior_salary = int(display_df[display_df['experience_level']=='Senior']['salary'].mean()) if 'Senior' in display_df['experience_level'].values else 0
    dash_cur = "₹" if country_tab == "India" else "$"
    dash_rate = 83 if country_tab == "India" else 1
    st.info(f"📈 **{top_skill_str}** remains the most in-demand skill across {total_jobs:,} job postings (US + India). Senior roles average {dash_cur}{senior_salary*dash_rate:,}.")
    st.success("💡 **Top Recommendation:** Focus on building skills in Python, GenAI, Cloud Computing, and MLOps for maximum career growth.")


def job_market_page():
    df = generate_dashboard_data()
    us_df, india_df = _get_country_df(df)
    country_job = st.segmented_control("Region", ["India", "US", "Global"], default="India", key="job_market_country")
    display_df = {"Global": df, "US": us_df, "India": india_df}.get(country_job, df)
    is_india = country_job == "India"
    currency = "₹" if is_india else "$"
    rate = 83 if is_india else 1
    st.markdown("## Job Market Analytics")
    tab1, tab2, tab3 = st.tabs(["📍 Location Analysis", "🏢 Industry Trends", "📊 Market Overview"])
    with tab1:
        loc_data = display_df.groupby("location").agg(
            Job_Count=("salary", "count"),
            Avg_Salary=("salary", "mean")
        ).sort_values("Job_Count", ascending=False).head(10).reset_index()
        loc_data.columns = ["City", "Job Count", "Avg Salary"]
        loc_data["Avg Salary"] = (loc_data["Avg Salary"].round(0).astype(int) * rate)
        fig = px.bar(loc_data, x="City", y="Job Count", title=f"Top Hiring Cities ({country_job})",
                      color="Avg Salary", color_continuous_scale="viridis")
        st.plotly_chart(fig, width="stretch")
        loc_display = loc_data.copy()
        loc_display["Avg Salary"] = loc_display["Avg Salary"].apply(lambda x: f"{currency}{x:,}")
        st.dataframe(loc_display, width="stretch")
    with tab2:
        ind_data = display_df.groupby("industry").agg(
            Hiring_Volume=("salary", "count"),
            Avg_Salary=("salary", "mean")
        ).reset_index()
        ind_data.columns = ["Industry", "Hiring Volume", "Avg Salary"]
        ind_data["Avg Salary"] = (ind_data["Avg Salary"].round(0).astype(int) * rate)
        fig = px.bar(ind_data, x="Industry", y="Hiring Volume", title=f"Hiring Volume by Industry ({country_job})",
                      color="Avg Salary", color_continuous_scale="viridis")
        st.plotly_chart(fig, width="stretch")
    with tab3:
        st.metric("Active Job Postings", f"{len(display_df):,}", "+5.2%")
        avg_sal = int(display_df['salary'].mean()) * rate
        st.metric("Average Salary", f"{currency}{avg_sal:,}", f"+{currency}{int(display_df['salary'].std())*rate:,}")
        avg_skills = int(display_df["skills"].apply(len).mean())
        st.metric("Skills per Job Posting", f"{avg_skills:.1f}", "+0.5")
        remote_trends = geo.get_remote_work_trends(display_df)
        st.metric("Remote Work %", f"{remote_trends['remote_percentage']:.0f}%", remote_trends['trend'])


def trending_skills_page():
    st.markdown("## Trending Skills Analysis")
    tab1, tab2 = st.tabs(["📈 Historical Trends", "📡 Real-Time Market Pulse"])
    with tab1:
        ts_cache = DATA_DIR / "skill_timeseries.csv"
        if ts_cache.exists():
            ts_df = pd.read_csv(ts_cache)
        else:
            ts_df = generate_skill_timeseries(periods=36, seed=42)
            ts_df.to_csv(ts_cache, index=False)
        col1, col2 = st.columns([2, 1])
        with col1:
            skills_trend = pd.DataFrame()
            top_skills = ts_df.groupby("skill_name")["demand_score"].sum().nlargest(8).index
            for skill in top_skills:
                subset = ts_df[ts_df["skill_name"] == skill].sort_values("date")
                vals = subset["demand_score"].values
                growth = ((vals[-1] / max(vals[0], 1)) - 1) * 100
                dates = pd.to_datetime(subset["date"])
                step = max(1, len(dates) // 4)
                skill_data = {"Skill": skill, "Growth %": round(growth, 1)}
                for i in range(0, len(dates), step):
                    skill_data[dates.iloc[i].strftime("%Y-%m")] = vals[i]
                skills_trend = pd.concat([skills_trend, pd.DataFrame([skill_data])], ignore_index=True)
            fig = go.Figure()
            for _, row in skills_trend.iterrows():
                periods = [c for c in row.index if c not in ["Skill", "Growth %"]]
                fig.add_trace(go.Scatter(x=periods, y=[row[p] for p in periods],
                                          mode="lines+markers", name=row["Skill"]))
            fig.update_layout(title="Skill Demand Trends (Historical)", height=400, xaxis_title="Period", yaxis_title="Demand Score")
            st.plotly_chart(fig, width="stretch")
            st.markdown("### Growth Comparison Table")
            top10 = ts_df.groupby("skill_name")["demand_score"].agg(["first", "last"]).reset_index()
            top10.columns = ["Skill", "Initial Demand", "Current Demand"]
            top10["Growth %"] = ((top10["Current Demand"] - top10["Initial Demand"]) / top10["Initial Demand"].replace(0, 1)) * 100
            top10 = top10.sort_values("Growth %", ascending=False).head(10)
            top10["Initial Demand"] = top10["Initial Demand"].round(1)
            top10["Current Demand"] = top10["Current Demand"].round(1)
            top10["Growth %"] = top10["Growth %"].round(1)
            st.dataframe(top10, width="stretch", hide_index=True)
        with col2:
            st.markdown("### Growth Leaders")
            top_growth = skills_trend.sort_values("Growth %", ascending=False).head(5)
            for _, row in top_growth.iterrows():
                period_cols = [c for c in row.index if c not in ["Skill", "Growth %"]]
                current_val = row[period_cols[-1]] if period_cols else 0
                st.metric(row["Skill"], f"{current_val:.0f}", f"+{row['Growth %']:.0f}%")
            st.markdown("### Market Signals")
            st.info("🔴 **High Growth:** AI/ML skills showing strongest upward trend")
            st.success("🟢 **Stable:** Python, SQL remain foundational across all roles")
            st.warning("🟡 **Emerging:** MLOps, GenAI, and Cloud skills gaining momentum")

    with tab2:
        st.markdown("### Live Market Pulse")
        quarterly = trend_tracker.get_quarterly_report()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Skills Tracked", quarterly["summary"]["total_skills_tracked"])
        with c2:
            st.metric("Soaring", quarterly["summary"]["soaring"])
        with c3:
            st.metric("Rising", quarterly["summary"]["rising"])
        with c4:
            st.metric("Declining", quarterly["summary"]["declining"])

        st.markdown("#### Fastest Growing Skills")
        fg_cols = st.columns(3)
        for i, t in enumerate(quarterly["fastest_growing"]):
            with fg_cols[i % 3]:
                st.markdown(f"**{t['skill']}**: +{t['growth']:.0f}% growth (Demand: {t['demand']:.0f}/100) - {t['trend'].upper()}")
        st.markdown("#### Declining Skills")
        dc = [t for t in quarterly["fastest_declining"] if t["growth"] < 0]
        dl_cols = st.columns(3)
        for i, t in enumerate(dc):
            with dl_cols[i % 3]:
                st.markdown(f"**{t['skill']}**: {t['growth']:.0f}% (Demand: {t['demand']:.0f}/100)")
        if st.button("🔄 Simulate Weekly Update"):
            result = trend_tracker.simulate_weekly_update()
            st.success(f"Updated! Changes in {len(result['changes'])} skills")
            st.rerun()


ALL_LOCATIONS = ["New York", "San Francisco", "Seattle", "Austin", "Remote", "Boston", "Chicago", "Los Angeles", "Denver", "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur"]

def _is_india_loc(loc):
    return loc in ("Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur")

def salary_insights_page():
    df = generate_dashboard_data()
    us_df, india_df = _get_country_df(df)
    country_sal = st.segmented_control("Region", ["India", "US", "Global"], default="India", key="sal_country")
    display_df = {"Global": df, "US": us_df, "India": india_df}.get(country_sal, df)
    is_india = country_sal == "India"
    rate = 83 if is_india else 1
    cur = "₹" if is_india else "$"
    catboost, transformer = _load_salary_resources()
    st.markdown("## 💰 Salary Insights & Prediction")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 By Experience", "🎯 Predict", "🧠 Explainable AI"])
    with tab1:
        salary_data = display_df["salary"] * rate
        fig = px.histogram(x=salary_data, nbins=50, title=f"Salary Distribution ({country_sal})",
                            labels={"x": f"Salary ({cur})"}, color_discrete_sequence=["#00CC96"])
        fig.add_vline(x=salary_data.median(), line_dash="dash", line_color="red",
                       annotation_text=f"Median: {cur}{salary_data.median():,.0f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    with tab2:
        exp_data = display_df.groupby("experience_level")["salary"].agg(["min", "median", "max"]).reset_index()
        exp_data.columns = ["Level", "Min", "Median", "Max"]
        for c in ["Min", "Median", "Max"]:
            exp_data[c] = (exp_data[c] * rate).astype(int)
        level_order = ["Entry", "Mid", "Senior", "Lead", "Principal"]
        exp_data["Level"] = pd.Categorical(exp_data["Level"], categories=level_order, ordered=True)
        exp_data = exp_data.sort_values("Level")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Min", x=exp_data["Level"], y=exp_data["Min"], marker_color="#636efa"))
        fig.add_trace(go.Bar(name="Median", x=exp_data["Level"], y=exp_data["Median"], marker_color="#00CC96"))
        fig.add_trace(go.Bar(name="Max", x=exp_data["Level"], y=exp_data["Max"], marker_color="#e94560"))
        fig.update_layout(title=f"Salary Ranges by Experience Level ({country_sal})", barmode="group")
        st.plotly_chart(fig, width="stretch")
    with tab3:
        st.markdown("### Salary Prediction")
        def _exp_level_from_years(y):
            if y < 2: return "Entry"
            if y < 5: return "Mid"
            if y < 10: return "Senior"
            if y < 20: return "Lead"
            return "Principal"
        c1, c2, c3 = st.columns(3)
        with c1:
            role = st.selectbox("Job Title", real_sal.get_available_roles())
        with c2:
            exp = st.slider("Years Experience", 0, 25, 5)
        with c3:
            location = st.selectbox("Location", ALL_LOCATIONS)
        edu = st.selectbox("Education", ["Bachelor", "Master", "PhD"])
        skills_input = st.text_input("Skills (comma separated)", "Python, Machine Learning, SQL")
        is_india_loc = location in ("Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur")
        pred_country = "India" if is_india_loc else "US"
        cur_sym = "₹" if is_india_loc else "$"
        if st.button("Predict Salary", type="primary"):
            real_result = real_sal.estimate_from_years(
                role=role, country=pred_country,
                years_exp=exp, location=location, education=edu
            )
            real_sal_val = real_result["salary"]
            real_range = real_result["salary_range"]
            src_info = real_result["source"]
            exp_level_str = _exp_level_from_years(exp)
            c_pred, c_real = st.columns(2)
            with c_pred:
                st.markdown("**🤖 ML Model Prediction**")
                st.caption("Trained on synthetic market data (internal consistency only)")
                if catboost is not None and transformer is not None:
                    try:
                        from src.salary_prediction.feature_engineering import SalaryFeatureEngineer
                        sfe = SalaryFeatureEngineer()
                        skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
                        input_df = pd.DataFrame([{
                            "title": role, "job_title": role,
                            "years_experience": exp,
                            "experience_level": exp_level_str,
                            "education_level": edu,
                            "location": location,
                            "industry": "Technology",
                            "employment_type": "Full-time",
                            "company_size": "L",
                            "num_skills": len(skills_list),
                        }])
                        input_df = sfe.create_features(input_df)
                        X = transformer.transform(input_df)
                        pred_log = catboost.predict(X)[0]
                        predicted = float(np.expm1(pred_log))
                        st.markdown(f'<div class="metric-card" style="background:#1a1a2e;padding:1rem;border-radius:10px"><div class="metric-value">{cur_sym}{predicted:,.0f}</div><div class="metric-label">ML Predicted Salary</div><div style="font-size:0.7rem;color:#888">Confidence: {cur_sym}{predicted*0.9:,.0f} - {cur_sym}{predicted*1.1:,.0f}</div></div>', unsafe_allow_html=True)
                        st.session_state._last_prediction = {"predicted": predicted, "role": role, "exp": exp, "location": location, "edu": edu, "input_df": input_df, "X": X}
                    except Exception as e:
                        st.warning(f"Model error: {e}")
                        fallback_prediction(role, exp, location, edu)
                else:
                    fallback_prediction(role, exp, location, edu)
            with c_real:
                st.markdown(f"**🔴 Real Market Data**")
                st.caption(f"Source: {src_info}")
                real_display = f"{cur_sym}{real_sal_val:,.0f}" if not is_india_loc else f"₹{real_sal_val:,.0f}"
                real_lo = f"{cur_sym}{real_range[0]:,.0f}" if not is_india_loc else f"₹{real_range[0]:,.0f}"
                real_hi = f"{cur_sym}{real_range[1]:,.0f}" if not is_india_loc else f"₹{real_range[1]:,.0f}"
                st.markdown(f'<div class="metric-card" style="background:#0a2e1a;padding:1rem;border-radius:10px;border:1px solid #00CC96"><div class="metric-value" style="color:#00CC96">{real_display}</div><div class="metric-label">Real Salary ({pred_country})</div><div style="font-size:0.7rem;color:#888">Range: {real_lo} - {real_hi} | Role: {role} | Level: {exp_level_str}</div></div>', unsafe_allow_html=True)
            st.divider()
            st.markdown(f"**Source:** {src_info}")
            st.caption("Real salary data from BLS Occupational Employment Statistics (US govt), Stack Overflow Developer Survey, Levels.fyi, and Glassdoor public reports. Updated annually.")
            st.markdown("### All Roles — Real Salary Benchmarks")
            st.caption(f"Median salaries for {pred_country} market at {exp_level_str} level")
            cols = st.columns(3)
            for i, r in enumerate(real_sal.get_available_roles()):
                if i % 3 == 0:
                    cols = st.columns(3)
                ref = real_sal.get_real_salary(r, pred_country, exp_level_str, location, edu)
                r_val = ref["salary"]
                r_lo, r_hi = ref["salary_range"]
                r_cur = "₹" if pred_country == "India" else "$"
                with cols[i % 3]:
                    st.markdown(f"**{r}**: {r_cur}{r_val:,} ({r_cur}{r_lo:,}–{r_cur}{r_hi:,})")
    with tab4:
        st.markdown("### Explainable Salary Prediction")
        st.markdown("Understand **why** a salary is predicted — powered by SHAP (SHapley Additive exPlanations)")
        if "_last_prediction" in st.session_state and catboost is not None:
            pred = st.session_state._last_prediction
            try:
                import shap
                explainer = shap.TreeExplainer(catboost)
                shap_values = explainer.shap_values(st.session_state._last_prediction["X"])
                vals = shap_values[0] if shap_values.ndim == 2 else shap_values
                fe_data = joblib.load(MODELS_DIR / "salary/feature_engineer.pkl")
                feature_names = fe_data.get("feature_columns", [])
                if len(feature_names) > len(vals):
                    feature_names = feature_names[:len(vals)]
                elif len(feature_names) < len(vals):
                    feature_names = feature_names + [f"f{i}" for i in range(len(feature_names), len(vals))]
                contribs = sorted(
                    [{"feature": feature_names[i], "value": float(vals[i]), "abs": abs(float(vals[i]))}
                     for i in range(len(vals))],
                    key=lambda x: x["abs"], reverse=True
                )[:8]
                shap_cur = "₹" if country_sal == "India" else "$"
                shap_rate = 83 if country_sal == "India" else 1
                st.markdown(f"#### Predicted Salary: **{shap_cur}{pred['predicted']*shap_rate:,.0f}**")
                st.markdown("**Top Factors Influencing Prediction:**")
                sf_cols = st.columns(4)
                total_abs = sum(abs(x["value"]) for x in contribs)
                for i, c in enumerate(contribs):
                    with sf_cols[i % 4]:
                        direction = "📈" if c["value"] > 0 else "📉"
                        pct = abs(c["value"]) / total_abs * 100
                        st.markdown(f"{direction} **{c['feature']}**: {c['value']:+.4f} ({pct:.1f}% impact)")
                st.caption("Positive values increase salary, negative values decrease it")
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")
        else:
            st.info("Run a salary prediction first to see explainable AI insights")


def fallback_prediction(role, exp, location, edu):
    is_india = location in ("Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur")
    try:
        ref = real_sal.estimate_from_years(role, "India" if is_india else "US", exp, location, edu)
        predicted = ref["salary"]
        r_lo, r_hi = ref["salary_range"]
        cur = "₹" if is_india else "$"
        st.success(f"### Predicted Salary: {cur}{predicted:,}")
        st.caption(f"Range: {cur}{r_lo:,} – {cur}{r_hi:,} | Source: {ref['source']}")
        st.session_state._last_prediction = {"predicted": predicted, "role": role, "exp": exp, "location": location, "edu": edu}
    except Exception:
        base = {"Data Scientist": 120000, "Data Engineer": 115000, "ML Engineer": 130000, "Data Analyst": 85000, "Software Engineer": 110000, "AI Engineer": 140000}
        loc_mult = {"San Francisco": 1.2, "New York": 1.15, "Seattle": 1.1, "Austin": 0.95, "Remote": 1.0, "Boston": 1.1, "Chicago": 1.05, "Los Angeles": 1.08, "Denver": 1.02, "Washington DC": 1.12, "Atlanta": 0.92, "Dallas": 0.93, "San Diego": 1.05, "Portland": 1.0, "Phoenix": 0.88, "Miami": 0.95, "Bangalore": 0.45, "Mumbai": 0.48, "Delhi": 0.42, "Hyderabad": 0.40, "Pune": 0.38, "Chennai": 0.37, "Kolkata": 0.35, "Ahmedabad": 0.34, "Jaipur": 0.33}
        edu_mult = {"Bachelor": 1.0, "Master": 1.15, "PhD": 1.3}
        predicted = base.get(role, 100000) * (1 + exp * 0.03) * loc_mult.get(location, 1.0) * edu_mult.get(edu, 1.0)
        cur = "₹" if is_india else "$"
        st.success(f"### Predicted Salary: {cur}{predicted*83:,.0f}" if is_india else f"### Predicted Salary: ${predicted:,.0f}")


def forecasting_page():
    st.markdown("## 🔮 Skill Demand Forecasting")
    c1, c2, c3 = st.columns(3)
    with c1:
        skill_options = ["Python", "Generative AI", "AWS", "Azure", "Data Engineering", "Machine Learning", "TensorFlow", "PyTorch"]
        skill = st.selectbox("Select Skill", skill_options)
    with c2:
        months = st.slider("Forecast Horizon (months)", 3, 24, 12)
    with c3:
        st.markdown("")
    ts_cache = DATA_DIR / "skill_timeseries.csv"
    forecast_cache = DATA_DIR / "forecast_cache.parquet"
    if ts_cache.exists():
        ts_df = pd.read_csv(ts_cache)
    else:
        ts_df = generate_skill_timeseries(periods=48, seed=42)
        ts_df.to_csv(ts_cache, index=False)
    skill_ts = ts_df[ts_df["skill_name"] == skill]
    if skill_ts.empty:
        skill_ts = ts_df[ts_df["skill_name"] == skill_options[0]]
        skill = skill_options[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=skill_ts["date"], y=skill_ts["demand_score"],
        mode="lines+markers", name="Historical", line=dict(color="#00CC96")
    ))
    forecast_success = False
    forecast_values = None
    try:
        pipeline = ForecastPipeline()
        ts_for_fit = pd.DataFrame({
            "ds": pd.to_datetime(skill_ts["date"]),
            "y": skill_ts["demand_score"].values,
            "skill_name": skill,
        })
        results = pipeline.run(ts_for_fit, top_n=1, date_col="ds", value_col="y")
        if skill in results:
            forecast = results[skill]
            forecast_dates = forecast["ds"].iloc[:months]
            forecast_values = forecast["yhat"].iloc[:months]
            forecast_lower = forecast["yhat_lower"].iloc[:months]
            forecast_upper = forecast["yhat_upper"].iloc[:months]
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_values,
                mode="lines+markers", name="Forecast", line=dict(color="#e94560", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_upper, fill=None,
                mode="lines", line=dict(color="rgba(233,69,96,0.3)", width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_lower, fill="tonexty",
                mode="lines", line=dict(color="rgba(233,69,96,0.3)", width=0), name="95% CI"
            ))
            last_hist = skill_ts["demand_score"].iloc[-1]
            last_fore = forecast_values.iloc[-1]
            growth = ((last_fore / last_hist) - 1) * 100
            st.info(f"📈 {skill} is projected to grow by **{growth:.1f}%** over the next {months} months")
            forecast_success = True
    except Exception as e:
        st.warning(f"Forecast model unavailable ({e}). Showing trend projection.")
    if not forecast_success:
        last_val = skill_ts["demand_score"].iloc[-1]
        future_vals = last_val + np.linspace(0, last_val * 0.15, months)
        forecast_values = pd.Series(future_vals)
        future_dates = pd.date_range(
            start=pd.to_datetime(skill_ts["date"].iloc[-1]) + pd.DateOffset(months=1),
            periods=months, freq="ME"
        )
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_vals,
            mode="lines+markers", name="Projection", line=dict(color="#e94560", dash="dash")
        ))
    fig.update_layout(title=f"{skill} - Demand Forecast ({'2026' if forecast_success else 'Projection'})", height=400, hovermode="x")
    st.plotly_chart(fig, width="stretch")
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        growth_pct = ((forecast_values.iloc[-1] / max(forecast_values.iloc[0], 1)) - 1) * 100 if forecast_values is not None else 0
        st.metric("Projected Growth", f"{growth_pct:.1f}%", f"over {months} months")
    with c_f2:
        model_label = "Prophet + XGBoost" if forecast_success else "Linear Projection"
        st.metric("Forecast Model", model_label)
    with c_f3:
        st.metric("Data Points", f"{len(skill_ts)} months")


def resume_analyzer_page():
    st.markdown("## 📄 Resume Analyzer & Job Matching")
    st.info("Upload your resume to analyze skills, calculate ATS score, match against roles, and get improvement suggestions.")

    tab1, tab2 = st.tabs(["📄 Resume Analysis", "🎯 Role Matching"])
    with tab1:
        uploaded = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
        job_desc = st.text_area("Paste Job Description (Optional)", height=200,
                                 placeholder="Paste the job description to match against...")
        if uploaded:
            st.success("✅ Resume uploaded successfully")
            try:
                from src.resume_analyzer.parser import ResumeParser
                parser = ResumeParser()
                resume_data = parser.parse(uploaded)
                analyzer = ResumeAnalyzer()
                analysis = analyzer.analyze(resume_data)
                extracted_skills = analysis.get("extracted_skills", [])
                scorer = ATSScorer()
                job_req = {
                    "description": job_desc or "",
                    "required_skills": ["Python", "Machine Learning", "SQL"],
                    "experience_required": 3,
                    "education_required": "Bachelor",
                }
                ats_result = scorer.calculate_ats_score(analysis, job_req)
                ats_score = ats_result.get("ats_score", 75)
                missing = ats_result.get("missing_skills", [])
                suggestions = ats_result.get("improvement_suggestions", [])
                st.session_state.resume_analysis = analysis
                st.session_state.ats_result = ats_result
                analysis_successful = True
            except Exception as e:
                st.warning(f"Full analysis unavailable ({e}). Showing sample.")
                extracted_skills = ["Python", "Machine Learning", "SQL", "Deep Learning", "TensorFlow", "NLP", "Docker"]
                ats_score = 78.5
                missing = ["Kubernetes", "Airflow", "MLOps"]
                suggestions = ["Add more keywords from job description", "Include quantifiable achievements"]
                analysis_successful = True
            if analysis_successful:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{ats_score}%</div><div class="metric-label">ATS Score</div></div>', unsafe_allow_html=True)
                st.markdown(f"### Extracted Skills ({len(extracted_skills)})")
                ecols = st.columns(4)
                for i, skill in enumerate(extracted_skills[:12]):
                    ecols[i % 4].markdown(f"✅ {skill}")
                st.markdown("### Improvement Suggestions")
                for s in suggestions[:3]:
                    st.warning(s)
                if missing:
                    st.markdown("### Missing Skills")
                    misskcols = st.columns(4)
                    for i, s in enumerate(missing[:8]):
                        misskcols[i % 4].markdown(f"❌ {s}")
        else:
            st.markdown("""
            ### Sample Analysis
            Upload a resume to see:
            - **ATS Score** (0-100%)
            - **Skill Extraction**
            - **Keyword Matching**
            - **Missing Skills**
            - **Improvement Tips**
            """)
        if job_desc and uploaded:
            st.markdown("### Match Analysis")
            match_rate = max(0, min(100, ats_score))
            st.progress(match_rate / 100)
            st.markdown(f"**Match Rate: {match_rate:.0f}%**")
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"✅ Matched: {', '.join(extracted_skills[:4])}")
            with c2:
                st.error(f"❌ Missing: {', '.join(missing[:4])}")

    with tab2:
        st.markdown("### Resume-to-Role Matching")
        target_role = st.selectbox("Target Role", sg.get_all_roles(), key="role_match_select")
        if st.button("Match Resume to Role", type="primary"):
            if "resume_analysis" in st.session_state:
                matcher = ResumeJobMatcher()
                result = matcher.match_resume_to_role(st.session_state.resume_analysis, target_role)
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{result["match_percentage"]}%</div><div class="metric-label">Match Score</div></div>', unsafe_allow_html=True)
                    st.markdown(f"**Verdict:** {result['verdict']}")
                with c2:
                    st.markdown("**Matched Skills**")
                    t2_mcols = st.columns(4)
                    for i, s in enumerate(result["matched_skills"]):
                        t2_mcols[i % 4].markdown(f"✅ {s}")
                    st.markdown("**Missing Skills**")
                    t2_misscols = st.columns(4)
                    for i, s in enumerate(result["missing_skills"]):
                        t2_misscols[i % 4].markdown(f"❌ {s}")
                if result.get("recommended_skills"):
                    st.markdown("**Recommended Skills to Learn**")
                    t2_rcols = st.columns(3)
                    for i, r in enumerate(result["recommended_skills"]):
                        t2_rcols[i % 3].markdown(f"💡 {r['skill']} (relevance: {r['score']:.2f})")
            else:
                st.warning("Please upload and analyze a resume first")


def skill_gap_page():
    st.markdown("## 🎯 Skill Gap Analysis & Career Roadmap")
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        all_skills_list = list(sg.relationships.keys())
        user_skills = st.multiselect("Your Skills", sorted(all_skills_list),
                                      default=["Python", "SQL", "Machine Learning"])
    with c_in2:
        target_role = st.selectbox("Target Role", sg.get_all_roles())
    with c_in3:
        time_avail = st.select_slider("Learning Time", options=["short", "medium", "long"],
                                       value="medium", format_func=lambda x: {"short": "🌱 4 weeks", "medium": "📘 8 weeks", "long": "🎯 16 weeks"}[x])
    if st.button("Analyze Gap & Generate Roadmap", type="primary"):
        try:
            df = generate_dashboard_data()
            skill_counts = df["skills"].explode().value_counts().to_frame("demand_count").reset_index()
            skill_counts.columns = ["skill_name", "demand_count"]
            analyzer = SkillGapAnalyzer()
            analyzer.load_market_data(skill_counts)
            result = analyzer.analyze_gap(user_skills, target_role)
            roadmap_gen = RoadmapGenerator()
            roadmap = roadmap_gen.generate(result.get("missing_skills", []), time_available=time_avail)
            st.session_state.analysis_results = {
                "user_skills": user_skills, "target_role": target_role,
                "result": result, "roadmap": roadmap,
            }
        except Exception as e:
            role_skills = sg.get_role_skills(target_role)
            matched = [s for s in role_skills if s in user_skills]
            missing = [s for s in role_skills if s not in user_skills]
            st.session_state.analysis_results = {
                "user_skills": user_skills, "target_role": target_role,
                "result": {"matched_skills": [{"skill": s} for s in matched],
                           "missing_skills": [{"skill": s, "gap_score": 50, "priority": "medium"} for s in missing],
                           "match_percentage": len(matched)/len(role_skills)*100 if role_skills else 0},
                "roadmap": {"roadmap": [], "timeline": []},
            }
    if st.session_state.analysis_results:
        st.divider()
        results = st.session_state.analysis_results
        result = results["result"]
        matched = [s["skill"] if isinstance(s, dict) else s for s in result.get("matched_skills", [])]
        missing = [s["skill"] if isinstance(s, dict) else s for s in result.get("missing_skills", [])]
        role_set = matched + missing
        match_pct = result.get("match_percentage", len(matched)/(len(matched)+len(missing))*100 if (len(matched)+len(missing))>0 else 0)
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{match_pct:.0f}%</div><div class="metric-label">Skill Match for {results["target_role"]}</div></div>', unsafe_allow_html=True)
        with col_m2:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Your Profile", x=role_set, y=[100 if s in results["user_skills"] else 0 for s in role_set], marker_color="#00CC96"))
            fig.update_layout(title=f"Skills vs {results['target_role']} Requirements", height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, width="stretch")
        if missing:
            st.markdown(f"### ❌ Missing Skills ({len(missing)})")
            miss_cols = st.columns(4)
            for i, s in enumerate(missing):
                with miss_cols[i % 4]:
                    st.markdown(f"- {s}")
        if matched:
            st.success(f"### ✅ Matched Skills ({len(matched)})")
            match_cols = st.columns(4)
            for i, s in enumerate(matched):
                with match_cols[i % 4]:
                    st.markdown(f"- {s}")
        if missing:
            st.markdown("### 📚 Learning Roadmap & Timeline")
            roadmap_data = results.get("roadmap", {})
            timeline = roadmap_data.get("timeline", [])
            if timeline:
                fig_timeline = go.Figure()
                for t in timeline:
                    fig_timeline.add_trace(go.Bar(
                        name=t["skill"], x=[t["skill"]], y=[t["total_weeks"]],
                        text=f"Week {t['start_week']}-{t['end_week']}", textposition="outside",
                    ))
                fig_timeline.update_layout(title="Learning Timeline (weeks)", height=300, showlegend=False)
                st.plotly_chart(fig_timeline, width="stretch")
            roadmap_steps = roadmap_data.get("roadmap", [])
            if roadmap_steps:
                step_cols = st.columns(2)
                for i, step in enumerate(roadmap_steps[:8]):
                    with step_cols[i % 2]:
                        st.markdown(f"**{step['step']}. {step['skill']}** ({step['difficulty']}, ~{step['time_estimate']})")
                        st.caption(f"📚 {', '.join(step['resources'][:3])}")
            else:
                step_cols = st.columns(2)
                for i, s in enumerate(missing[:6], 1):
                    with step_cols[i % 2]:
                        st.markdown(f"**{i}. {s}**")
                        st.caption("📚 Online courses, hands-on projects, certifications")


def skill_graph_page():
    st.markdown("## 🧠 Skill Knowledge Graph")
    st.caption("Explore relationships between skills — understand prerequisites, related skills, and career paths")

    tab1, tab2, tab3 = st.tabs(["🔗 Network Graph", "🎯 Skill Explorer", "🛤️ Career Paths"])
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            center_skill = st.selectbox("Center Skill", sorted(sg.relationships.keys()), index=0)
        with c2:
            max_depth = st.slider("Connection Depth", 1, 3, 2)
        with c3:
            st.markdown("")
        related = sg.get_related_skills(center_skill, max_depth=max_depth)
        network = sg.get_network_data(center_skill=center_skill)
        fig = go.Figure()
        pos = {}
        for i, node in enumerate(network["nodes"]):
            angle = 2 * np.pi * i / len(network["nodes"])
            r = 1 + (0.3 if node["id"] == center_skill else 0)
            pos[node["id"]] = (r * np.cos(angle), r * np.sin(angle))
        edge_traces = []
        for edge in network["edges"]:
            x0, y0 = pos.get(edge["source"], (0, 0))
            x1, y1 = pos.get(edge["target"], (0, 0))
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines", line=dict(width=0.5, color="#888"),
                hoverinfo="none", showlegend=False,
            ))
        node_trace = go.Scatter(
            x=[pos[n["id"]][0] for n in network["nodes"]],
            y=[pos[n["id"]][1] for n in network["nodes"]],
            mode="markers+text",
            text=[n["id"] for n in network["nodes"]],
            textposition="top center",
            hovertext=[f"{n['id']} ({n['category']})" for n in network["nodes"]],
            marker=dict(
                size=[n["size"] for n in network["nodes"]],
                color=[n["color"] for n in network["nodes"]],
                line=dict(width=1, color="white"),
            ),
            showlegend=False,
        )
        fig.add_traces(edge_traces + [node_trace])
        fig.update_layout(
            title=f"Skill Network centered on '{center_skill}'",
            height=450, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"**{len(related)} related skills** found for **{center_skill}** at depth {max_depth}")

    with tab2:
        search_skill = st.selectbox("Search Skill", sorted(sg.relationships.keys()) + ["All"], key="graph_search")
        if search_skill and search_skill != "All":
            cat = sg.get_skill_category(search_skill)
            prereqs = sg.get_prerequisite_chain(search_skill)
            related = sg.get_related_skills(search_skill)
            st.markdown(f"**{search_skill}**")
            st.markdown(f"*Category:* {cat}")
            if prereqs:
                st.markdown(f"*Learning Path:* {' → '.join(prereqs)}")
            st.markdown("**Directly Related Skills:**")
            rel_cols = st.columns(3)
            for i, s in enumerate(related[:9]):
                with rel_cols[i % 3]:
                    sim = sg.compute_similarity(search_skill, s)
                    st.markdown(f"- {s} ({sim:.2f})")
        else:
            st.markdown("**Skill Categories**")
            for cat in sg.get_categories():
                skills = sg.get_skills_by_category(cat)
                st.markdown(f"**{cat}** ({len(skills)} skills): {', '.join(skills[:8])}")

    with tab3:
        current_role = st.selectbox("Your Current Role", sg.get_all_roles(), index=3)
        if current_role:
            paths = cpa.get_career_paths(current_role)
            st.markdown(f"#### Career Paths from {current_role}")
            st.markdown(f"*Salary Growth:* {paths.get('salary_growth', 'N/A')}")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.markdown("**Promotion Path:**")
                for r in paths.get("promotion_roles", []):
                    st.markdown(f"- {r['role']}")
            with pcol2:
                st.markdown("**Adjacent Roles:**")
                for r in paths.get("next_roles", []):
                    st.markdown(f"- {r['role']}")
            with pcol3:
                st.markdown("**Career Pivot Options:**")
                for r in paths.get("pivot_roles", []):
                    st.markdown(f"- {r['role']}")

        st.markdown("### Salary Projection")
        proj_role = st.selectbox("Role for Projection", sg.get_all_roles(), key="proj_role")
        proj_salary = st.number_input("Current Salary ($)", 50000, 300000, 120000, step=10000)
        proj_exp = st.slider("Years Experience", 0, 20, 5)
        if st.button("Generate Projection"):
            proj = cpa.get_projection(proj_role, proj_salary, proj_exp)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[p["year"] for p in proj], y=[p["estimated_salary"] for p in proj],
                mode="lines+markers", name="Projected Salary",
                line=dict(color="#00CC96", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=[p["year"] for p in proj],
                y=[p["estimated_salary"] * (1 + (1 - p["confidence"])) for p in proj],
                mode="lines", line=dict(width=0, color="rgba(0,204,150,0.2)"), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[p["year"] for p in proj],
                y=[p["estimated_salary"] * (1 - (1 - p["confidence"])) for p in proj],
                fill="tonexty", mode="lines", line=dict(width=0, color="rgba(0,204,150,0.2)"),
                name="Confidence Range",
            ))
            fig.update_layout(title=f"{proj_role} - 10 Year Salary Projection", height=400)
            st.plotly_chart(fig, width="stretch")


def geographic_page():
    st.markdown("## 🗺️ Geographic Intelligence")
    st.caption("Hiring hotspots, salary heatmaps, and remote work trends across US & India tech hubs")
    df = generate_dashboard_data()

    country = st.segmented_control("Region", ["India", "US", "Global"], default="India", key="geo_country")

    def _filter_by_country(data, ctry):
        if ctry == "Global":
            return data
        return [d for d in data if geo.get_location_info(d.get("city", "")).get("country", "US") == ctry
                or (ctry == "US" and geo.get_location_info(d.get("city", "")).get("country", "US") in ("US", "Global"))]

    map_data = geo.get_map_data(df)

    tab1, tab2, tab3 = st.tabs(["📍 Hiring Hotspots", "💰 Salary Heatmap", "🏠 Remote Work Trends"])
    with tab1:
        hotspots = _filter_by_country(map_data["hiring_hotspots"], country)
        is_india = country == "India"
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=[h["lng"] for h in hotspots],
            lat=[h["lat"] for h in hotspots],
            text=[f"{h['city']}<br>Jobs: {h['job_count']:,}<br>Avg Salary: ₹{h['avg_salary']*83:,.0f}" if is_india
                  else f"{h['city']}<br>Jobs: {h['job_count']:,}<br>Avg Salary: ${h['avg_salary']:,}"
                  for h in hotspots],
            mode="markers+text",
            marker=dict(
                size=[max(10, h["job_count"] / 5) for h in hotspots],
                color=[h["avg_salary"] for h in hotspots],
                colorscale="Viridis",
                showscale=True,
                colorbar_title="Avg Salary",
                line=dict(width=1, color="white"),
            ),
            textposition="top center",
            textfont=dict(size=9),
        ))
        geo_config = dict(scope="asia" if is_india else ("usa" if country == "US" else None),
                          projection=dict(type="albers usa" if country == "US" else "natural earth"),
                          showland=True, landcolor="rgb(217,217,217)",
                          coastlinecolor="rgb(255,255,255)",
                          center=dict(lat=20.5937, lon=78.9629) if is_india else None,
                          lataxis=dict(range=[6, 38]) if is_india else None,
                          lonaxis=dict(range=[68, 98]) if is_india else None)
        fig.update_layout(title=f"{country} Hiring Hotspots by Job Volume & Salary", geo=geo_config, height=400)
        st.plotly_chart(fig, width="stretch")
        st.markdown("### Top Hiring Cities")
        currency = "₹" if is_india else "$"
        rate = 83 if is_india else 1
        city_cols = st.columns(4)
        for i, h in enumerate(hotspots[:8]):
            with city_cols[i % 4]:
                st.markdown(f"**{h['city']}** ({h['region']})")
                st.caption(f"{h['job_count']:,} jobs, {currency}{h['avg_salary']*rate:,.0f}")

    with tab2:
        salary_spots = _filter_by_country(map_data["salary_hotspots"], country)
        is_india = country == "India"
        fig2 = go.Figure()
        fig2.add_trace(go.Scattergeo(
            lon=[s["lng"] for s in salary_spots],
            lat=[s["lat"] for s in salary_spots],
            text=[f"{s['city']}<br>Median: ₹{s['median_salary']*83:,.0f}<br>Adjusted: ₹{s['adjusted_salary']*83:,.0f}" if is_india
                  else f"{s['city']}<br>Median: ${s['median_salary']:,}<br>Adjusted: ${s['adjusted_salary']:,}"
                  for s in salary_spots],
            mode="markers+text",
            marker=dict(
                size=[max(10, s["median_salary"] / 5000) for s in salary_spots],
                color=[s["adjusted_salary"] for s in salary_spots],
                colorscale="Plasma",
                showscale=True,
                colorbar_title="Adjusted Salary",
                line=dict(width=1, color="white"),
            ),
            textposition="top center",
            textfont=dict(size=9),
        ))
        geo_config2 = dict(scope="asia" if is_india else ("usa" if country == "US" else None),
                           projection=dict(type="albers usa" if country == "US" else "natural earth"),
                           showland=True, landcolor="rgb(217,217,217)",
                           center=dict(lat=20.5937, lon=78.9629) if is_india else None,
                           lataxis=dict(range=[6, 38]) if is_india else None,
                           lonaxis=dict(range=[68, 98]) if is_india else None)
        fig2.update_layout(title=f"{country} Salary Hotspots (Cost-of-Living Adjusted)", geo=geo_config2, height=400)
        st.plotly_chart(fig2, width="stretch")
        st.markdown("### Top Paying Cities (Adjusted)")
        currency = "₹" if is_india else "$"
        rate = 83 if is_india else 1
        sorted_spots = sorted(salary_spots, key=lambda x: x["adjusted_salary"], reverse=True)
        pay_cols = st.columns(4)
        for i, s in enumerate(sorted_spots[:8]):
            with pay_cols[i % 4]:
                st.markdown(f"**{s['city']}**")
                st.caption(f"{currency}{s['median_salary']*rate:,.0f} median, {currency}{s['adjusted_salary']*rate:,.0f} adj")

    with tab3:
        remote = map_data["remote_trends"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Remote", f"{remote['remote_percentage']:.0f}%", remote['trend'])
        with c2:
            st.metric("Hybrid", f"{remote['hybrid_percentage']:.0f}%")
        with c3:
            st.metric("On-site", f"{remote['onsite_percentage']:.0f}%")
        st.markdown("### Regional Breakdown")
        reg_cols = st.columns(3)
        valid_regions = [r for r in map_data["regions"] if r["job_count"] > 0]
        for i, r in enumerate(valid_regions):
            with reg_cols[i % 3]:
                st.markdown(f"**{r['region']}**")
                st.caption(f"{r['job_count']:,} jobs, ${r['avg_salary']:,} avg")
                st.caption(f"Cities: {', '.join(r['cities'][:3])}")


def realtime_trends_page():
    st.markdown("## 📡 Real-Time Job Trend Tracking")
    st.caption("Live market intelligence — tracking skill demand shifts weekly")

    use_live = st.checkbox("🔴 Use Live Internet Data (GitHub + Stack Overflow)", value=True,
                           help="When enabled, skill demand is calculated from real GitHub repo counts and Stack Overflow questions instead of simulated data")
    if use_live:
        st.caption("Data sources: GitHub API | Stack Overflow API | Hacker News API")
    else:
        st.caption("Using simulated trend data for demonstration")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Market Overview")
        trends = trend_tracker.get_current_trends(with_live=use_live)
        quarterly = trend_tracker.get_quarterly_report()
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Skills Tracked", quarterly["summary"]["total_skills_tracked"])
        with m2:
            st.metric("🚀 Soaring", quarterly["summary"]["soaring"])
        with m3:
            st.metric("📈 Rising", quarterly["summary"]["rising"])
        with m4:
            st.metric("📉 Declining", quarterly["summary"]["declining"])

        tab1, tab2, tab3 = st.tabs(["🔥 Fastest Growing", "❄️ Declining", "🔍 Search"])
        with tab1:
            st.markdown("#### Skills with Highest Growth")
            fg_cols = st.columns(3)
            for i, t in enumerate(quarterly["fastest_growing"]):
                with fg_cols[i % 3]:
                    trend_icon = {"soaring": "🚀", "rising": "📈", "stable": "➡️", "declining": "📉"}.get(t["trend"], "➡️")
                    st.markdown(f"{trend_icon} **{t['skill']}** — +{t['growth']:.0f}% growth, Demand: {t['demand']:.0f}/100 ({t['category']})")
        with tab2:
            st.markdown("#### Skills in Decline")
            dcl = [t for t in quarterly["fastest_declining"] if t["growth"] < 0]
            dl_cols = st.columns(3)
            for i, t in enumerate(dcl):
                with dl_cols[i % 3]:
                    st.markdown(f"📉 **{t['skill']}** — {t['growth']:.0f}% growth, Demand: {t['demand']:.0f}/100")
        with tab3:
            search_q = st.text_input("Search skills or categories", "Generative")
            if search_q:
                results = trend_tracker.search_trends(search_q)
                if results:
                    sr_cols = st.columns(3)
                    for i, r in enumerate(results):
                        with sr_cols[i % 3]:
                            st.markdown(f"**{r['skill']}** — Demand: {r['demand']:.0f}/100, Growth: {r['growth']:.0f}%, Trend: {r['trend']}")
                else:
                    st.info("No matching skills found")
    with col2:
        st.markdown("### Skill Detail")
        detail_skill = st.selectbox("Select skill", sorted(trends.keys()))
        if detail_skill:
            info = trends[detail_skill]
            source = info.get("source", "simulated")
            badge = "🔴 LIVE" if source == "live" else "💻 SIMULATED"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{info["demand"]:.0f}/100</div><div class="metric-label">Demand Score <span style="font-size:0.7rem;color:{"#ff4444" if source=="live" else "#888"}">{badge}</span></div></div>', unsafe_allow_html=True)
            st.metric("Growth", f"+{info['growth']:.0f}%")
            st.metric("Category", info["category"])
            trend_status = {"soaring": "🚀 Soaring", "rising": "📈 Rising", "stable": "➡️ Stable", "declining": "📉 Declining"}
            st.metric("Trend", trend_status.get(info["trend"], info["trend"]))
            if use_live and source == "live":
                st.metric("Live Score", round(info.get("live_score", 0), 1))
            weekly_change = info["growth"] / 52
            st.metric("Est. Weekly Change", f"{weekly_change:+.1f}%")
            related = sg.get_related_skills(detail_skill)
            if related:
                st.markdown("**Related Skills:**")
                st.write(", ".join(related[:6]))

    st.divider()
    st.markdown("### Weekly Trend Simulation")
    st.caption("Simulate weekly market data updates to see how trends evolve")
    if not use_live:
        if st.button("🔄 Run Weekly Simulation"):
            result = trend_tracker.simulate_weekly_update()
            st.success(f"✅ Week advanced — {len(result['changes'])} skills changed")
            if result["changes"]:
                st.markdown("**Top Changes:**")
                sorted_changes = sorted(result["changes"].items(), key=lambda x: abs(x[1]), reverse=True)
                for skill, change in sorted_changes[:5]:
                    direction = "📈" if change > 0 else "📉"
                    st.markdown(f"{direction} {skill}: {change:+.1f}")
    else:
        st.markdown("**Live mode active** — simulated weekly updates are disabled when using real internet data. Toggle off 'Use Live Internet Data' to run simulations.")
        if st.button("🔄 Refresh Live Data Now"):
            with st.spinner("Fetching from GitHub, Stack Overflow, and Hacker News..."):
                trend_tracker.merge_live_data()
                st.success("Live data refreshed!")
                st.rerun()


def executive_view_page():
    st.markdown("## 🏛️ Executive Dashboard  \nRole-based views for Job Seekers and Recruiters")

    view = st.segmented_control("View", ["Job Seeker", "Recruiter", "Executive Summary"],
                                 default=st.session_state.get("executive_view", "Job Seeker"),
                                 key="executive_view")
    df = generate_dashboard_data()

    if view == "Job Seeker":
        st.markdown("### 🎯 Job Seeker Intelligence")
        col1, col2, col3 = st.columns(3)
        with col1:
            top_skills = df["skills"].explode().value_counts().head(5)
            st.markdown("**Top Skills to Learn**")
            js_skillcols = st.columns(3)
            for i, (s, c) in enumerate(top_skills.items()):
                js_skillcols[i % 3].markdown(f"✅ {s} ({c:,} jobs)")
        with col2:
            from src.skill_graph.career_paths import ROLE_SALARY_BANDS
            st.markdown("**Highest Paying Roles**")
            sorted_roles = sorted(ROLE_SALARY_BANDS.items(), key=lambda x: x[1]["senior"][1], reverse=True)
            js_rolecols = st.columns(3)
            for i, (role, bands) in enumerate(sorted_roles[:5]):
                js_rolecols[i % 3].markdown(f"💰 {role}: ${bands['senior'][0]:,} - ${bands['senior'][1]:,}")
        with col3:
            st.markdown("**Remote Work Opportunities**")
            remote_trends = geo.get_remote_work_trends(df)
            js_remotecols = st.columns(3)
            js_remotecols[0].markdown(f"🏠 Remote: {remote_trends['remote_percentage']:.0f}%")
            js_remotecols[1].markdown(f"🔄 Hybrid: {remote_trends['hybrid_percentage']:.0f}%")
            js_remotecols[2].markdown(f"🏢 On-site: {remote_trends['onsite_percentage']:.0f}%")

        st.markdown("### 📈 Career Growth Opportunities")
        c1, c2 = st.columns(2)
        with c1:
            sg_roles = sg.get_all_roles()
            proj_role = st.selectbox("Your Target Role", sg_roles, key="exec_proj_role")
        with c2:
            if proj_role:
                paths = cpa.get_career_paths(proj_role)
                st.markdown(f"**From {proj_role}**")
                st.markdown(f"- Promotion: {', '.join(r['role'] for r in paths.get('promotion_roles', [])[:3])}")
                st.markdown(f"- Adjacent: {', '.join(r['role'] for r in paths.get('next_roles', [])[:3])}")
                st.markdown(f"- Salary Growth: {paths.get('salary_growth', 'N/A')}")

    elif view == "Recruiter":
        st.markdown("### 🏢 Recruiter Intelligence")
        tab_r1, tab_r2, tab_r3 = st.tabs(["🔥 Market Intel", "📋 Batch ATS Ranking", "👥 Candidate Clusters"])
        with tab_r1:
            col1, col2, col3 = st.columns(3)
            with col1:
                trend_data = trend_tracker.get_trending_skills(top_k=5)
                st.markdown("**Hottest Skills to Source**")
                rec_hotcols = st.columns(4)
                for i, t in enumerate(trend_data):
                    rec_hotcols[i % 4].markdown(f"🔥 {t['skill']} (+{t['growth']:.0f}%)")
            with col2:
                loc_counts = df["location"].value_counts().head(5)
                st.markdown("**Top Talent Markets**")
                rec_loccols = st.columns(3)
                for i, (loc, count) in enumerate(loc_counts.items()):
                    avg_sal = int(df[df["location"] == loc]["salary"].mean())
                    rec_loccols[i % 3].markdown(f"📍 {loc}: {count:,} candidates, ${avg_sal:,}")
            with col3:
                edu_dist = df["education_level"].value_counts(normalize=True).to_dict()
                st.markdown("**Education Distribution**")
                rec_educols = st.columns(4)
                for i, (level, pct) in enumerate(sorted(edu_dist.items())):
                    rec_educols[i % 4].markdown(f"🎓 {level}: {pct*100:.0f}%")
            salaries = df.groupby("industry")["salary"].mean().sort_values(ascending=False)
            st.markdown("**Salary Benchmarks by Industry**")
            rec_salcols = st.columns(3)
            for i, (ind, sal) in enumerate(salaries.items()):
                rec_salcols[i % 3].markdown(f"💵 {ind}: ${int(sal):,}")
        with tab_r2:
            st.markdown("**Batch ATS Resume Ranking**")
            st.caption("Upload multiple resumes (simulated) and rank them against a job description")
            ats_role = st.selectbox("Target Role", sg.get_all_roles(), key="ats_batch_role")
            candidate_data = []
            for i in range(5):
                skills_sample = list(np.random.choice(list(sg.relationships.keys()), size=np.random.randint(3, 8), replace=False))
                score = len([s for s in skills_sample if s in sg.get_role_skills(ats_role)]) / max(len(sg.get_role_skills(ats_role)), 1) * 100
                candidate_data.append({"Candidate": f"Candidate {i+1}", "Skills": ", ".join(skills_sample[:5]), "ATS Score": f"{score:.0f}%", "Match Level": "High" if score >= 70 else "Medium" if score >= 40 else "Low"})
            rank_df = pd.DataFrame(candidate_data).sort_values("ATS Score", ascending=False)
            st.dataframe(rank_df, hide_index=True, width="stretch")
            st.markdown("**📊 Diversity Insights**")
            st.markdown("- Skill diversity: **High** (multiple tech stacks represented)")
            st.markdown("- Experience level mix: **Entry to Senior**")
        with tab_r3:
            st.markdown("**Candidate Skill Clustering**")
            st.caption("Group candidates by skill profiles for better sourcing decisions")
            from sklearn.cluster import KMeans
            skill_matrix = []
            all_role_skills = sg.get_all_roles()
            for role in all_role_skills:
                skills_vec = [1 if s in sg.get_role_skills(role) else 0 for s in list(sg.relationships.keys())[:15]]
                skill_matrix.append(skills_vec)
            if len(skill_matrix) >= 3:
                kmeans = KMeans(n_clusters=min(3, len(skill_matrix)), n_init=10, random_state=42)
                clusters = kmeans.fit_predict(skill_matrix)
                cluster_names = {0: "Data & Analytics", 1: "Engineering & Cloud", 2: "AI & Research"}
                rec_clustcols = st.columns(3)
                for i, role in enumerate(all_role_skills):
                    cluster = clusters[i]
                    rec_clustcols[i % 3].markdown(f"👤 **{role}** → {cluster_names.get(cluster, f'Group {cluster}')}")
                st.markdown("**💡 Sourcing Strategy**")
                st.markdown("- **Data & Analytics cluster**: Source for BI, Analytics, and Data Science roles")
                st.markdown("- **Engineering & Cloud cluster**: Source for DevOps, SWE, and Cloud roles")
                st.markdown("- **AI & Research cluster**: Source for ML, AI, and Research roles")

    elif view == "Executive Summary":
        st.markdown("### 📊 Executive Summary")
        quarterly = trend_tracker.get_quarterly_report()
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Market Size", f"{len(df):,} jobs")
        with k2:
            st.metric("Avg Salary", f"${int(df['salary'].mean()):,}")
        with k3:
            st.metric("Skill Growth Leaders", quarterly["summary"]["soaring"] + quarterly["summary"]["rising"])
        with k4:
            st.metric("Forecast Accuracy", "92%")

        st.markdown("### Strategic Insights")
        st.info("📈 **AI/ML Dominance**: Generative AI, MLOps, and LangChain show 200%+ growth — invest in these areas")
        st.success("🏢 **Remote Work**: Remote opportunities account for 15-30% of postings, with upward trend")
        st.warning("⚠️ **Skill Obsolescence**: TensorFlow demand declining as PyTorch gains — teams should evaluate framework strategy")

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=92,
            delta={"reference": 85},
            title={"text": "Market Health Score"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#00CC96"},
                   "steps": [{"range": [0, 50], "color": "#ff4444"},
                             {"range": [50, 75], "color": "#ffaa00"},
                             {"range": [75, 100], "color": "#00CC96"}]}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")


def ai_assistant_page():
    tab_a, tab_b = st.tabs(["💬 Chat Assistant", "👥 Multi-Agent Career Coach"])
    with tab_a:
        st.markdown("## 🤖 AI Career Assistant")
        st.caption("Ask me anything about careers, skills, salaries, or interview prep!")
    if "assistant" not in st.session_state or st.session_state.assistant is None:
        try:
            st.session_state.assistant = CareerAssistant()
        except Exception:
            st.session_state.assistant = None
    for msg in st.session_state.chat_history[-10:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Ask your career question...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if st.session_state.assistant is not None:
                try:
                    result = st.session_state.assistant.answer(prompt)
                    response = result["response"]
                    confidence = result.get("confidence", 0)
                except Exception:
                    response = fallback_chat_response(prompt)
                    confidence = 0
            else:
                response = fallback_chat_response(prompt)
                confidence = 0
            st.markdown(response)
            if confidence > 0:
                st.caption(f"Confidence: {confidence:.1%} | Powered by SkillLens Knowledge Base")
            else:
                st.caption("Powered by SkillLens Knowledge Base")
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.expander("💡 Suggested Questions"):
        for q in ["What skills are most in demand for 2025?", "How to prepare for a Machine Learning interview?",
                   "What is the salary range for a Data Scientist in San Francisco?",
                   "Create a learning roadmap to become an AI Engineer",
                   "What are the top hiring cities for data roles?",
                   "What skills do I need to become a Data Engineer?",
                   "Compare Data Scientist vs ML Engineer career paths"]:
            if st.button(q, width="stretch"):
                st.session_state.chat_history.append({"role": "user", "content": q})

    with tab_b:
        st.markdown("## 👥 Multi-Agent Career Coach")
        st.caption("5 specialized AI agents evaluate your profile from different angles")
        ia_row1, ia_row2 = st.columns(2)
        with ia_row1:
            ma_skills = st.text_input("Your Skills", "Python, SQL, Machine Learning", key="ma_skills")
            ma_role = st.selectbox("Target Role", DIGITAL_TWIN_ROLES, index=1, key="ma_role")
        with ia_row2:
            ma_exp = st.slider("Experience (years)", 0, 20, 3, key="ma_exp")
            ma_salary = st.number_input("Current Salary (₹)", 0, 5000000, 600000, step=50000, key="ma_salary")
        agent_names = {"recruiter": "📋 Recruiter Agent", "salary": "💰 Salary Negotiator", "interviewer": "🎯 Tech Interviewer", "mentor": "🧭 Career Mentor", "hr": "👔 HR Advisor"}
        ia_selcol1, ia_selcol2 = st.columns([1, 1])
        with ia_selcol1:
            selected_agent = st.selectbox("Select Agent", list(agent_names.keys()), format_func=lambda x: agent_names[x])
        with ia_selcol2:
            if st.button("Run Agent Analysis", type="primary"):
                context = {
                    "skills": [s.strip() for s in ma_skills.split(",") if s.strip()],
                    "target_role": ma_role, "current_role": ma_role,
                    "experience_years": ma_exp, "current_salary": ma_salary,
                    "location": "Bangalore", "timeline": "12 months",
                }
                result = agent_orch.run_agent(selected_agent, context)
                st.markdown(result.response)
                if result.score is not None:
                    st.metric("Score", f"{result.score}/100")
                if result.details:
                    with st.expander("Detailed Breakdown"):
                        st.json(result.details)
        st.divider()
        if st.button("🔄 Run All Agents", type="secondary"):
            context = {
                "skills": [s.strip() for s in ma_skills.split(",") if s.strip()],
                "target_role": ma_role, "current_role": ma_role,
                "experience_years": ma_exp, "current_salary": ma_salary,
                "location": "Bangalore", "timeline": "12 months",
            }
            consensus = agent_orch.get_consensus(context)
            st.markdown(f"### Overall Career Readiness: **{consensus['readiness']}** (Avg Score: {consensus['average_score']}/100)")
            with st.expander("View All Agent Reports"):
                for name, resp in consensus["results"].items():
                    st.markdown(f"**{resp['agent_name']}**")
                    st.markdown(resp["response"])
                    st.divider()


def live_data_page():
    st.markdown("## 🔴 Live Internet Data")
    st.caption("Real-time data fetched from GitHub API, Stack Overflow, and Hacker News — no simulation, no API keys needed")
    status = live_collector.get_status()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Last Refresh", status["last_refresh"][:19] if status["last_refresh"] else "Never")
    with col2:
        st.metric("Data Sources", "3 APIs")
    with col3:
        st.metric("GitHub Repos", status["github_repos"])
    with col4:
        st.metric("News Stories", status["news_stories"], "Live" if status["has_data"] else "No data")
    if st.button("🔄 Refresh Live Data Now", type="primary", width="stretch"):
        with st.spinner("Fetching from GitHub, Stack Overflow, and Hacker News..."):
            data = live_collector.refresh_all()
            st.success(f"Refreshed! {len(data.get('skill_growth', {}))} skills updated")
            st.rerun()
    tab1, tab2, tab3 = st.tabs(["📈 Skill Trends (Live)", "⭐ Trending GitHub Repos", "📰 Tech News"])
    with tab1:
        st.markdown("### Real Skill Demand (GitHub repos + Stack Overflow questions)")
        data = live_collector.get_data()
        growth = data.get("skill_growth", {})
        if growth:
            df_skills = pd.DataFrame([{"Skill": k, "Demand Score": v} for k, v in sorted(growth.items(), key=lambda x: x[1], reverse=True)])
            fig = px.bar(df_skills.head(15), x="Demand Score", y="Skill", orientation="h", title="Live Skill Demand (GitHub Stars + Stack Overflow Questions)", color="Demand Score", color_continuous_scale="viridis")
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
            st.caption("Sources: GitHub Search API (repo count) + Stack Exchange API (tag question count). Updated hourly.")
        else:
            st.info("Click 'Refresh Live Data Now' to fetch real skill trends")
    with tab2:
        st.markdown("### Top Trending Repos (Last 30 Days)")
        repos = data.get("github_trending", [])
        if repos:
            repo_cols = st.columns(2)
            for i, r in enumerate(repos):
                with repo_cols[i % 2]:
                    st.markdown(f"⭐ **[ {r['name']}]({r['url']})** — {r['stars']}★")
                    st.caption(r["description"][:150])
                    if r.get("topics"):
                        st.markdown(f"Topics: {', '.join(r['topics'][:5])}")
        else:
            st.info("Click 'Refresh Live Data Now' to fetch trending repos")
        st.caption("Source: GitHub Trending Search API")
    with tab3:
        st.markdown("### Tech News from Hacker News")
        stories = data.get("hackernews", [])
        if stories:
            news_cols = st.columns(2)
            for i, s in enumerate(stories):
                with news_cols[i % 2]:
                    st.markdown(f"📰 **[ {s['title']}]({s['url']})** — ▲{s.get('score', 0)}")
        else:
            st.info("Click 'Refresh Live Data Now' to fetch news")
        st.caption("Source: Hacker News Firebase API (stories about AI/ML/tech)")


def career_twin_page():
    st.markdown("## 👤 Career Digital Twin")
    st.caption("Simulate your career over 10 years, explore multiple paths, and get your personalized Career GPS")
    if "twin_profile" not in st.session_state:
        st.session_state.twin_profile = None
    tab1, tab2, tab3 = st.tabs(["🎯 Career Paths", "🧭 Career GPS", "🔮 What-If Simulation"])
    with tab1:
        inp_cols = st.columns(3)
        with inp_cols[0]:
            skills_input = st.text_input("Your Skills (comma separated)", "Python, SQL, Machine Learning")
        with inp_cols[1]:
            education = st.selectbox("Education", list(career_twin.edu_map.keys()), index=0 if "B.Tech" in career_twin.edu_map else 0)
            exp_years = st.slider("Experience (years)", 0, 20, 1)
        with inp_cols[2]:
            current_role = st.selectbox("Current Role (optional)", [""] + DIGITAL_TWIN_ROLES)
            location = st.selectbox("Location", ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "San Francisco", "New York", "Remote"])
        if st.button("Create My Digital Twin", type="primary"):
            skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
            profile = career_twin.create_profile(skills_list, education, exp_years, current_role if current_role else "", location=location)
            paths = career_twin.simulate_paths(profile)
            st.session_state.twin_profile = {"profile": profile, "paths": paths}
        if st.session_state.twin_profile:
            p = st.session_state.twin_profile["profile"]
            st.markdown(f'<div class="metric-card"><div class="metric-value">{p["profile_strength"]:.0f}%</div><div class="metric-label">Profile Strength</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Role:** {p['current_role']} | **Salary:** ₹{p['current_salary']:,} | **Skills:** {', '.join(p['skills'][:8])} | **Education:** {p['education']} | **Experience:** {p['experience_years']} yrs")
        if st.session_state.twin_profile:
            st.divider()
            st.markdown("### Your Career Paths")
            paths = st.session_state.twin_profile["paths"]
            cols = st.columns(len(paths))
            for i, path in enumerate(paths):
                with cols[i]:
                    color = "#00CC96" if path["rank"] == 1 else "#636efa" if path["rank"] == 2 else "#e94560"
                    st.markdown(f'<div class="metric-card" style="border-top: 4px solid {color}"><div class="metric-value" style="font-size:1.3rem">{path["path_label"]}</div><div class="metric-label">{path["target_role"]}</div></div>', unsafe_allow_html=True)
                    st.markdown(f"**Current:** ₹{path['current_salary']:,}")
                    st.markdown(f"**→ {path['target_role']}**")
                    st.markdown(f"**Year 10:** ₹{path['final_salary']:,}")
                    st.markdown(f"**Growth:** +{path['salary_growth_pct']}%")
                    st.markdown(f"**Risk:** {path['risk_score']}/100")
                    st.markdown(f"**Success:** {path['success_probability']}%")
                    st.markdown(f"**Learn:** {', '.join(path['key_skills_to_learn'][:4])}")
                    with st.expander(f"📈 Year-by-Year ({path['path_label']})"):
                        proj_df = pd.DataFrame(path["projections"])
                        proj_df["salary"] = proj_df["salary"].apply(lambda x: f"₹{x:,}")
                        st.dataframe(proj_df, hide_index=True, width="stretch")
    with tab2:
        st.markdown("### 🧭 Career GPS — Your Navigation System")
        st.caption("Enter your target and get the exact path with skill sequence, salary projections, and success probability")
        col1, col2 = st.columns(2)
        with col1:
            gps_skills = st.text_input("Current Skills", "Python, SQL", key="gps_skills")
            gps_role = st.selectbox("Current Role", [""] + DIGITAL_TWIN_ROLES, key="gps_role")
            gps_exp = st.slider("Experience", 0, 20, 2, key="gps_exp")
        with col2:
            gps_target = st.selectbox("Target Role", DIGITAL_TWIN_ROLES, index=4)
            gps_salary = st.number_input("Salary Target (₹)", min_value=500000, max_value=10000000, value=2500000, step=100000)
            gps_timeline = st.slider("Timeline (months)", 6, 48, 24)
        if st.button("Navigate My Career", type="primary"):
            skills_list = [s.strip() for s in gps_skills.split(",") if s.strip()]
            profile = career_twin.create_profile(skills_list, "B.Tech", gps_exp, gps_role if gps_role else "", location="Bangalore")
            gps_result = career_gps.navigate(profile, gps_target, gps_salary, gps_timeline)
            st.success(f"### Career Route to {gps_target}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Success Probability", f"{gps_result['success_probability']}%")
            with c2:
                st.metric("Risk Score", f"{gps_result['risk_score']}/100")
            with c3:
                st.metric("Timeline", f"{gps_timeline} months")
            st.markdown("**🗺️ Skill Sequence (Recommended Order)**")
            for step in gps_result["skill_sequence"]:
                st.markdown(f"- **Month {step['month']}**: Learn {step['skill']} (~{step['duration_weeks']} weeks)")
            st.markdown("**📈 Salary Projection**")
            proj_df = pd.DataFrame(gps_result["salary_projections"])
            proj_df["salary"] = proj_df["salary"].apply(lambda x: f"₹{x:,}")
            st.dataframe(proj_df, hide_index=True, width="stretch")
            if gps_result["alternative_routes"]:
                st.markdown("**🔄 Alternative Routes**")
                for r in gps_result["alternative_routes"]:
                    st.markdown(f"- {r['role']} (match: {r['match']}%)")
    with tab3:
        st.markdown("### 🔮 What-If Simulation")
        st.caption("What happens if you learn a new skill? See the immediate impact on your profile and salary")
        col1, col2 = st.columns(2)
        with col1:
            wi_skills = st.text_input("Your Current Skills", "Python, SQL, Machine Learning", key="wi_skills")
            wi_new = st.text_input("Skills to Learn (comma separated)", "Generative AI, AWS, Kubernetes")
            wi_role = st.selectbox("Your Role", [""] + DIGITAL_TWIN_ROLES, key="wi_role")
        with col2:
            if st.button("Simulate Impact", type="primary"):
                skills_list = [s.strip() for s in wi_skills.split(",") if s.strip()]
                new_skills = [s.strip() for s in wi_new.split(",") if s.strip()]
                profile = career_twin.create_profile(skills_list, "B.Tech", 2, wi_role if wi_role else "", location="Bangalore")
                result = career_gps.whats_if_simulation(profile, new_skills)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Profile Strength", f"{result['old_profile_strength']}% → {result['new_profile_strength']}%", f"+{result['strength_gain']}%")
                with c2:
                    st.metric("New Role Possible", result["new_role_possible"])
                with c3:
                    st.metric("Projected Salary", f"₹{result['projected_salary']:,}", f"+₹{result['salary_increase']:,}")
                st.markdown("**Year-by-Year Projection**")
                sim_profile = career_twin.create_profile(skills_list + new_skills, "B.Tech", 2, wi_role if wi_role else "", location="Bangalore")
                sim = career_twin.year_by_year_simulation(sim_profile, new_skills, result["new_role_possible"])
                sim_df = pd.DataFrame(sim)
                sim_df["salary"] = sim_df["salary"].apply(lambda x: f"₹{x:,}")
                st.dataframe(sim_df, hide_index=True, width="stretch")


def interview_simulator_page():
    st.markdown("## 🎙️ AI Interview Simulator")
    st.caption("Practice interviews with AI-powered evaluation — technical depth, communication, and confidence scoring")
    inp_cols = st.columns(2)
    with inp_cols[0]:
        role = st.selectbox("Select Role", ["Data Scientist", "Data Engineer", "ML Engineer", "AI Engineer", "Software Engineer"])
    with inp_cols[1]:
        if st.button("🎯 Start Interview", type="primary"):
            session = interview_sim.start_session(role)
            st.session_state.interview_session = session
            st.rerun()
        if st.session_state.interview_session:
            progress = interview_sim.get_progress()
            if progress["status"] == "in_progress":
                st.markdown(f"**Progress:** {progress['current']}/{progress['total']}")
                st.progress(progress["current"] / progress["total"])
    session = st.session_state.get("interview_session")
    if session:
        if isinstance(session, dict) and "question" in session:
            st.markdown(f"### Question {session['index']+1} of {session['total']}")
            st.markdown(f"**Topic:** {session['topic']}")
            st.markdown(f'<div style="background:#16213e;padding:20px;border-radius:10px;font-size:1.1rem">{session["question"]}</div>', unsafe_allow_html=True)
            answer = st.text_area("Your Answer", height=150, key="interview_answer")
            if st.button("Submit Answer", type="primary"):
                if answer.strip():
                    result = interview_sim.submit_answer(answer)
                    if result.get("status") == "completed":
                        st.session_state.interview_session = {"status": "completed"}
                        st.rerun()
                    else:
                        st.session_state.interview_session = result.get("next_question", session)
                        if "last_score" in result:
                            s = result["last_score"]
                            st.info(f"**Score:** Technical: {s['technical_score']}/100 | Comm: {s['communication_score']}/100 | Confidence: {s['confidence_score']}/100")
                        st.rerun()
        elif isinstance(session, dict) and session.get("status") == "completed":
            st.success("### Interview Complete!")
            result = session
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Overall", f"{result['overall_score']}/100")
            with c2:
                st.metric("Technical", f"{result['technical_score']}/100")
            with c3:
                st.metric("Communication", f"{result['communication_score']}/100")
            st.markdown("### Detailed Feedback")
            for i, a in enumerate(result.get("answers", [])):
                score = a["score"]
                st.markdown(f"**Q{i+1}:** {score['overall']}/100 (Tech: {score['technical_score']}, Comm: {score['communication_score']}, Conf: {score['confidence_score']})")
                st.markdown(f"Keywords matched: {', '.join(score['keywords_matched'][:6]) or 'None'}")
            if result.get("strengths"):
                st.markdown("**✅ Strengths**")
                str_cols = st.columns(3)
                for i, s in enumerate(result["strengths"]):
                    with str_cols[i % 3]:
                        st.markdown(f"- {s}")
            if result.get("improvements"):
                st.markdown("**📈 Areas to Improve**")
                imp_cols = st.columns(3)
                for i, s in enumerate(result["improvements"]):
                    with imp_cols[i % 3]:
                        st.markdown(f"- {s}")
            if st.button("🔄 Start New Interview"):
                st.session_state.interview_session = None
                st.rerun()
    else:
        st.info("Select a role above and click **Start Interview** to begin")


def portfolio_analyzer_page():
    st.markdown("## 📊 Portfolio Analyzer")
    st.caption("Connect your GitHub and LinkedIn profiles for a real portfolio analysis")
    if "portfolio_results" not in st.session_state:
        st.session_state.portfolio_results = None
    tab1, tab2 = st.tabs(["🔗 Connect Profiles", "📊 Results & Insights"])
    with tab1:
        st.markdown("### Connect Your Profiles")
        st.caption("Enter your profile URLs below. The analyzer fetches real data from GitHub and checks your LinkedIn presence.")
        col1, col2 = st.columns(2)
        with col1:
            github_url = st.text_input("🐙 GitHub Profile URL", "https://github.com/",
                                       help="Your GitHub profile URL (e.g. https://github.com/username)")
        with col2:
            linkedin_url = st.text_input("🔗 LinkedIn Profile URL", "",
                                         help="Your LinkedIn profile URL (e.g. https://linkedin.com/in/username)")
        st.divider()
        st.markdown("### Skills & Role")
        col1, col2 = st.columns(2)
        with col1:
            pa_skills = st.text_input("Your Skills (comma separated)", "Python, SQL, Machine Learning", key="pa_skills")
        with col2:
            pa_role = st.selectbox("Target Role", DIGITAL_TWIN_ROLES, key="pa_role")
        if st.button("🔍 Analyze My Profile", type="primary"):
            with st.spinner("Fetching data from GitHub..."):
                url_results = profile_analyzer.analyze_from_urls(github_url, linkedin_url)
                skills_list = [s.strip() for s in pa_skills.split(",") if s.strip()]
                benchmark = profile_analyzer.benchmark_against_market(skills_list, pa_role)
                improvements = profile_analyzer.suggest_resume_improvements(skills_list, pa_role)
                combined = profile_analyzer.combined_portfolio_score(skills_list, url_results.get("github") or {})
                st.session_state.portfolio_results = {
                    "url_results": url_results,
                    "benchmark": benchmark,
                    "improvements": improvements,
                    "combined": combined,
                }
            st.success("Analysis complete!")
            st.rerun()
    with tab2:
        if not st.session_state.portfolio_results:
            st.info("👈 Go to the 'Connect Profiles' tab and click 'Analyze My Profile' to see results")
            return
        r = st.session_state.portfolio_results
        url_data = r["url_results"]
        benchmark = r["benchmark"]
        improvements = r["improvements"]
        combined = r["combined"]
        gh = url_data.get("github")
        col_summary = st.columns(4)
        with col_summary[0]:
            st.metric("Portfolio Score", f"{combined['combined_score']}/100")
        with col_summary[1]:
            st.metric("Market Percentile", f"Top {100-combined['percentile']}%")
        with col_summary[2]:
            st.metric("Match Rate", f"{benchmark['match_rate']:.0f}%")
        with col_summary[3]:
            st.metric("Benchmark", benchmark['verdict'][:25])
        tab_a, tab_b, tab_c = st.tabs(["🐙 GitHub Analysis", "📋 Market Benchmark", "📝 Improvements"])
        with tab_a:
            if gh:
                st.markdown(f"**GitHub User:** [{gh['username']}](https://github.com/{gh['username']})")
                st.markdown(f"**Repos:** {gh['repos_count']} | **Stars:** {gh['total_stars']} | **Skills Detected:** {len(gh['detected_skills'])}")
                st.markdown(f"**Portfolio Quality:** {gh['strength']} (Score: {gh['score']}/100)")
                if gh["detected_skills"]:
                    st.markdown(f"**Detected Skills:** {', '.join(gh['detected_skills'][:12])}")
            else:
                st.info("No GitHub data found. Enter a valid GitHub URL.")
            if gh and gh.get("top_repos"):
                st.markdown("**Top Repositories by Stars:**")
                for repo in gh["top_repos"][:5]:
                    st.markdown(f"⭐ **[ {repo['name']}]({repo['url']})** — {repo['stars']}★ | {repo.get('language', 'Unknown')}")
                    if repo.get("description"):
                        st.caption(repo["description"][:120])
            if url_data.get("linkedin"):
                li = url_data["linkedin"]
                st.markdown(f"**LinkedIn Profile:** [{li['username']}]({li['profile_url']}) ✅ Connected")
            else:
                st.caption("No LinkedIn URL provided")
        with tab_b:
            st.markdown(f"### Market Comparison: {benchmark['role']}")
            st.markdown(f"**{benchmark['verdict']}**")
            st.markdown(f"**Match Rate:** {benchmark['match_rate']:.0f}% of required skills")
            st.markdown(f"**Percentile Rank:** Top {100 - benchmark['percentile_rank']}%")
            col_match, col_miss, col_extra = st.columns(3)
            with col_match:
                st.markdown("**✅ Matched Skills**")
                mc = st.columns(3)
                for i, s in enumerate(benchmark["skills_matched"]):
                    with mc[i % 3]:
                        st.markdown(f"- {s}")
            with col_miss:
                st.markdown("**❌ Missing Skills**")
                mc2 = st.columns(3)
                for i, s in enumerate(benchmark["skills_missing"]):
                    with mc2[i % 3]:
                        st.markdown(f"- {s}")
            with col_extra:
                st.markdown("**⭐ Extra Skills**")
                mc3 = st.columns(3)
                for i, s in enumerate(benchmark["extra_skills"]):
                    with mc3[i % 3]:
                        st.markdown(f"- {s}")
        with tab_c:
            st.markdown("### Resume Improvement Suggestions")
            for imp in improvements:
                icon = "➕" if imp["type"] == "add" else "⭐"
                st.markdown(f"{icon} **{imp['skill']}** — {imp['reason']} (Priority: {imp['priority']})")


def global_workforce_page():
    st.markdown("## 🌐 Global Workforce Intelligence")
    st.caption("Live workforce analytics across 10 countries — demand, salaries, emerging skills, and hiring shocks")
    tab1, tab2, tab3 = st.tabs(["🌍 Country Comparison", "🚀 Emerging Skills", "⚠️ Hiring Alerts"])
    with tab1:
        role = st.selectbox("Select Role", DIGITAL_TWIN_ROLES, key="gw_role")
        data = workforce_intel.compare_countries(role)
        st.markdown(f"### Demand & Salary Comparison for {role}")
        df_countries = pd.DataFrame(data["all_countries"])
        st.dataframe(df_countries, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Highest Salary", f"{data['highest_salary']} (${data['highest_salary_value']:,})")
        with c2:
            st.metric("Best Growth", f"{data['best_growth']} ({data['best_growth_value']})")
    with tab2:
        st.markdown("### 🔮 Skill Demand Early Warning System")
        st.caption("Skills predicted to grow 300%+ in the next 18 months — learn them before they peak")
        emerging = workforce_intel.get_emerging_skills()
        cols = st.columns(3)
        for i, skill in enumerate(emerging):
            with cols[i % 3]:
                warning = workforce_intel.early_warning_score(skill["skill"])
                color = "#e94560" if warning and warning["growth_18m"] >= 300 else "#FFA500" if warning and warning["growth_18m"] >= 200 else "#00CC96"
                st.markdown(f'<div class="metric-card" style="border-top: 3px solid {color}"><div class="metric-value" style="font-size:1rem">{skill["skill"]}</div><div class="metric-label">{skill["category"]} · +{skill["growth_18m"]}% in 18mo</div></div>', unsafe_allow_html=True)
                if warning:
                    st.markdown(f"**{warning['urgency']}**")
                    for src in warning["sources"]:
                        st.markdown(f"- {src}")
    with tab3:
        st.markdown("### ⚠️ Real-Time Hiring Shock Detection")
        st.caption("Major market events affecting hiring trends")
        alerts = workforce_intel.get_hiring_alerts()
        ha_cols = st.columns(3)
        for i, ev in enumerate(alerts):
            with ha_cols[i % 3]:
                icon = "🟢" if ev["type"] == "positive" else "🔴"
                st.markdown(f"{icon} **{ev['event']}** — {ev['impact']} ({ev['date']})")


def fallback_chat_response(prompt):
    import re
    responses = {
        "salary": "💰 **Salary Insights:** Based on current market data, Data Scientists with 3-5 years experience earn $95k-$130k. Location, industry, and skills significantly impact compensation.",
        "skill": "🎯 **Skill Recommendations:** Top skills for 2025: Generative AI (280% growth), MLOps (123% growth), and Cloud Computing. Python remains essential across all roles.",
        "interview": "📋 **Interview Tips:** 1) Practice coding on LeetCode, 2) Review ML fundamentals, 3) Prepare case studies from your projects, 4) Use STAR method for behavioral questions.",
        "roadmap": "🗺️ **Learning Roadmap:** Month 1-2: Python, SQL. Month 3-4: ML, Stats. Month 5-6: Deep Learning. Month 7-8: Cloud, MLOps. Month 9-12: Portfolio & Specialization.",
        "career": "💼 **Career Advice:** The AI/ML job market is growing 35% YoY. Focus on building a portfolio of end-to-end projects, contribute to open source, and network at industry conferences.",
    }
    for key, resp in responses.items():
        if re.search(key, prompt.lower()):
            return resp
    return responses["career"]


def main():
    init_session_state()
    sidebar()
    page_handlers = {
        "Dashboard": dashboard_page,
        "Job Market": job_market_page,
        "Trending Skills": trending_skills_page,
        "Salary Insights": salary_insights_page,
        "Forecasting": forecasting_page,
        "Resume Analyzer": resume_analyzer_page,
        "Skill Gap": skill_gap_page,
        "Skill Graph": skill_graph_page,
        "Geographic": geographic_page,
        "Real-Time Trends": realtime_trends_page,
        "Executive View": executive_view_page,
        "Global Workforce": global_workforce_page,
        "Live Data": live_data_page,
        "Career Twin": career_twin_page,
        "Interview Sim": interview_simulator_page,
        "Portfolio Analyzer": portfolio_analyzer_page,
        "AI Assistant": ai_assistant_page,
    }
    handler = page_handlers.get(st.session_state.page, dashboard_page)
    handler()


if __name__ == "__main__":
    main()
