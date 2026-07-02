# SkillLens — Agent Context

## Project Title
**SkillLens: An Intelligent Career Intelligence Platform for Skill Demand Forecasting, Salary Prediction, Resume Matching, and Workforce Analytics Using NLP, Machine Learning, Explainable AI, and Generative AI**

## Project Overview
SkillLens is a Data Science capstone project and a production-grade **Career Intelligence & Workforce Analytics Platform** spanning 12+ modules: data collection, NLP, forecasting, salary prediction, resume analysis, skill gap analysis, skill knowledge graph, career path analysis, geographic intelligence, real-time trend tracking, RAG chatbot, explainable AI, and MLOps — all served via a Streamlit dashboard and FastAPI backend.

## Quick Reference

| Command | Description |
|---------|-------------|
| `python -m src.train_pipeline --mode quick` | Quick test (2000 records) |
| `python -m src.train_pipeline --mode demo` | Full demo (5000 records + forecasting) |
| `python -m src.train_pipeline --mode quick --mlflow` | With MLflow tracking |
| `python -m pytest tests/ -v` | Run all tests |
| `streamlit run app.py` | Launch dashboard (12 pages) |
| `uvicorn api.main:app --reload` | Launch API (9 endpoints) |
| `docker-compose up --build` | Full stack with MLflow + Postgres |

## Dashboard Pages (12 total)
- **Dashboard** — KPIs, market pulse, education/experience distribution
- **Job Market** — Location, industry, market overview with remote work stats
- **Trending Skills** — Historical trends + real-time market pulse with weekly simulations
- **Salary Insights** — Distribution, experience analysis, prediction, **Explainable AI (SHAP)**
- **Forecasting** — Prophet/XGBoost demand forecast with 95% CI
- **Resume Analyzer** — ATS scoring + **Resume-to-Role Matching Engine**
- **Skill Gap** — Gap analysis + **Personalized Learning Roadmap with Timeline**
- **Skill Graph** — **Knowledge graph** network viz, skill explorer, career path projections
- **Geographic** — **Interactive US maps** for hiring hotspots, salary heatmaps, remote trends
- **Real-Time Trends** — **Live market pulse** with quarterly reports and weekly simulations
- **Executive View** — **Job Seeker/Recruiter/Executive Summary** role-based dashboards
- **AI Assistant** — RAG chatbot with career knowledge base enrichment

## Key Architecture
```
                    Data Sources (Generated/Kaggle/O*NET/BLS/WEF)
                         │
               Data Collection Layer
                         │
                Data Warehouse (SQL/Parquet)
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
 Skill Extraction   Salary Model     Trend Forecasting
      │                  │                  │
      └──────────────────┼──────────────────┘
                         │
                Skill Knowledge Graph
                         │
          Career Intelligence Engine
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
 Resume Analyzer   AI Career Coach   Executive Dashboards
```

## Implemented Features (12/12)

| # | Feature | Status | File(s) |
|---|---------|--------|---------|
| 1 | **Skill Knowledge Graph** | ✅ 10+ tests | `src/skill_graph/skill_graph.py` |
| 2 | **Personalized Career Roadmaps** | ✅ Upgrade with timeline | `src/skill_gap/roadmap_generator.py` |
| 3 | **Resume ↔ Job Matching** | ✅ Role matcher + batch | `src/resume_analyzer/matching_engine.py` |
| 4 | **Real-Time Trend Tracking** | ✅ Weekly sim + quarterly | `src/data_collection/realtime_tracker.py` |
| 5 | **Explainable AI (SHAP)** | ✅ Wired in dashboard | `src/salary_prediction/explainable_predictor.py` |
| 6 | **AI Career Coach** | ✅ Enriched RAG | `src/chatbot/rag_chatbot.py` |
| 7 | **Forecasting** | ✅ Prophet/XGBoost/LSTM | `src/forecasting/` |
| 8 | **Geographic Intelligence** | ✅ Interactive maps | `src/geographic/geo_intelligence.py` |
| 9 | **MLOps Pipeline** | ✅ MLflow/Docker/CI-CD | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci_cd.yml` |
| 10 | **Executive Dashboards** | ✅ Job Seeker/Recruiter/Exec | `app.py` |
| 11 | **Explainable Salary Prediction** | ✅ SHAP in dashboard | `app.py` (salary_insights_page tab4) |
| 12 | **Career Path Analysis** | ✅ Transitions + projections | `src/skill_graph/career_paths.py` |

## Test Coverage
Total tests: 108 existing + 26 new (skill_graph, geographic, realtime_tracker) = 134 tests across 12+ modules.

## Key Decisions
- **Skill Graph**: Pre-defined 35+ skill relationships across 10 categories with BFS pathfinding and Jaccard similarity
- **Career Paths**: 10 roles with transition maps, salary bands, and 10-year projections
- **Real-Time Tracker**: JSON-backed simulated weekly updates with quarterly reporting
- **SHAP Integration**: TreeExplainer for CatBoost with top-8 factor display in dashboard
- **Geographic Intelligence**: Plotly Scattergeo maps for US hiring/salary hotspots + regional breakdowns
- **Executive Views**: Role-based segmented control switching Job Seeker ↔ Recruiter ↔ Executive Summary
- **All models** saved as pickle files; no cloud storage dependency
- **Dashboard** caches market data + skill timeseries for performance

## Common Issues
- `faiss` not installed → `pip install faiss-cpu`
- `tensorflow` not installed → LSTM forecasting skipped (Prophet/XGBoost still work)
- PostgreSQL unavailable → system auto-falls back to generated data
- Test failures → ensure `nltk.download('punkt_tab')` and `python -m spacy download en_core_web_lg` are run
- Dashboard `segmented_control` requires Streamlit ≥1.55.0
