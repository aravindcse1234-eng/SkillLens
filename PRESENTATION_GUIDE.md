# SkillLens — Full Presentation Guide

## Slide 1: Title Slide

**Title:** SkillLens: Career Intelligence & Workforce Analytics Platform
**Subtitle:** An AI-Powered Platform for Skill Demand Forecasting, Salary Prediction, Resume Matching, and Workforce Analytics
**Presenter:** [Your Name]
**Date:** [Date]
**Tagline:** Transforming Workforce Decisions Through AI Intelligence

---

## Slide 2: The Problem

**Headline:** The Workforce Intelligence Gap

**Bullet Points:**
- Organizations struggle to predict skill demand in rapidly changing markets
- Salary benchmarking is fragmented across multiple sources (BLS, Glassdoor, Levels.fyi)
- Resume screening is manual, biased, and inconsistent
- Career path planning lacks data-driven guidance
- Geographic talent distribution is poorly understood
- No unified platform connects all these dots

**Key Stat:** 87% of organizations face skill gaps, and 74% of recruiters say finding qualified candidates is their biggest challenge (Source: WEF, LinkedIn 2024)

---

## Slide 3: The Solution

**Headline:** SkillLens — One Platform, 12 Intelligence Modules

**Description:**
An end-to-end Career Intelligence & Workforce Analytics platform that combines:

- **NLP** for skill extraction and resume parsing
- **Machine Learning** (CatBoost R²=0.9947) for salary prediction
- **Time Series Forecasting** (Prophet, XGBoost) for skill demand trends
- **Explainable AI** (SHAP) for transparent salary decisions
- **Knowledge Graphs** for skill relationships and career paths
- **RAG Chatbots** for AI-powered career advising
- **Geographic Intelligence** for talent location analytics

---

## Slide 4: Architecture Overview

**Title:** System Architecture

**Diagram Description (describe or draw):**

```
                    ┌─────────────────────────────┐
                    │   Data Sources              │
                    │  (Generated/Kaggle/O*NET/   │
                    │   BLS/WEF/Live APIs)        │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Data Collection Layer     │
                    │  (Web Scraper, API Clients, │
                    │   Live Collector)           │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   Data Warehouse            │
                    │  (SQLite/Parquet/CSV)       │
                    └──────────────┬──────────────┘
                                   ▼
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌─────────────────┐     ┌────────────────────┐     ┌──────────────────┐
│  Skill          │     │  Salary Prediction │     │  Trend           │
│  Extraction     │     │  (CatBoost R²=0.99)│     │  Forecasting     │
│  (NLP/Spacy)    │     │  + SHAP Explain    │     │  (Prophet/XGBoost)│
└────────┬────────┘     └─────────┬──────────┘     └────────┬─────────┘
         ▼                        ▼                         ▼
         └────────────────────────┼─────────────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   Skill Knowledge Graph     │
                    │  (35+ skills, 10 categories,│
                    │   BFS Pathfinding)          │
                    └──────────────┬──────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   Career Intelligence Engine │
                    │  (Resume Matcher, RAG,      │
                    │   Career Twin, Geo Intel)   │
                    └──────────────┬──────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   Streamlit Dashboard       │
                    │  (12 pages, Plotly charts)  │
                    └─────────────────────────────┘
```

**Key Technologies:**
- Python 3.14.3, Streamlit 1.55.0, FastAPI
- CatBoost, XGBoost, LightGBM, scikit-learn
- Prophet (Meta), spaCy, NLTK
- Plotly, SHAP, NetworkX
- FAISS, Sentence-Transformers

---

## Slide 5: Data Pipeline

**Title:** How Data Flows Through SkillLens

**Step 1 — Data Generation & Collection:**
- `generate_market_data(n=3000)`: Generates realistic job postings with skills, salaries, locations
- Real data sources: BLS OES (2024), Stack Overflow Survey (2024), Levels.fyi, AmbitionBox India
- Live data: GitHub API (repo counts), Stack Overflow API (tag trends), Hacker News API (tech mentions)
- O*NET database integration for standardized skill taxonomies

**Step 2 — Preprocessing:**
- Column standardization, duplicate removal, missing value handling
- Salary log-transformation for better model performance
- Skill extraction via spaCy NER + custom patterns
- Feature engineering: 36→43 features (one-hot encoding, aggregations)

**Step 3 — Model Training:**
- 6 models trained: LinearRegression, RandomForest, XGBoost, LightGBM, CatBoost, Ensemble
- **Best model: CatBoost — R²=0.9947, RMSE=0.039**
- 5-fold cross-validation with mean R²=0.9943
- Models saved as pickle files for inference

**Step 4 — Forecasting:**
- Prophet (Facebook/Meta) for trend decomposition
- XGBoost regressor for feature-driven forecasting
- LSTM (when TensorFlow available) for deep learning
- 95% confidence intervals on all forecasts

**Step 5 — Dashboard Rendering:**
- `@st.cache_data(ttl=3600)` caches data for 1 hour
- `@st.cache_resource` caches ML models permanently
- Version-based cache busting (`v3.1-india`)

---

## Slide 6: Dashboard — 12 Pages Overview

**Title:** 12 Intelligence Modules

| # | Page | Purpose | Key Feature |
|---|------|---------|-------------|
| 1 | **Dashboard** | Executive KPIs | Market pulse, 12 metrics |
| 2 | **Job Market** | Supply/demand | Location heatmap, remote % |
| 3 | **Trending Skills** | Skill trends | 6/12/24mo time series |
| 4 | **Salary Insights** | Compensation | ML predict + real salaries |
| 5 | **Forecasting** | Demand forecast | Prophet 95% CI |
| 6 | **Resume Analyzer** | ATS scoring | 6-dim Candidate DNA |
| 7 | **Skill Gap** | Gap analysis | Personalized roadmap |
| 8 | **Skill Graph** | Knowledge graph | BFS path, Jaccard sim |
| 9 | **Geographic** | Talent maps | Plotly Scattergeo |
| 10 | **Real-Time Trends** | Live pulse | Weekly simulations |
| 11 | **Executive View** | Role-based views | 3 segmented views |
| 12 | **AI Assistant** | RAG chatbot | Career knowledge base |

---

## Slide 7: Deep Dive — Salary Prediction Module

**Title:** Salary Intelligence Engine

**How it works:**
1. User selects: Role, Experience (years), Location (city), Education
2. Features are engineered (36→43 features)
3. CatBoost model predicts log-transformed salary
4. Prediction is exponentiated back to dollar/rupee value
5. SHAP explains top 8 factors driving the prediction

**Performance Metrics:**
```
CatBoost:         R²=0.9947 | RMSE=0.039 | MAPE=26.7%
XGBoost:          R²=0.9930 | RMSE=0.045 | MAPE=28.7%
LightGBM:         R²=0.9932 | RMSE=0.044 | MAPE=28.8%
RandomForest:     R²=0.9906 | RMSE=0.052 | MAPE=30.1%
LinearRegression: R²=0.9615 | RMSE=0.105 | MAPE=70.1%
Ensemble:         R²=0.9940 | RMSE=0.041 | MAPE=26.9%
```

**Two Salary Sources Shown Side-by-Side:**
1. **ML Predicted** — CatBoost model prediction with confidence range
2. **Real Market** — Embedded BLS/Stack Overflow/Levels.fyi data with city adjustment

**SHAP Explainability:**
- Top 8 features shown in waterfall plot
- "Why did the model predict this salary?" answered
- Factors: years_experience, education, location, industry, skills count

**Live Demo:**
> "Let's predict salary for a Senior Data Scientist in Bangalore with 5 years experience and a Master's degree"

---

## Slide 8: Deep Dive — Resume Analyzer

**Title:** Candidate Intelligence Center

**Pipeline:**
1. Upload Resume (PDF/DOCX/TXT) or paste text
2. spaCy NER extracts skills, education, experience
3. ATS Scorer evaluates against target role:
   - Skill Match % (Jaccard similarity with role requirements)
   - Experience Relevance
   - Education Fit
4. Matching Engine ranks candidates for batch processing

**Candidate DNA™ Profile (6 Dimensions):**
| Dimension | What It Measures |
|-----------|-----------------|
| Technical Capability | Skill depth + breadth |
| Learning Velocity | Rate of skill acquisition |
| Market Competitiveness | How in-demand their skills are |
| Role Readiness | How prepared for target role |
| Leadership Potential | Management/mentorship signals |
| Career Stability | Tenure patterns |

**Batch ATS Ranking:**
- Upload 5+ resumes
- System scores and ranks all candidates
- KMeans clustering groups candidates by skill profile

---

## Slide 9: Deep Dive — Skill Graph & Career Paths

**Title:** Talent Ecosystem Map

**Skill Graph Structure:**
- 35+ skills across 10 categories (ML, Cloud, Data, Backend, Frontend, Mobile, DevOps, Database, Security, Soft Skills)
- Relationships: prerequisite, complementary, adjacent
- Pathfinding: BFS (Breadth-First Search) for shortest skill path
- Similarity: Jaccard coefficient for skill overlap

**Career Path Analysis:**
- 10 roles with transition maps (e.g., Junior Dev → Senior → Lead → Architect)
- Salary bands for each role (entry, mid, senior)
- 10-year salary projections with YoY growth
- Multiple path options (e.g., Data Scientist → ML Engineer OR → Data Engineer)

**Live Demo:**
> "Show me the shortest path from Python Developer to AI Engineer"
> "What skills do I need to transition from Data Analyst to Data Scientist?"

---

## Slide 10: Forecasting Engine

**Title:** Workforce Forecast Engine

**Three Models:**
- **Prophet:** Handles seasonality, trend, holiday effects. Best for long-term trends.
- **XGBoost:** Feature-driven (GDP, population, tech adoption). Best for short-term.
- **LSTM:** Deep learning (when TensorFlow available). Best for complex patterns.

**Example Forecast Output:**
| Skill | Current | 6mo | 12mo | 24mo | Trend |
|-------|---------|-----|------|------|-------|
| AI/ML | 85 | 92 | 105 | 130 | 📈 +53% |
| Cloud | 78 | 83 | 88 | 95 | 📈 +22% |
| Python | 72 | 76 | 80 | 85 | 📈 +18% |
| Blockchain | 45 | 42 | 38 | 30 | 📉 -33% |

**Confidence Intervals:** All forecasts include 95% CI bands
**Alerting:** Skills with >20% projected growth are flagged as "High Demand"

---

## Slide 11: Explainable AI (SHAP)

**Title:** Transparent AI Decisions

**Why SHAP?**
- Recruiters need to trust and explain salary predictions
- Candidates deserve transparency in compensation decisions
- Regulatory compliance (EU AI Act, NYC Local Law 144)

**How It Works:**
1. Train CatBoost model on job market data
2. Apply TreeExplainer (model-agnostic SHAP for tree ensembles)
3. For each prediction, compute feature contributions
4. Display top 8 contributing factors

**Sample SHAP Output:**
```
Feature                  Contribution    Impact
─────────────────────────────────────────────
years_experience (5yr)   +$15,230       📈 Positive
location (Bangalore)     -$5,200        📉 Negative
education (Master's)     +$8,100        📈 Positive
skill_count (8)          +$3,400        📈 Positive
industry (Tech)          +$6,700        📈 Positive
title_encoded            +$2,100        📈 Positive
employment_type (Full)   +$1,800        📈 Positive
company_size (Large)     +$1,200        📈 Positive
─────────────────────────────────────────────
Base Value: $85,000
Final Prediction: $118,330
```

**Dashboard Integration:**
- Salary Insights page tab: "Explainable AI"
- Force plot + waterfall chart
- "Why this salary?" button with one-click explanation

---

## Slide 12: Geographic Intelligence

**Title:** Global Talent Radar

**9 Indian Cities:**
| City | Weight | Avg Salary (₹) | Talent Pool |
|------|--------|----------------|-------------|
| Bangalore | 10% | ₹12,00,000 | Highest |
| Mumbai | 8% | ₹11,50,000 | High |
| Delhi | 8% | ₹10,80,000 | High |
| Hyderabad | 6% | ₹10,20,000 | High |
| Pune | 5% | ₹9,50,000 | Medium |
| Chennai | 4% | ₹9,00,000 | Medium |
| Kolkata | 2% | ₹7,50,000 | Low |
| Ahmedabad | 2% | ₹7,00,000 | Low |
| Jaipur | 1% | ₹6,50,000 | Low |

**US Cities:** 10 major cities with lat/lng coordinates

**Features:**
- Plotly Scattergeo maps for hiring hotspots
- Salary heatmaps by region
- Remote work trends (remote/hybrid/onsite %)
- Country segmented control (India/US/Global)
- Currency switching ($ / ₹)

---

## Slide 13: Real-Time Trends & Live Data

**Title:** Live Market Intelligence

**Live Data Sources (No API Keys Needed):**
- **GitHub API** (60 req/hr): Repository counts by tech topic
- **Stack Overflow API:** Tag question volume trends
- **Hacker News API:** Tech story frequency

**Weekly Simulations:**
- 52 weeks of realistic trend data
- Skill growth rates with positive/negative trends
- "Soaring" and "Declining" skill categories

**Quarterly Reports:**
- JSON-backed quarterly snapshots
- 5 top growing skills
- 5 top declining skills
- Total skills tracked

**Caching:** 1-hour TTL in `data/live_cache/`

---

## Slide 14: RAG AI Assistant

**Title:** HIGH SIGHT AI Advisor

**Architecture:**
```
User Question
    │
    ▼
┌─────────────────────────────┐
│  FAISS Vector Search        │
│  (sentence-transformers)    │
│  Finds relevant career docs │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Context Assembly           │
│  (question + retrieved docs)│
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  LLM Response Generation    │
│  (fallback: rule-based)     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Response + Source Citations │
└─────────────────────────────┘
```

**Knowledge Base:**
- Career transition guides
- Skill development roadmaps
- Salary negotiation tips
- Industry trends and insights
- Resume optimization strategies

**Response Format:**
```
💡 Insight: [Key observation]
⚠️ Risk: [Potential challenge]
🎯 Opportunity: [Actionable next step]
📋 Recommendation: [What to do next]
```

---

## Slide 15: Career Twin & Portfolio Analyzer

**Title:** Digital Workforce Identity

**Career Twin Engine:**
- Creates a digital profile matching user's skills, experience, education
- Suggests optimal career paths based on skill graph
- Projects salary growth over 10 years
- Identifies skill gaps and learning priorities

**Portfolio Analyzer:**
- Accepts GitHub URL + LinkedIn URL
- `fetch_github_repos()`: Fetches real repos via GitHub API
- Skill detection from repo topics and languages
- Analyzes project diversity, tech stack breadth

**Interview Simulator:**
- AI-powered interview practice
- Role-specific questions (technical + behavioral)
- Feedback on answer quality
- Session history tracking

---

## Slide 16: MLOps & Deployment

**Title:** Production-Ready Infrastructure

**Container Stack:**
```yaml
services:
  dashboard:  # Streamlit on port 8501
  api:        # FastAPI on port 8000
  mlflow:     # Model tracking on port 5000
  postgres:   # Metadata store
```

**MLflow Integration:**
- Experiment tracking for all 6 models
- Parameter logging (iterations, learning rate, depth)
- Metric logging (R², RMSE, MAE, MAPE)
- Model registry for versioning

**CI/CD Pipeline:**
```yaml
# .github/workflows/ci_cd.yml
- Test: pytest with 134 tests
- Lint: flake8 + black
- Build: Docker compose
- Deploy: Automatic on merge
```

**Environment Management:**
- `requirements.txt` for core dependencies
- `requirements-full.txt` for complete environment
- `.env.example` for configuration
- Docker multi-stage builds

---

## Slide 17: Test Coverage

**Title:** 134 Tests Across 12 Modules

| Module | Tests | Key Tested Feature |
|--------|-------|-------------------|
| Skill Graph | 10+ | BFS, Jaccard, relationships |
| Geographic | 10+ | Lat/lng, city filtering, maps |
| Realtime Tracker | 10+ | Weekly sim, quarterly reports |
| Data Generator | 10+ | Market data, timeseries |
| Salary Prediction | 10+ | Models, feature engineering |
| Forecasting | 10+ | Prophet, XGBoost, pipeline |
| Resume Analyzer | 10+ | ATS scoring, matching |
| Skill Gap | 10+ | Gap analysis, roadmaps |
| Chatbot | 5+ | RAG, FAISS search |
| NLP | 5+ | Skill extraction, NER |
| API | 5+ | Endpoint responses |
| EDA | 5+ | Statistical analysis |

**Run Tests:** `python -m pytest tests/ -v`

---

## Slide 18: Key Results & Metrics

**Title:** Performance Benchmarks

| Metric | Value |
|--------|-------|
| Salary Prediction R² | **0.9947** (CatBoost) |
| Salary Prediction RMSE | **$0.039** (log space) |
| Best Model | CatBoost |
| Skills in Graph | **35+** across 10 categories |
| Career Roles | **10** with transition maps |
| Indian Cities | **9** with lat/lng |
| Data Records | **3,000** market data points |
| Trend Period | **52 weeks** simulated |
| Test Coverage | **134 tests** across 12 modules |
| Dashboard Pages | **12** |
| API Endpoints | **9** |
| Forecast Models | 3 (Prophet, XGBoost, LSTM) |
| Resume Dimensions | **6** (Candidate DNA Profile) |

---

## Slide 19: Demo Script

**Title:** Live Demo Walkthrough (5 minutes)

**Demo 1: Salary Prediction (1 min)**
1. Navigate to Salary Insights page
2. Select "Senior Data Scientist", Bangalore, 5 years, Master's
3. Show CatBoost predicted salary with SHAP explanation
4. Show real salary from BLS/Levels.fyi for comparison
5. Switch to India → Currency changes to ₹

**Demo 2: Resume Analysis (1 min)**
1. Navigate to Resume Analyzer
2. Upload a sample resume
3. Show ATS score, Candidate DNA, skill breakdown
4. Show role matching results

**Demo 3: Skill Gap (1 min)**
1. Navigate to Skill Gap page
2. Select current role (Python Developer)
3. Select target role (Data Scientist)
4. Show missing skills, matched skills, roadmap timeline

**Demo 4: Forecasting (1 min)**
1. Navigate to Forecasting page
2. Select "AI/ML" skill
3. Show Prophet forecast with 95% CI
4. Toggle between 6mo, 12mo, 24mo views

**Demo 5: AI Assistant (1 min)**
1. Navigate to AI Assistant
2. Ask: "What skills should I learn to become a Data Scientist?"
3. Show structured response with Insight/Risk/Opportunity/Recommendation

---

## Slide 20: Future Work

**Title:** Roadmap & Enhancements

**Short Term:**
- Integrate real LLM (GPT-4 / Claude) for RAG chatbot (current: rule-based fallback)
- Add TensorFlow/Keras LSTM support when Python 3.14 wheel becomes available
- Docker deployment testing in cloud environment
- User authentication (login/registration)
- Database persistence (PostgreSQL instead of Parquet)

**Medium Term:**
- Real-time job board scraping (LinkedIn, Indeed, Naukri)
- Multi-language resume parsing
- Company salary benchmarking (by org size, industry)
- Skill demand heatmaps by metro region
- Export reports (PDF, Excel)

**Long Term:**
- Multi-user SaaS with team workspaces
- API marketplace for third-party integrations
- Mobile app (React Native / Flutter)
- Real-time labor market data feeds
- Automated workforce planning with "what-if" scenario modeling

---

## Appendix: Tech Stack Reference

**Languages:** Python 3.14.3
**Framework:** Streamlit 1.55.0, FastAPI
**ML/DL:** CatBoost 1.2+, XGBoost 2.0+, LightGBM 4.0+, scikit-learn 1.4+
**Forecasting:** Prophet 1.1+, TensorFlow (optional)
**NLP:** spaCy 3.7+ (en_core_web_lg), NLTK, sentence-transformers
**XAI:** SHAP 0.44+
**Visualization:** Plotly 5.20+, Matplotlib, NetworkX
**Vector Search:** FAISS (faiss-cpu)
**Infrastructure:** Docker, Docker Compose, MLflow
**Testing:** pytest 7.4+

---

## Appendix: Key Files to Reference

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit dashboard (~1957 lines, 17 pages) |
| `src/train_pipeline.py` | End-to-end training pipeline |
| `src/salary_prediction/trainer.py` | SalaryTrainer with 6 models |
| `src/salary_prediction/models.py` | Model implementations |
| `src/salary_prediction/explainable_predictor.py` | SHAP explainer |
| `src/skill_graph/skill_graph.py` | Skill knowledge graph |
| `src/skill_graph/career_paths.py` | Career path analysis |
| `src/resume_analyzer/ats_scorer.py` | ATS scoring engine |
| `src/resume_analyzer/matching_engine.py` | Resume-to-role matching |
| `src/forecasting/forecast_pipeline.py` | Prophet/XGBoost pipeline |
| `src/data_collection/live_collector.py` | GitHub/SO/HN live data |
| `src/data_collection/realtime_tracker.py` | Weekly trend simulations |
| `src/geographic/geo_intelligence.py` | Location intelligence |
| `src/chatbot/rag_chatbot.py` | RAG-powered AI assistant |
| `src/data/data_generator.py` | Synthetic market data generation |
| `src/salary_prediction/real_salary_collector.py` | Real salary benchmarks |
| `docker-compose.yml` | Full stack deployment |
| `.github/workflows/ci_cd.yml` | CI/CD pipeline |
