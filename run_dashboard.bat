@echo off
cd /d "%~dp0"
echo Starting SkillLens Dashboard...
echo Open http://localhost:8501 in your browser
streamlit run app.py
pause