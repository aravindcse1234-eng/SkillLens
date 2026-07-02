import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestRootAndHealth:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["app"] == "SkillLens"

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


class TestSkillsEndpoints:
    def test_extract_skills(self):
        resp = client.post("/api/v1/skills/extract", data={"text": "I know Python, SQL and Machine Learning"})
        assert resp.status_code == 200
        data = resp.json()
        assert "skills" in data
        assert "count" in data
        assert data["count"] >= 3

    def test_extract_skills_empty(self):
        resp = client.post("/api/v1/skills/extract", data={"text": " "})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_skill_trends(self):
        resp = client.post("/api/v1/skills/trends", params={"skill_name": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill"] == "Python"
        assert "trend" in data
        assert "growth_rate" in data


class TestSalaryEndpoint:
    def test_predict_salary(self):
        resp = client.post("/api/v1/salary/predict", json={
            "job_title": "Data Scientist",
            "years_experience": 5,
            "location": "San Francisco",
            "education": "Master",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_salary" in data
        assert data["predicted_salary"] > 0
        assert "confidence_interval" in data

    def test_predict_salary_unknown_role(self):
        resp = client.post("/api/v1/salary/predict", json={
            "job_title": "Unknown",
            "years_experience": 2,
            "location": "Remote",
            "education": "Bachelor",
        })
        assert resp.status_code == 200
        assert resp.json()["predicted_salary"] > 0


class TestResumeEndpoint:
    def test_analyze_resume_no_file(self):
        resp = client.post("/api/v1/resume/analyze")
        assert resp.status_code == 422


class TestChatEndpoint:
    def test_chat_returns_response(self):
        resp = client.post("/api/v1/chat", json={
            "message": "What skills are in demand?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "confidence" in data


class TestForecastEndpoint:
    def test_forecast_demand(self):
        resp = client.post("/api/v1/forecast", json={
            "skill_name": "Python",
            "periods": 6,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill"] == "Python"
        assert len(data["forecast"]) == 6


class TestSkillGapEndpoint:
    def test_skill_gap_analysis(self):
        resp = client.post("/api/v1/skill-gap", json={
            "skills": ["Python", "SQL", "Machine Learning"],
            "target_role": "Data Scientist",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_skill_gap_no_target(self):
        resp = client.post("/api/v1/skill-gap", json={
            "skills": ["Python"],
        })
        assert resp.status_code == 200
