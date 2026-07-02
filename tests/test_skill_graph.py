import pytest
from src.skill_graph.skill_graph import SkillGraph


class TestSkillGraph:
    def test_init(self):
        sg = SkillGraph()
        assert len(sg.relationships) > 10

    def test_get_related_skills(self):
        sg = SkillGraph()
        related = sg.get_related_skills("Python")
        assert "Pandas" in related
        assert "NumPy" in related

    def test_get_skill_category(self):
        sg = SkillGraph()
        assert sg.get_skill_category("Python") == "Programming"
        assert sg.get_skill_category("Unknown") is None

    def test_get_skills_by_category(self):
        sg = SkillGraph()
        ml_skills = sg.get_skills_by_category("AI/ML")
        assert len(ml_skills) > 5
        assert "Machine Learning" in ml_skills

    def test_get_path_exists(self):
        sg = SkillGraph()
        path = sg.get_path("Python", "Kubernetes")
        assert len(path) >= 2
        assert path[0] == "Python"
        assert path[-1] == "Kubernetes"

    def test_get_path_same_skill(self):
        sg = SkillGraph()
        path = sg.get_path("Python", "Python")
        assert path == ["Python"]

    def test_recommend_skills(self):
        sg = SkillGraph()
        recs = sg.recommend_skills(["Python", "Machine Learning"], top_k=3)
        assert len(recs) <= 3
        for r in recs:
            assert "skill" in r
            assert "score" in r
            assert r["skill"] not in ["Python", "Machine Learning"]

    def test_get_network_data(self):
        sg = SkillGraph()
        data = sg.get_network_data(center_skill="Python")
        assert "nodes" in data
        assert "edges" in data
        assert any(n["id"] == "Python" for n in data["nodes"])

    def test_compute_similarity(self):
        sg = SkillGraph()
        sim = sg.compute_similarity("TensorFlow", "PyTorch")
        assert sim > 0

    def test_get_role_skills(self):
        sg = SkillGraph()
        skills = sg.get_role_skills("Data Scientist")
        assert "Python" in skills
        assert "Machine Learning" in skills

    def test_get_all_roles(self):
        sg = SkillGraph()
        roles = sg.get_all_roles()
        assert "Data Scientist" in roles

    def test_get_prerequisite_chain(self):
        sg = SkillGraph()
        chain = sg.get_prerequisite_chain("Deep Learning")
        assert "Python" in chain
        assert "Machine Learning" in chain
        assert "Deep Learning" in chain


class TestCareerPathAnalyzer:
    def test_career_paths(self):
        from src.skill_graph.career_paths import CareerPathAnalyzer
        cpa = CareerPathAnalyzer()
        paths = cpa.get_career_paths("Data Scientist")
        assert paths["current_role"] == "Data Scientist"
        assert len(paths["next_roles"]) > 0
        assert len(paths["promotion_roles"]) > 0

    def test_get_salary_band(self):
        from src.skill_graph.career_paths import CareerPathAnalyzer
        cpa = CareerPathAnalyzer()
        band = cpa.get_salary_band("Data Scientist", "mid")
        assert band is not None
        assert len(band) == 2
        assert band[0] < band[1]

    def test_get_projection(self):
        from src.skill_graph.career_paths import CareerPathAnalyzer
        cpa = CareerPathAnalyzer()
        proj = cpa.get_projection("Data Scientist", 120000, 3)
        assert len(proj) > 0
        assert proj[0]["estimated_salary"] > 0
