import pytest
import tempfile
import os
from src.data_collection.realtime_tracker import JobTrendTracker


class TestJobTrendTracker:
    @pytest.fixture
    def tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield JobTrendTracker(data_dir=tmpdir)

    def test_init(self, tracker):
        assert len(tracker.current_trends) > 0

    def test_get_current_trends(self, tracker):
        trends = tracker.get_current_trends()
        assert "Generative AI" in trends
        assert trends["Generative AI"]["growth"] > 0

    def test_get_trending_skills(self, tracker):
        trending = tracker.get_trending_skills(min_growth=10, top_k=5)
        assert len(trending) <= 5
        for t in trending:
            assert t["growth"] >= 10

    def test_get_declining_skills(self, tracker):
        declining = tracker.get_declining_skills(top_k=3)
        assert len(declining) >= 0

    def test_get_skill_trend(self, tracker):
        result = tracker.get_skill_trend_since("Python", weeks=4)
        assert result["skill"] == "Python"
        assert "current_demand" in result

    def test_simulate_weekly_update(self, tracker):
        result = tracker.simulate_weekly_update()
        assert "date" in result
        assert "changes" in result

    def test_get_quarterly_report(self, tracker):
        report = tracker.get_quarterly_report()
        assert "summary" in report
        assert len(report["fastest_growing"]) > 0

    def test_search_trends(self, tracker):
        results = tracker.search_trends("python")
        assert len(results) >= 1

    def test_save_and_load(self, tracker):
        initial_count = len(tracker.current_trends)
        tracker2 = JobTrendTracker(data_dir=str(tracker.data_dir))
        assert len(tracker2.current_trends) == initial_count
