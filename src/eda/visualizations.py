import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SkillLensVisualizer:
    def __init__(self, template: str = "plotly_dark"):
        self.template = template
        self.colors = px.colors.sequential.Viridis

    def top_skills_bar(self, skill_counts: Dict[str, int], top_n: int = 20) -> go.Figure:
        items = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        skills, counts = zip(*items) if items else ([], [])
        fig = px.bar(
            x=list(counts), y=list(skills), orientation="h",
            title="Top Skills in Demand",
            labels={"x": "Demand Count", "y": "Skill"},
            color=list(counts), color_continuous_scale="viridis",
            template=self.template
        )
        fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
        return fig

    def salary_distribution_hist(self, salary_data: pd.Series) -> go.Figure:
        fig = px.histogram(
            salary_data, nbins=50, title="Salary Distribution",
            labels={"value": "Salary (USD)", "count": "Frequency"},
            color_discrete_sequence=["#00CC96"],
            template=self.template
        )
        fig.add_vline(x=salary_data.median(), line_dash="dash", line_color="red",
                      annotation_text=f"Median: ${salary_data.median():,.0f}")
        fig.add_vline(x=salary_data.mean(), line_dash="dot", line_color="yellow",
                      annotation_text=f"Mean: ${salary_data.mean():,.0f}")
        return fig

    def salary_by_experience_box(self, df: pd.DataFrame, salary_col: str = "salary",
                                  exp_col: str = "experience_level") -> go.Figure:
        fig = px.box(
            df, x=exp_col, y=salary_col, title="Salary by Experience Level",
            color=exp_col, template=self.template,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        return fig

    def location_job_count_map(self, location_counts: pd.DataFrame) -> go.Figure:
        fig = px.choropleth(
            location_counts,
            locations="location",
            locationmode="country names",
            color="count",
            title="Job Distribution by Location",
            color_continuous_scale="viridis",
            template=self.template
        )
        return fig

    def location_job_count_bar(self, location_counts: pd.DataFrame) -> go.Figure:
        fig = px.bar(
            location_counts.head(15),
            x="count", y="location", orientation="h",
            title="Top Hiring Locations",
            color="count", color_continuous_scale="viridis",
            template=self.template
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig

    def skill_trend_line(self, trend_data: pd.DataFrame) -> go.Figure:
        fig = px.line(
            trend_data, x="date", y="demand_score", color="skill_name",
            title="Skill Demand Trends Over Time",
            markers=True, template=self.template
        )
        return fig

    def industry_pie(self, industry_counts: pd.DataFrame) -> go.Figure:
        fig = px.pie(
            industry_counts.head(10), values="count", names="industry",
            title="Industry-wise Hiring Distribution",
            template=self.template, hole=0.4
        )
        return fig

    def experience_level_pie(self, exp_counts: pd.DataFrame) -> go.Figure:
        fig = px.pie(
            exp_counts, values="count", names="level",
            title="Experience Level Distribution",
            template=self.template, hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        return fig

    def correlation_heatmap(self, corr_df: pd.DataFrame) -> go.Figure:
        fig = px.imshow(
            corr_df, text_auto=".2f", aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r", template=self.template,
            width=800, height=700
        )
        return fig

    def education_requirement_bar(self, edu_counts: pd.DataFrame) -> go.Figure:
        fig = px.bar(
            edu_counts, x="education", y="count",
            title="Education Requirements Distribution",
            color="count", color_continuous_scale="viridis",
            template=self.template
        )
        return fig

    def salary_trend_over_time(self, df: pd.DataFrame, date_col: str = "posted_date",
                                salary_col: str = "salary") -> go.Figure:
        fig = px.scatter(
            df, x=date_col, y=salary_col, title="Salary Trends Over Time",
            trendline="lowess", template=self.template,
            color_discrete_sequence=["#00CC96"]
        )
        return fig

    def skill_wordcloud(self, skill_counts: Dict[str, int]) -> go.Figure:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(width=800, height=400, background_color="black",
                       colormap="viridis", max_words=100)
        wc.generate_from_frequencies(skill_counts)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig

    def create_dashboard_grid(self, figures: List[go.Figure], titles: List[str],
                               cols: int = 2) -> go.Figure:
        rows = (len(figures) + cols - 1) // cols
        specs = [[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, specs=specs)
        for i, f in enumerate(figures):
            row = i // cols + 1
            col = i % cols + 1
            for trace in f.data:
                fig.add_trace(trace, row=row, col=col)
        fig.update_layout(height=400 * rows, template=self.template, showlegend=False)
        return fig

    def skill_gap_radar(self, user_skills: List[str], market_skills: Dict[str, float]) -> go.Figure:
        categories = list(market_skills.keys())
        user_values = [100 if s in user_skills else 0 for s in categories]
        market_values = [market_skills[s] for s in categories]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=user_values, theta=categories, fill="toself",
                                       name="Your Skills", line_color="#00CC96"))
        fig.add_trace(go.Scatterpolar(r=market_values, theta=categories, fill="toself",
                                       name="Market Demand", line_color="#FF4B4B"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          title="Skill Gap Analysis", template=self.template)
        return fig

    def forecast_plot(self, dates, actual, forecast, upper=None, lower=None,
                       title="Forecast") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=actual, mode="markers",
                                  name="Actual", marker=dict(color="#00CC96")))
        fig.add_trace(go.Scatter(x=dates, y=forecast, mode="lines",
                                  name="Forecast", line=dict(color="#FF4B4B")))
        if upper is not None and lower is not None:
            fig.add_trace(go.Scatter(x=dates, y=upper, mode="lines",
                                      line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=dates, y=lower, mode="lines",
                                      fill="tonexty", fillcolor="rgba(255,75,75,0.2)",
                                      line=dict(width=0), name="Confidence Interval"))
        fig.update_layout(title=title, template=self.template)
        return fig
