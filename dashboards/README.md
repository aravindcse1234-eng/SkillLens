# Power BI Dashboards

This directory contains Power BI (.pbix) templates and dataset exports for building job market intelligence dashboards.

## Available Dashboards

| Dashboard | Description | Data Source |
|-----------|-------------|-------------|
| `skilllens_job_market.pbix` | Job Market Overview — top skills, salaries by role/location/experience, hiring trends | `data/processed/` pipeline exports |
| `skilllens_forecast.pbix` | Skill Demand Forecast — Prophet/XGBoost forecast charts with trend analysis | `reports/forecast_report_*.json` |
| `skilllens_salary.pbix` | Salary Prediction Explorer — model performance, feature importance, what-if scenarios | `reports/training_report_*.json` |

## Usage

1. Run the pipeline first to generate data:
   ```bash
   python -m src.train_pipeline --mode full
   ```

2. Open the `.pbix` files in Power BI Desktop.

3. Update data source paths to point to your local `data/processed/` and `reports/` directories.

## Exporting Data for Power BI

```python
from src.data.data_generator import generate_market_data
df = generate_market_data(n_samples=10000, seed=42)
df.to_csv("dashboards/export/market_data.csv", index=False)
```
