import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.helpers import set_seed, get_device, ensure_dir, Timer
from src.utils.logger import setup_logger
from src.data_collection.collector_pipeline import DataCollectionPipeline
from src.data_cleaning.preprocessing_pipeline import PreprocessingPipeline
from src.eda.analyzer import EDAnalyzer
from src.salary_prediction.trainer import SalaryTrainer
from src.forecasting.forecast_pipeline import ForecastPipeline
from src.reports.report_generator import ReportGenerator
from src.data.data_generator import generate_market_data, generate_skill_timeseries

logger = setup_logger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SkillLens Training Pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["quick", "full", "data", "train", "demo"],
                        help="Pipeline execution mode")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    return parser.parse_args()


def _get_data(mode: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    size_map = {"quick": 2000, "train": 5000, "full": 10000, "demo": 5000}
    n = sample_size or size_map.get(mode, 2000)
    return generate_market_data(n_samples=n)


def run_data_pipeline(args):
    logger.info("Running data collection pipeline...")
    pipeline = DataCollectionPipeline()
    results = pipeline.run_all()
    df = results.get("combined", pd.DataFrame())
    if df.empty:
        logger.warning("No real data collected. Generating realistic market data...")
        df = _get_data(args.mode, args.sample_size)
    return df


def run_training_pipeline(df: pd.DataFrame, args):
    logger.info("Running training pipeline...")
    reports = ReportGenerator()
    demo_mode = getattr(args, "demo", False) or args.mode == "demo"

    if args.mlflow and _MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("SkillLensAI_Training")
        mlflow.start_run(run_name=f"train_{args.mode}_{pd.Timestamp.now():%Y%m%d_%H%M}")
        mlflow.log_params({"mode": args.mode, "n_samples": len(df)})

    with Timer():
        logger.info("Preprocessing data...")
        prep = PreprocessingPipeline()
        df_clean = prep.run(df, "job_postings")
        report = prep.get_report()
        logger.info(f"Preprocessing report: {report}")
        if args.mlflow and _MLFLOW_AVAILABLE:
            mlflow.log_metrics({"rows_after": report["pipeline"]["rows_after"]})

        logger.info("Running EDA...")
        eda = EDAnalyzer(df_clean)
        insights = eda.comprehensive_analysis()
        logger.info(f"EDA complete: {len(insights)} insight categories")
        reports.generate_eda_report(insights)

        logger.info("Training salary prediction models...")
        trainer = SalaryTrainer()
        results_df = trainer.train_all(df_clean)
        logger.info(f"Salary model results:\n{results_df}")

        reports.generate_training_report(
            results_df,
            {"total_samples": len(df), "features": list(df_clean.columns)},
            best_model=trainer.best_model.name if trainer.best_model else "none",
        )

        if args.mlflow and _MLFLOW_AVAILABLE and not results_df.empty:
            best = results_df.iloc[0]
            mlflow.log_metrics({
                "best_r2": best["test_r2"],
                "best_rmse": best["test_rmse"],
                "best_mae": best["test_mae"],
            })
            mlflow.log_params({"best_model": best["model"]})

        best_model = trainer.best_model
        if best_model:
            model_dir = ensure_dir(f"{args.model_dir}/salary")
            trainer.save_all_models(str(model_dir))
            logger.info(f"Best model: {trainer.best_model.name}")

        if demo_mode:
            logger.info("Training forecast models on skill time-series...")
            ts_df = generate_skill_timeseries(periods=48)
            ts_df.to_csv("data/processed/skill_timeseries.csv", index=False)
            forecaster = ForecastPipeline()
            forecast_results = forecaster.run(ts_df, top_n=5)
            if forecast_results:
                reports.generate_forecast_report(forecast_results)
                logger.info(f"Forecasts generated for {len(forecast_results)} skills")

    if args.mlflow and _MLFLOW_AVAILABLE:
        mlflow.end_run()


def main():
    args = parse_args()
    set_seed(42)
    logger.info(f"SkillLens Pipeline - Mode: {args.mode}")
    logger.info(f"Device: {get_device()}")

    args.demo = args.mode == "demo"
    mode_handlers = {
        "data": lambda: run_data_pipeline(args),
        "train": lambda: run_training_pipeline(_get_data("train", args.sample_size), args),
        "quick": lambda: run_training_pipeline(_get_data("quick", args.sample_size), args),
        "full": lambda: run_training_pipeline(run_data_pipeline(args), args),
        "demo": lambda: run_training_pipeline(_get_data("demo", args.sample_size or 5000), args),
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        handler()
    else:
        logger.error(f"Unknown mode: {args.mode}")

    logger.info("Pipeline execution complete")


if __name__ == "__main__":
    main()
