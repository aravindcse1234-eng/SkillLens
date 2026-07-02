import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "skilllens")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    schema: str = "public"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def sqlalchemy_url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    n_folds: int = 5
    n_trials: int = 100
    early_stopping_rounds: int = 50
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    max_seq_length: int = 512
    embedding_dim: int = 768


@dataclass
class NLPConfig:
    # NER Models
    bert_model: str = "bert-base-uncased"
    distilbert_model: str = "distilbert-base-uncased"
    roberta_model: str = "roberta-base"
    
    # Sentence Embeddings
    sentence_transformer: str = "all-MiniLM-L6-v2"
    
    # spaCy Model
    spacy_model: str = "en_core_web_lg"
    
    # Skill Extraction
    skill_keywords_path: str = "data/external/skill_keywords.json"
    max_skills_per_job: int = 20
    confidence_threshold: float = 0.7


@dataclass
class ForecastingConfig:
    forecast_horizon: int = 12  # months
    seasonality_mode: str = "multiplicative"
    changepoint_prior_scale: float = 0.05
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    lstm_dropout: float = 0.2
    xgboost_params: Dict = field(default_factory=lambda: {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })


@dataclass
class SalaryConfig:
    target_encodings: Dict = field(default_factory=lambda: {
        "education": {"PhD": 5, "Master": 4, "Bachelor": 3, "Diploma": 2, "High School": 1},
        "experience_level": {"Executive": 4, "Senior": 3, "Mid": 2, "Entry": 1}
    })
    categorical_features: List[str] = field(default_factory=lambda: [
        "job_title", "location", "industry", "company_size", "employment_type"
    ])
    numerical_features: List[str] = field(default_factory=lambda: [
        "years_experience", "num_skills", "skill_score"
    ])


@dataclass
class ResumeConfig:
    max_resume_length: int = 2048
    ats_weights: Dict = field(default_factory=lambda: {
        "keyword_match": 0.4,
        "skills_match": 0.3,
        "experience_match": 0.2,
        "education_match": 0.1
    })
    similarity_threshold: float = 0.75


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 512
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/gemma-2b-it"


@dataclass
class PathConfig:
    root: Path = Path(__file__).parent.parent
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    data_external: Path = root / "data" / "external"
    models: Path = root / "models"
    notebooks: Path = root / "notebooks"
    reports: Path = root / "reports"
    docs: Path = root / "docs"
    config: Path = root / "config"


@dataclass
class AppConfig:
    app_name: str = "SkillLens AI"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")
    use_gpu: bool = os.getenv("USE_GPU", "true").lower() == "true"
    
    # Sub-configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    salary: SalaryConfig = field(default_factory=SalaryConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    paths: PathConfig = field(default_factory=PathConfig)


CONFIG = AppConfig()
