import pandas as pd
from typing import Optional, List, Dict
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None

    def connect(self) -> None:
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def disconnect(self) -> None:
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        return pd.read_sql_query(text(query), self.engine, params=params)

    def execute_statement(self, statement: str, params: Optional[Dict] = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(statement), params or {})

    def to_sql(self, df: pd.DataFrame, table_name: str, if_exists: str = "append",
               schema: str = "public") -> None:
        df.to_sql(
            table_name, self.engine, schema=schema,
            if_exists=if_exists, index=False, method="multi",
            chunksize=1000
        )
        logger.info(f"Inserted {len(df)} rows into {schema}.{table_name}")

    def table_exists(self, table_name: str, schema: str = "public") -> bool:
        insp = inspect(self.engine)
        return insp.has_table(table_name, schema=schema)

    def list_tables(self, schema: str = "public") -> List[str]:
        insp = inspect(self.engine)
        return insp.get_table_names(schema=schema)


class SkillLensDB(DatabaseManager):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)

    def init_schema(self) -> None:
        schema_sql = """
        CREATE SCHEMA IF NOT EXISTS skilllens;
        """
        self.execute_statement(schema_sql)

    def create_tables(self) -> None:
        tables_sql = """
        CREATE TABLE IF NOT EXISTS skilllens.job_postings (
            id SERIAL PRIMARY KEY,
            job_id VARCHAR(255) UNIQUE,
            title VARCHAR(500),
            company VARCHAR(500),
            location VARCHAR(500),
            description TEXT,
            skills TEXT[],
            salary_min FLOAT,
            salary_max FLOAT,
            salary_currency VARCHAR(10),
            experience_required FLOAT,
            education_required VARCHAR(255),
            industry VARCHAR(255),
            employment_type VARCHAR(100),
            posted_date DATE,
            source VARCHAR(100),
            url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skilllens.skills_taxonomy (
            id SERIAL PRIMARY KEY,
            skill_name VARCHAR(255) UNIQUE,
            category VARCHAR(255),
            subcategory VARCHAR(255),
            description TEXT,
            skill_type VARCHAR(100),
            onet_code VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skilllens.job_skills (
            id SERIAL PRIMARY KEY,
            job_id VARCHAR(255) REFERENCES skilllens.job_postings(job_id),
            skill_name VARCHAR(255) REFERENCES skilllens.skills_taxonomy(skill_name),
            importance_score FLOAT,
            is_required BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skilllens.salary_data (
            id SERIAL PRIMARY KEY,
            job_id VARCHAR(255) REFERENCES skilllens.job_postings(job_id),
            salary FLOAT,
            currency VARCHAR(10),
            experience_level VARCHAR(100),
            education_level VARCHAR(100),
            location VARCHAR(500),
            job_title VARCHAR(500),
            industry VARCHAR(255),
            company_size VARCHAR(50),
            employment_type VARCHAR(100),
            year INT,
            source VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skilllens.skill_demand (
            id SERIAL PRIMARY KEY,
            skill_name VARCHAR(255) REFERENCES skilllens.skills_taxonomy(skill_name),
            year INT,
            month INT,
            demand_count INT,
            demand_score FLOAT,
            growth_rate FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS skilllens.resume_analyses (
            id SERIAL PRIMARY KEY,
            candidate_name VARCHAR(500),
            extracted_skills TEXT[],
            ats_score FLOAT,
            matched_skills TEXT[],
            missing_skills TEXT[],
            recommendations TEXT,
            resume_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        statements = [s.strip() for s in tables_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                self.execute_statement(stmt + ";")
            except SQLAlchemyError as e:
                logger.warning(f"Table creation stmt warning: {e}")
        logger.info("Database tables initialized")

    def insert_job_posting(self, data: Dict) -> None:
        query = """
        INSERT INTO skilllens.job_postings
        (job_id, title, company, location, description, skills, salary_min, salary_max,
         salary_currency, experience_required, education_required, industry,
         employment_type, posted_date, source, url)
        VALUES (:job_id, :title, :company, :location, :description, :skills,
                :salary_min, :salary_max, :salary_currency, :experience_required,
                :education_required, :industry, :employment_type, :posted_date,
                :source, :url)
        ON CONFLICT (job_id) DO NOTHING
        """
        self.execute_statement(query, data)

    def get_skill_trends(self, skill_name: Optional[str] = None) -> pd.DataFrame:
        query = """
        SELECT skill_name, year, month, demand_count, demand_score, growth_rate
        FROM skilllens.skill_demand
        """
        if skill_name:
            query += " WHERE skill_name = :skill_name"
            return self.execute_query(query, {"skill_name": skill_name})
        return self.execute_query(query)

    def get_top_skills(self, limit: int = 20) -> pd.DataFrame:
        query = """
        SELECT skill_name, COUNT(*) as demand_count
        FROM skilllens.job_skills
        GROUP BY skill_name
        ORDER BY demand_count DESC
        LIMIT :limit
        """
        return self.execute_query(query, {"limit": limit})

    def get_salary_stats(self) -> pd.DataFrame:
        query = """
        SELECT job_title, location, experience_level, education_level,
               AVG(salary) as avg_salary, MIN(salary) as min_salary,
               MAX(salary) as max_salary, COUNT(*) as count
        FROM skilllens.salary_data
        GROUP BY job_title, location, experience_level, education_level
        """
        return self.execute_query(query)

    def search_jobs(self, keyword: str, location: Optional[str] = None,
                    limit: int = 50) -> pd.DataFrame:
        query = """
        SELECT title, company, location, salary_min, salary_max,
               skills, experience_required, industry, posted_date, source
        FROM skilllens.job_postings
        WHERE title ILIKE :keyword OR description ILIKE :keyword
        """
        params = {"keyword": f"%{keyword}%"}
        if location:
            query += " AND location ILIKE :location"
            params["location"] = f"%{location}%"
        query += " LIMIT :limit"
        params["limit"] = limit
        return self.execute_query(query, params)
