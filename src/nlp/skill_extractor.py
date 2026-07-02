import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    pipeline, TrainingArguments, Trainer
)
from src.utils.helpers import get_device, save_json
from src.utils.metrics import compute_ner_metrics
from src.utils.logger import setup_logger
import re

logger = setup_logger(__name__)


class SkillExtractor:
    SKILL_PATTERNS = {
        "programming": r"\b(python|java|javascript|typescript|r\b|scala|ruby|go\b|rust|c\+\+|c#|swift|kotlin)\b",
        "ml_dl": r"\b(machine learning|deep learning|neural network|natural language processing|computer vision|nlp|cv|reinforcement learning|generative ai|genai|llm)\b",
        "frameworks": r"\b(tensorflow|pytorch|keras|scikit-learn|sklearn|xgboost|lightgbm|catboost|hugging face|transformers|langchain|llama|openai)\b",
        "cloud": r"\b(aws|azure|gcp|google cloud|amazon web services|cloud computing|serverless)\b",
        "data": r"\b(sql|nosql|mongodb|postgresql|mysql|redis|elasticsearch|bigquery|snowflake)\b",
        "big_data": r"\b(spark|hadoop|kafka|airflow|hive|databricks|snowflake|redshift)\b",
        "devops": r"\b(docker|kubernetes|k8s|jenkins|gitlab|github actions|terraform|ansible|helm)\b",
        "visualization": r"\b(tableau|power bi|looker|matplotlib|seaborn|plotly|d3\.js|ggplot)\b",
        "statistics": r"\b(statistics|probability|hypothesis testing|a/b testing|regression|bayesian)\b"
    }

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or str(get_device())
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.load_model()

    def load_model(self):
        logger.info(f"Loading {self.model_name} for NER...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=3,
            id2label={0: "O", 1: "B-SKILL", 2: "I-SKILL"},
            label2id={"O": 0, "B-SKILL": 1, "I-SKILL": 2}
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def extract_with_regex(self, text: str) -> List[str]:
        if not text:
            return []
        text_lower = text.lower()
        skills = set()
        for category, pattern in self.SKILL_PATTERNS.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                skill = match.strip().title() if match.isalpha() else match.strip()
                skills.add(skill)
        return sorted(skills)

    def extract_with_ner(self, text: str) -> List[Dict]:
        if self.ner_pipeline is None:
            self.ner_pipeline = pipeline(
                "ner", model=self.model, tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        if not text or len(text.strip()) < 10:
            return []
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        results = []
        for chunk in chunks:
            try:
                ner_results = self.ner_pipeline(chunk)
                skills = self._aggregate_ner_results(ner_results)
                results.extend(skills)
            except Exception as e:
                logger.warning(f"NER extraction error: {e}")
        return results

    def _aggregate_ner_results(self, ner_results: List[Dict]) -> List[str]:
        skills = []
        current_skill = []
        for token in ner_results:
            if token["entity"] == "B-SKILL":
                if current_skill:
                    skills.append(" ".join(current_skill))
                current_skill = [token["word"]]
            elif token["entity"] == "I-SKILL":
                current_skill.append(token["word"].replace("##", ""))
            else:
                if current_skill:
                    skills.append(" ".join(current_skill))
                    current_skill = []
        if current_skill:
            skills.append(" ".join(current_skill))
        return skills

    def extract(self, text: str, method: str = "hybrid") -> List[str]:
        if method == "regex":
            return self.extract_with_regex(text)
        elif method == "ner":
            return self.extract_with_ner(text)
        else:
            regex_skills = set(self.extract_with_regex(text))
            ner_skills = set(s.lower() for s in self.extract_with_ner(text) if isinstance(s, str))
            combined = regex_skills | ner_skills
            return sorted(combined)

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        extractor = cls.__new__(cls)
        extractor.model_name = path
        extractor.device = device or str(get_device())
        extractor.tokenizer = AutoTokenizer.from_pretrained(path)
        extractor.model = AutoModelForTokenClassification.from_pretrained(path)
        extractor.model.to(extractor.device)
        extractor.model.eval()
        extractor.ner_pipeline = pipeline("ner", model=extractor.model, tokenizer=extractor.tokenizer,
                                          device=0 if extractor.device == "cuda" else -1)
        return extractor


class BERTSkillExtractor(SkillExtractor):
    def __init__(self, device: Optional[str] = None):
        super().__init__(model_name="bert-base-uncased", device=device)


class DistilBERTSkillExtractor(SkillExtractor):
    def __init__(self, device: Optional[str] = None):
        super().__init__(model_name="distilbert-base-uncased", device=device)


class RoBERTaSkillExtractor(SkillExtractor):
    def __init__(self, device: Optional[str] = None):
        super().__init__(model_name="roberta-base", device=device)


class SkillDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
