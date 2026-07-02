import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.utils.helpers import ensure_dir, save_json, load_json
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class KnowledgeBase:
    def __init__(self, base_path: str = "data/knowledge_base"):
        self.base_path = ensure_dir(base_path)
        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []

    def add_documents(self, documents: List[Dict]) -> None:
        self.documents.extend(documents)
        self._chunk_documents()
        logger.info(f"Added {len(documents)} documents")

    def _chunk_documents(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunks = []
        for doc in self.documents:
            text = doc.get("text", "")
            if len(text) <= chunk_size:
                self.chunks.append({
                    "id": f"{doc.get('id', 'doc')}_0",
                    "text": text,
                    "metadata": doc.get("metadata", {}),
                    "source": doc.get("source", "")
                })
            else:
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_text = " ".join(words[i:i + chunk_size])
                    self.chunks.append({
                        "id": f"{doc.get('id', 'doc')}_{i}",
                        "text": chunk_text,
                        "metadata": doc.get("metadata", {}),
                        "source": doc.get("source", "")
                    })
        logger.info(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")

    def search(self, query: str, index, embedding_fn, top_k: int = 5) -> List[Dict]:
        query_emb = embedding_fn([query])
        query_emb = query_emb.astype(np.float32)
        import faiss
        faiss.normalize_L2(query_emb)
        scores, indices = index.search(query_emb, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk["text"],
                    "score": float(scores[0][i]),
                    "metadata": chunk["metadata"],
                    "source": chunk["source"],
                })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def save(self, path: str):
        data = {"documents": self.documents, "chunks": self.chunks}
        save_json(data, path)

    def load(self, path: str):
        data = load_json(path)
        self.documents = data["documents"]
        self.chunks = data["chunks"]
        logger.info(f"Loaded {len(self.documents)} documents from {path}")
        return self


class CareerKnowledgeBase(KnowledgeBase):
    def __init__(self, base_path: str = "data/knowledge_base"):
        super().__init__(base_path)
        self._init_career_data()

    def _init_career_data(self) -> None:
        career_docs = [
            {
                "id": "career_1",
                "text": "Data Science is one of the fastest-growing fields. Key skills include Python, SQL, Machine Learning, Statistics, and Data Visualization. Average salary ranges from $90,000 to $150,000 depending on experience.",
                "metadata": {"category": "career", "subcategory": "data_science"},
                "source": "SkillLens AI Knowledge Base"
            },
            {
                "id": "career_2",
                "text": "Machine Learning Engineers design and implement ML systems. They need strong programming skills, deep learning knowledge, and experience with MLOps. Salaries range from $110,000 to $180,000.",
                "metadata": {"category": "career", "subcategory": "ml_engineering"},
                "source": "SkillLens AI Knowledge Base"
            },
            {
                "id": "career_3",
                "text": "For interview preparation: 1) Practice coding on LeetCode, 2) Review ML fundamentals, 3) Prepare case studies from past projects, 4) Study system design for ML systems, 5) Practice behavioral questions using STAR method.",
                "metadata": {"category": "interview", "subcategory": "preparation"},
                "source": "SkillLens AI Knowledge Base"
            },
            {
                "id": "career_4",
                "text": "Top skills for 2025: Generative AI (GenAI), Large Language Models, Prompt Engineering, MLOps, Cloud Computing (AWS/Azure/GCP), Data Engineering, and Cybersecurity. These skills have seen 40%+ growth in job postings.",
                "metadata": {"category": "skills", "subcategory": "trending"},
                "source": "SkillLens AI Knowledge Base"
            },
            {
                "id": "career_5",
                "text": "To transition into AI/ML: 1) Master Python and SQL, 2) Take Andrew Ng's ML course, 3) Build a portfolio of 3-5 projects, 4) Contribute to open source, 5) Network at AI conferences, 6) Get relevant certifications.",
                "metadata": {"category": "career", "subcategory": "transition"},
                "source": "SkillLens AI Knowledge Base"
            },
            {
                "id": "career_6",
                "text": "Salary insights by experience: Entry-level (0-2 yrs): $70k-$95k, Mid-level (3-5 yrs): $95k-$130k, Senior (5-8 yrs): $130k-$170k, Lead (8+ yrs): $170k-$220k. Location and industry significantly impact these ranges.",
                "metadata": {"category": "salary", "subcategory": "insights"},
                "source": "SkillLens AI Knowledge Base"
            },
            {
                "id": "career_7",
                "text": "Learning roadmap for Data Science: Month 1-2: Python, SQL, Statistics. Month 3-4: Machine Learning, Data Visualization. Month 5-6: Deep Learning, NLP. Month 7-8: Big Data Tools, Cloud. Month 9-12: Advanced Topics, Portfolio Projects.",
                "metadata": {"category": "learning", "subcategory": "roadmap"},
                "source": "SkillLens AI Knowledge Base"
            },
            {
                "id": "career_8",
                "text": "Data Engineering skills are in high demand. Key technologies: Apache Spark, Airflow, Kafka, Snowflake, dbt, Docker, Kubernetes, SQL, Python. Data Engineers build and maintain data pipelines and infrastructure.",
                "metadata": {"category": "career", "subcategory": "data_engineering"},
                "source": "SkillLens AI Knowledge Base"
            },
        ]
        self.add_documents(career_docs)
