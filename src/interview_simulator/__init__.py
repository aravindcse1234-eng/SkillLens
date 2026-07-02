from typing import Dict, List, Optional
import random
import re
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

INTERVIEW_QUESTIONS = {
    "Data Scientist": [
        {"q": "Explain the bias-variance tradeoff and how you would diagnose high bias vs high variance in a model.", "topic": "ML Theory", "keywords": ["bias", "variance", "overfitting", "underfitting", "cross-validation", "learning curve"]},
        {"q": "How do you handle class imbalance in a classification problem?", "topic": "ML Practice", "keywords": ["SMOTE", "class weight", "resampling", "F1", "AUC", "stratified"]},
        {"q": "Explain gradient descent and its variants (SGD, Adam, RMSprop).", "topic": "Optimization", "keywords": ["learning rate", "momentum", "convergence", "adaptive", "batch", "stochastic"]},
        {"q": "What is regularization and why is it important?", "topic": "ML Theory", "keywords": ["L1", "L2", "dropout", "overfitting", "generalization", "lambda"]},
        {"q": "Describe a data science project end-to-end that you've worked on.", "topic": "Experience", "keywords": ["problem", "data", "feature", "model", "evaluation", "deployment"]},
    ],
    "Data Engineer": [
        {"q": "What's the difference between batch processing and stream processing?", "topic": "Data Architecture", "keywords": ["batch", "stream", "real-time", "lambda", "Kafka", "Spark"]},
        {"q": "Design a data pipeline for a real-time recommendation system.", "topic": "Pipeline Design", "keywords": ["ingestion", "processing", "storage", "orchestration", "monitoring", "Airflow"]},
        {"q": "Explain how you would optimize a slow Hive/Presto query.", "topic": "Performance", "keywords": ["partition", "index", "optimize", "shuffle", "broadcast", "caching"]},
        {"q": "What's the CAP theorem and how does it apply to distributed databases?", "topic": "Distributed Systems", "keywords": ["consistency", "availability", "partition", "CP", "AP", "tradeoff"]},
        {"q": "How do you ensure data quality in a production pipeline?", "topic": "Data Quality", "keywords": ["validation", "monitoring", "alerting", "schema", "lineage", "testing"]},
    ],
    "ML Engineer": [
        {"q": "Explain MLOps and how it differs from traditional DevOps.", "topic": "MLOps", "keywords": ["pipeline", "automation", "monitoring", "versioning", "reproducibility", "CI/CD"]},
        {"q": "How would you deploy a machine learning model to production?", "topic": "Deployment", "keywords": ["container", "API", "scaling", "monitoring", "A/B testing", "rollback"]},
        {"q": "What is model drift and how do you detect it?", "topic": "Monitoring", "keywords": ["concept drift", "data drift", "distribution", "alert", "retrain", "champion"]},
        {"q": "Explain feature store and its benefits.", "topic": "Feature Engineering", "keywords": ["centralized", "reusable", "serving", "consistency", "online", "offline"]},
        {"q": "How do you handle large-scale model training?", "topic": "Scaling", "keywords": ["distributed", "GPU", "parallel", "hyperparameter", "early stopping", "checkpoint"]},
    ],
    "AI Engineer": [
        {"q": "Explain the Transformer architecture and why it's revolutionary.", "topic": "Deep Learning", "keywords": ["attention", "self-attention", "encoder", "decoder", "positional", "parallel"]},
        {"q": "How does RAG (Retrieval-Augmented Generation) work?", "topic": "GenAI", "keywords": ["retrieval", "generation", "embedding", "context", "vector DB", "prompt"]},
        {"q": "What are the key challenges in deploying LLMs to production?", "topic": "LLM Ops", "keywords": ["latency", "cost", "hallucination", "safety", "prompt injection", "caching"]},
        {"q": "Explain fine-tuning vs RAG vs prompt engineering.", "topic": "GenAI", "keywords": ["fine-tune", "retrieval", "prompt", "cost", "quality", "control"]},
        {"q": "How would you evaluate the performance of a generative AI system?", "topic": "Evaluation", "keywords": ["ROUGE", "BLEU", "human eval", "hallucination", "relevance", "faithfulness"]},
    ],
    "Software Engineer": [
        {"q": "Explain the SOLID principles of object-oriented design.", "topic": "Design", "keywords": ["single responsibility", "open-closed", "substitution", "interface", "dependency"]},
        {"q": "How would you design a rate limiter for a large-scale API?", "topic": "System Design", "keywords": ["token bucket", "sliding window", "distributed", "Redis", "throttling", "scaling"]},
        {"q": "What's the difference between REST and GraphQL?", "topic": "API Design", "keywords": ["query", "mutation", "over-fetching", "schema", "endpoint", "flexibility"]},
    ],
}

KEYWORD_WEIGHTS = {
    "technical": 1.0, "conceptual": 0.7, "experience": 0.5,
}


class InterviewSimulator:
    def __init__(self):
        self.questions = INTERVIEW_QUESTIONS
        self.sessions = {}

    def start_session(self, role: str, user_id: str = "default") -> Dict:
        questions = self.questions.get(role, self.questions.get("Data Scientist", []))
        shuffled = random.sample(questions, min(len(questions), 5))
        session = {
            "role": role,
            "questions": shuffled,
            "current_index": 0,
            "answers": [],
            "status": "in_progress",
        }
        self.sessions[user_id] = session
        return self._get_current(session)

    def _get_current(self, session: Dict) -> Optional[Dict]:
        idx = session["current_index"]
        if idx >= len(session["questions"]):
            return None
        q = session["questions"][idx]
        return {"index": idx, "total": len(session["questions"]), "question": q["q"], "topic": q["topic"]}

    def submit_answer(self, answer: str, user_id: str = "default") -> Dict:
        session = self.sessions.get(user_id)
        if not session or session["status"] != "in_progress":
            return {"error": "No active session"}
        idx = session["current_index"]
        q = session["questions"][idx]
        score = self._score_answer(answer, q)
        session["answers"].append({"question": q["q"], "answer": answer, "score": score})
        session["current_index"] += 1
        if session["current_index"] >= len(session["questions"]):
            session["status"] = "completed"
            result = self._finalize(session)
            return result
        return {
            "status": "in_progress",
            "next_question": self._get_current(session),
            "last_score": score,
        }

    def _score_answer(self, answer: str, question: Dict) -> Dict:
        answer_lower = answer.lower()
        keywords = question.get("keywords", [])
        matched = [kw for kw in keywords if kw.lower() in answer_lower]
        technical_depth = min(100, len(matched) / max(len(keywords), 1) * 100)
        keyword_coverage = len(matched) / max(len(keywords), 1)
        word_count = len(answer.split())
        completeness = min(100, word_count / 30 * 100) if word_count > 10 else max(0, word_count / 10 * 50)
        technical_score = int(technical_depth * 0.6 + completeness * 0.4)
        communication_score = int(min(100, (1 - abs(len(answer.split()) - 50) / 100) * 70 + 30))
        confidence_score = int(min(100, keyword_coverage * 80 + (1 if len(answer) > 100 else 0) * 20))
        overall = int(technical_score * 0.5 + communication_score * 0.25 + confidence_score * 0.25)
        return {
            "technical_score": technical_score,
            "communication_score": communication_score,
            "confidence_score": confidence_score,
            "overall": overall,
            "keywords_matched": matched,
            "keywords_total": len(keywords),
            "keyword_coverage": round(keyword_coverage, 2),
            "word_count": word_count,
        }

    def _finalize(self, session: Dict) -> Dict:
        scores = [a["score"] for a in session["answers"]]
        avg_technical = sum(s["technical_score"] for s in scores) / len(scores)
        avg_communication = sum(s["communication_score"] for s in scores) / len(scores)
        avg_confidence = sum(s["confidence_score"] for s in scores) / len(scores)
        overall = int(avg_technical * 0.5 + avg_communication * 0.25 + avg_confidence * 0.25)
        strengths = []
        improvements = []
        for i, a in enumerate(session["answers"]):
            if a["score"]["overall"] >= 70:
                strengths.append(f"Q{i+1}: {a['score']['overall']}/100")
            else:
                improvements.append(f"Q{i+1}: {a['score']['overall']}/100 - need more {'keywords' if a['score']['keyword_coverage'] < 0.5 else 'depth'}")
        return {
            "status": "completed",
            "role": session["role"],
            "answers": session["answers"],
            "overall_score": overall,
            "technical_score": int(avg_technical),
            "communication_score": int(avg_communication),
            "confidence_score": int(avg_confidence),
            "strengths": strengths,
            "improvements": improvements,
            "total_questions": len(session["answers"]),
        }

    def get_progress(self, user_id: str = "default") -> Dict:
        session = self.sessions.get(user_id)
        if not session:
            return {"status": "not_started"}
        return {
            "status": session["status"],
            "current": session["current_index"],
            "total": len(session["questions"]),
            "question": self._get_current(session),
        }
