from typing import Dict, List, Optional
from src.chatbot.knowledge_base import CareerKnowledgeBase
from src.chatbot.embeddings import EmbeddingManager
from src.utils.helpers import get_device
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGChatbot:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.knowledge_base = CareerKnowledgeBase()
        self.index = None
        self._build_index()

    def _build_index(self):
        if not self.knowledge_base.chunks:
            logger.warning("Knowledge base is empty")
            return
        texts = [chunk["text"] for chunk in self.knowledge_base.chunks]
        embeddings = self.embedding_manager.encode(texts)
        self.index = self.embedding_manager.build_faiss_index(embeddings)
        logger.info(f"FAISS index built with {len(texts)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            return []
        return self.knowledge_base.search(query, self.index, self.embedding_manager.encode, top_k)

    def generate_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in context])
        prompt = f"""You are an expert AI Career Assistant. Use the following context to answer the user's question.

Context:
{context_text}

User Question: {query}

Provide a helpful, accurate, and concise response based on the context. If the context doesn't contain enough information, acknowledge that and provide general guidance."""
        return prompt

    def answer(self, query: str, use_llm: bool = False) -> Dict:
        retrieved_docs = self.retrieve(query)
        enriched = self._enrich_with_knowledge(query, retrieved_docs)
        if use_llm:
            prompt = self.generate_prompt(query, enriched)
            response = self._call_llm(prompt)
        else:
            response = self._generate_response(query, enriched)
        return {
            "query": query,
            "response": response,
            "sources": [{"text": d["text"][:200], "score": d["score"]} for d in enriched[:3]],
            "confidence": max([d["score"] for d in enriched]) if enriched else 0,
        }

    def _enrich_with_knowledge(self, query: str, docs: List[Dict]) -> List[Dict]:
        q = query.lower()
        from src.skill_graph.skill_graph import SkillGraph
        try:
            sg = SkillGraph()
            enrichment = []
            for role in sg.get_all_roles():
                if role.lower() in q:
                    skills = sg.get_role_skills(role)
                    enrichment.append({
                        "source": "Skill Graph",
                        "text": f"The {role} role typically requires: {', '.join(skills)}. "
                                f"Related skills can be explored through our skill graph.",
                        "score": 0.9,
                    })
            for category in sg.get_categories():
                if category.lower() in q:
                    skills = sg.get_skills_by_category(category)[:8]
                    enrichment.append({
                        "source": "Skill Graph",
                        "text": f"Skills in the {category} category include: {', '.join(skills)}.",
                        "score": 0.85,
                    })
            if "salary" in q:
                for role in sg.get_all_roles():
                    if role.lower() in q:
                        from src.skill_graph.career_paths import ROLE_SALARY_BANDS
                        bands = ROLE_SALARY_BANDS.get(role, {})
                        if bands:
                            enrichment.append({
                                "source": "Salary Data",
                                "text": f"Salary bands for {role}: Entry: ${bands['entry'][0]:,}-${bands['entry'][1]:,}, "
                                        f"Mid: ${bands['mid'][0]:,}-${bands['mid'][1]:,}, "
                                        f"Senior: ${bands['senior'][0]:,}-${bands['senior'][1]:,}.",
                                "score": 0.95,
                            })
            all_docs = enrichment + docs
            all_docs.sort(key=lambda x: x["score"], reverse=True)
            return all_docs[:5]
        except ImportError:
            return docs

    def _generate_response(self, query: str, context: List[Dict]) -> str:
        if not context:
            return ("I'm here to help with career advice! Try asking about:\n"
                    "• Skills in demand for specific roles\n"
                    "• Salary ranges by position and location\n"
                    "• Learning roadmaps and career transitions\n"
                    "• Resume improvement and interview preparation")

        parts = []
        for doc in context[:3]:
            source = doc.get("source", "Knowledge Base")
            text = doc["text"]
            if len(text) > 500:
                text = text[:500] + "..."
            parts.append(f"📚 *From {source}*\n{text}")

        return "\n\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        try:
            from transformers import pipeline
            generator = pipeline(
                "text-generation",
                model="google/gemma-2b-it",
                device=0 if get_device().type == "cuda" else -1,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )
            result = generator(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
            return result[0]["generated_text"].replace(prompt, "").strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return self._generate_response(
                prompt.split("User Question:")[-1].strip(),
                self.retrieve(prompt.split("User Question:")[-1].strip()))


class CareerAssistant(RAGChatbot):
    def __init__(self):
        super().__init__()

    def get_career_advice(self, query: str) -> Dict:
        return self.answer(f"Career advice: {query}")

    def get_skill_recommendations(self, current_skills: List[str], target_role: str) -> Dict:
        query = f"My skills are {', '.join(current_skills)}. I want to become a {target_role}. What skills should I learn?"
        return self.answer(query)

    def get_interview_tips(self, role: str) -> Dict:
        return self.answer(f"Interview preparation tips for {role} position")

    def get_salary_insights(self, role: str, experience: str, location: str) -> Dict:
        return self.answer(f"Salary insights for {role} with {experience} experience in {location}")

    def get_learning_roadmap(self, target_role: str, timeline: str = "6 months") -> Dict:
        return self.answer(f"Learning roadmap to become a {target_role} within {timeline}")
