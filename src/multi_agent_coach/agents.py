from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from src.utils.logger import setup_logger
from src.skill_graph.skill_graph import SkillGraph, ROLE_SKILL_MAP
from src.skill_graph.career_paths import ROLE_SALARY_BANDS, CAREER_TRANSITIONS

logger = setup_logger(__name__)


@dataclass
class AgentResponse:
    agent_name: str
    agent_type: str
    response: str
    score: Optional[float] = None
    details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class BaseCareerAgent:
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.sg = SkillGraph()

    def process(self, context: Dict) -> AgentResponse:
        raise NotImplementedError

    def _get_role_skills(self, role: str) -> List[str]:
        return ROLE_SKILL_MAP.get(role, [])

    def _get_salary_band(self, role: str, level: str = "mid") -> tuple:
        band = ROLE_SALARY_BANDS.get(role, {})
        return band.get(level, (0, 0))


class RecruiterAgent(BaseCareerAgent):
    def __init__(self):
        super().__init__("Recruiter Agent", "recruiter")

    def process(self, context: Dict) -> AgentResponse:
        skills = context.get("skills", [])
        role = context.get("target_role", context.get("current_role", ""))
        experience = context.get("experience_years", 0)
        required = self._get_role_skills(role)
        matched = [s for s in skills if s in required]
        missing = [s for s in required if s not in skills]
        match_pct = len(matched) / max(len(required), 1) * 100
        score = min(100, match_pct + experience * 2)

        suggestions = []
        if match_pct < 50:
            suggestions.append(f"Critical gap: need {', '.join(missing[:3])}")
        if experience < 2:
            suggestions.append("Gain more hands-on project experience")
        if match_pct >= 70:
            suggestions.append("Strong match — ready for interview")

        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            response=f"**Resume Evaluation for {role}**\n- Match: {match_pct:.0f}%\n- Matched: {', '.join(matched[:5]) or 'None'}\n- Missing: {', '.join(missing[:5]) or 'None'}\n- Verdict: {'✅ Strong' if match_pct >= 70 else '⚠️ Needs work' if match_pct >= 40 else '❌ Major gaps'}",
            score=round(score, 1),
            details={"match_pct": round(match_pct, 1), "matched_skills": matched, "missing_skills": missing, "suggestions": suggestions},
        )


class SalaryNegotiatorAgent(BaseCareerAgent):
    def __init__(self):
        super().__init__("Salary Negotiator", "salary")

    def process(self, context: Dict) -> AgentResponse:
        role = context.get("target_role", context.get("current_role", ""))
        experience = context.get("experience_years", 0)
        current_salary = context.get("current_salary", 0)
        location = context.get("location", "Bangalore")
        level = "entry" if experience < 2 else "mid" if experience < 5 else "senior" if experience < 10 else "lead"
        band = self._get_salary_band(role, level)
        market_min, market_max = band
        is_india = location in ("Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad", "Jaipur")
        currency = "\u20b9" if is_india else "$"
        rate = 83 if is_india else 1
        expected_min = int(market_min * rate)
        expected_max = int(market_max * rate)
        negotiation_range = int(expected_min * 1.1), int(expected_max * 1.15)

        tips = []
        if current_salary and current_salary * rate < expected_min:
            tips.append(f"You're underpaid — ask for {currency}{expected_min:,}")
        tips.append(f"Market range: {currency}{expected_min:,} - {currency}{expected_max:,}")
        tips.append(f"Strong negotiation range: {currency}{negotiation_range[0]:,} - {currency}{negotiation_range[1]:,}")

        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            response=f"**Salary Guidance for {role} ({level})**\n- Market Range: {currency}{expected_min:,} - {currency}{expected_max:,}\n- Negotiation Target: {currency}{negotiation_range[0]:,} - {currency}{negotiation_range[1]:,}\n- {'Current: '+currency+str(current_salary*rate)+'' if current_salary else ''}",
            details={"market_min": expected_min, "market_max": expected_max, "negotiation_range": list(negotiation_range), "currency": currency, "tips": tips},
        )


class TechnicalInterviewerAgent(BaseCareerAgent):
    def __init__(self):
        super().__init__("Tech Interviewer", "interviewer")
        self._questions = self._build_questions()

    def _build_questions(self) -> Dict[str, List[Dict]]:
        return {
            "Data Scientist": [
                {"q": "Explain the bias-variance tradeoff", "topic": "ML Theory", "keywords": ["bias", "variance", "overfitting", "underfitting"]},
                {"q": "How would you handle imbalanced datasets?", "topic": "ML Practice", "keywords": ["sampling", "SMOTE", "class weight", "F1"]},
                {"q": "Explain gradient descent", "topic": "Optimization", "keywords": ["learning rate", "convergence", "backpropagation", "derivative"]},
            ],
            "Data Engineer": [
                {"q": "What's the difference between batch and stream processing?", "topic": "Data Architecture", "keywords": ["batch", "stream", "real-time", "lambda"]},
                {"q": "Explain how you'd design a data pipeline", "topic": "Pipeline Design", "keywords": ["ETL", "orchestration", "Airflow", "monitoring"]},
            ],
            "AI Engineer": [
                {"q": "How does a transformer model work?", "topic": "Deep Learning", "keywords": ["attention", "self-attention", "encoder", "decoder"]},
                {"q": "Explain RAG architecture", "topic": "GenAI", "keywords": ["retrieval", "generation", "embedding", "context"]},
            ],
        }

    def process(self, context: Dict) -> AgentResponse:
        role = context.get("target_role", context.get("current_role", "Data Scientist"))
        questions = self._questions.get(role, self._questions.get("Data Scientist", []))
        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            response=f"**Technical Interview: {role}**\n\nHere are practice questions:\n" + "\n".join(f"**{q['q']}** *(Topic: {q['topic']})*" for q in questions[:3]),
            details={"role": role, "questions": questions[:5], "total_questions": len(questions)},
        )


class CareerMentorAgent(BaseCareerAgent):
    def __init__(self):
        super().__init__("Career Mentor", "mentor")

    def process(self, context: Dict) -> AgentResponse:
        skills = context.get("skills", [])
        role = context.get("target_role", context.get("current_role", "Data Scientist"))
        timeline = context.get("timeline", "12 months")
        required = self._get_role_skills(role)
        missing = [s for s in required if s not in skills]

        roadmap = []
        for i, skill in enumerate(missing[:6]):
            month = i * 2 + 1
            roadmap.append({"step": i + 1, "skill": skill, "month": f"Month {month}-{month+1}", "effort": "4 weeks"})

        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            response=f"**Learning Roadmap: {role}**\nTimeline: {timeline}\n" + "\n".join(f"**Step {r['step']}**: Learn {r['skill']} ({r['month']})" for r in roadmap),
            details={"target_role": role, "roadmap": roadmap, "missing_skills": missing, "timeline": timeline},
        )


class HRAgent(BaseCareerAgent):
    def __init__(self):
        super().__init__("HR Advisor", "hr")

    def process(self, context: Dict) -> AgentResponse:
        skills = context.get("skills", [])
        role = context.get("target_role", context.get("current_role", ""))
        required = self._get_role_skills(role)
        matched_technical = len([s for s in skills if s in required])
        has_soft = any(s.lower() in ("communication", "leadership", "teamwork", "presentation") for s in skills)
        soft_score = 70 if has_soft else 40
        tech_score = matched_technical / max(len(required), 1) * 100
        overall = int(tech_score * 0.6 + soft_score * 0.4)

        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            response=f"**Soft Skills & Cultural Fit Analysis**\n- Technical Fit: {tech_score:.0f}%\n- Soft Skills Score: {soft_score}/100\n- Overall Fit: {overall}%\n- {'✅ Good cultural fit' if overall >= 70 else '⚠️ Consider highlighting soft skills'}",
            score=overall,
            details={"tech_score": round(tech_score, 1), "soft_score": soft_score, "overall_fit": overall, "has_soft_skills": has_soft},
        )


class CareerAgentOrchestrator:
    def __init__(self):
        self.agents = {
            "recruiter": RecruiterAgent(),
            "salary": SalaryNegotiatorAgent(),
            "interviewer": TechnicalInterviewerAgent(),
            "mentor": CareerMentorAgent(),
            "hr": HRAgent(),
        }

    def get_all_agents(self) -> List[str]:
        return list(self.agents.keys())

    def run_agent(self, agent_name: str, context: Dict) -> AgentResponse:
        agent = self.agents.get(agent_name)
        if not agent:
            return AgentResponse("Unknown", "error", f"Agent '{agent_name}' not found", score=0)
        return agent.process(context)

    def run_all(self, context: Dict) -> Dict[str, AgentResponse]:
        return {name: agent.process(context) for name, agent in self.agents.items()}

    def get_consensus(self, context: Dict) -> Dict:
        results = self.run_all(context)
        scores = []
        summary_parts = []
        for name, resp in results.items():
            if resp.score is not None:
                scores.append(resp.score)
            summary_parts.append(f"**{resp.agent_name}**: {resp.response}")
        avg_score = sum(scores) / len(scores) if scores else 0
        return {
            "results": {name: r.to_dict() for name, r in results.items()},
            "average_score": round(avg_score, 1),
            "summary": "\n\n---\n\n".join(summary_parts),
            "readiness": "High" if avg_score >= 70 else "Medium" if avg_score >= 40 else "Low",
        }
