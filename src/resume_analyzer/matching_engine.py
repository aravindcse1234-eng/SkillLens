from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.resume_analyzer.analyzer import ResumeAnalyzer
from src.resume_analyzer.ats_scorer import ATSScorer
from src.nlp.skill_extractor import SkillExtractor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResumeJobMatcher:
    def __init__(self):
        self.resume_analyzer = ResumeAnalyzer()
        self.ats_scorer = ATSScorer()
        self.skill_extractor = SkillExtractor()

    def match_resume_to_jobs(self, resume_data: Dict,
                              jobs_df: Optional[pd.DataFrame] = None) -> Dict:
        resume_skills = resume_data.get("extracted_skills", [])
        resume_exp = resume_data.get("experience_years", 0)
        resume_edu = resume_data.get("education_level", "")

        if jobs_df is not None and not jobs_df.empty:
            results = []
            for _, job in jobs_df.iterrows():
                job_skills = job.get("skills", []) if isinstance(job.get("skills"), list) else []
                if isinstance(job_skills, str):
                    job_skills = [s.strip() for s in job_skills.split(",")]

                job_req = {
                    "description": job.get("description", ""),
                    "required_skills": job_skills,
                    "experience_required": job.get("years_experience", 0),
                    "education_required": job.get("education_level", ""),
                }
                score_result = self.ats_scorer.calculate_ats_score(resume_data, job_req)
                results.append({
                    "job_title": job.get("title", job.get("job_title", "")),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "salary": job.get("salary", 0),
                    "ats_score": score_result["ats_score"],
                    "components": score_result["components"],
                    "matched_skills": score_result["matched_skills"],
                    "missing_skills": score_result["missing_skills"],
                })
            results.sort(key=lambda x: x["ats_score"], reverse=True)
            return {"results": results, "total_jobs": len(results), "top_match": results[0] if results else None}

        return {"results": [], "total_jobs": 0, "top_match": None}

    def match_resume_to_role(self, resume_data: Dict, target_role: str) -> Dict:
        resume_skills = resume_data.get("extracted_skills", [])
        from src.skill_graph.skill_graph import SkillGraph
        sg = SkillGraph()
        role_skills = sg.get_role_skills(target_role)
        missing = [s for s in role_skills if s not in resume_skills]
        matched = [s for s in role_skills if s in resume_skills]
        match_pct = len(matched) / len(role_skills) * 100 if role_skills else 0
        recs = sg.recommend_skills(resume_skills, top_k=5)

        return {
            "target_role": target_role,
            "match_percentage": round(match_pct, 1),
            "matched_skills": matched,
            "missing_skills": missing,
            "total_required": len(role_skills),
            "total_matched": len(matched),
            "recommended_skills": recs,
            "verdict": "Highly Suitable" if match_pct >= 80 else (
                "Good Match" if match_pct >= 60 else (
                    "Fair Match" if match_pct >= 40 else "Needs Improvement")),
        }

    def generate_match_report(self, resume_data: Dict,
                               match_result: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("RESUME-JOB MATCH REPORT")
        lines.append("=" * 60)
        m = match_result
        lines.append(f"\nTarget Role: {m.get('target_role', 'N/A')}")
        lines.append(f"Match Score: {m.get('match_percentage', 0)}%")
        lines.append(f"Verdict: {m.get('verdict', 'N/A')}")
        lines.append(f"\nSkills Matched ({m.get('total_matched', 0)}/{m.get('total_required', 0)}):")
        for s in m.get("matched_skills", []):
            lines.append(f"  ✅ {s}")
        lines.append(f"\nSkills to Acquire ({len(m.get('missing_skills', []))}):")
        for s in m.get("missing_skills", []):
            lines.append(f"  ❌ {s}")
        if m.get("recommended_skills"):
            lines.append("\nRecommended Skills to Learn Next:")
            for r in m["recommended_skills"]:
                lines.append(f"  💡 {r['skill']} (relevance: {r['score']:.2f})")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def batch_match(self, resumes_dir: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
        import os
        from src.resume_analyzer.parser import ResumeParser
        parser = ResumeParser()
        all_results = []
        for fname in os.listdir(resumes_dir):
            if fname.endswith((".pdf", ".docx")):
                fpath = os.path.join(resumes_dir, fname)
                try:
                    resume_data = parser.parse(fpath)
                    analysis = self.resume_analyzer.analyze(resume_data)
                    match = self.match_resume_to_jobs(analysis, jobs_df)
                    for r in match.get("results", []):
                        all_results.append({
                            "candidate": fname,
                            **r,
                        })
                except Exception as e:
                    logger.warning(f"Failed to process {fname}: {e}")
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
