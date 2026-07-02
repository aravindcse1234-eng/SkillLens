import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.utils.helpers import ensure_dir
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ResumeParser:
    def __init__(self):
        self.supported_formats = [".pdf", ".docx", ".txt"]

    def parse(self, file_path: str) -> Dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            text = self._parse_pdf(path)
        elif ext == ".docx":
            text = self._parse_docx(path)
        elif ext == ".txt":
            text = path.read_text("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported format: {ext}")

        return {
            "file_path": str(path),
            "file_name": path.name,
            "raw_text": text,
            "clean_text": self._clean_text(text),
        }

    def _parse_pdf(self, path: Path) -> str:
        logger.info(f"Parsing PDF: {path.name}")
        try:
            import pdfminer.high_level as miner
            text = miner.extract_text(str(path))
            return text if text.strip() else ""
        except Exception as e:
            logger.warning(f"pdfminer failed: {e}")
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            except Exception as e2:
                logger.error(f"PyPDF2 also failed: {e2}")
                return ""

    def _parse_docx(self, path: Path) -> str:
        logger.info(f"Parsing DOCX: {path.name}")
        try:
            import docx
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s@.,;:!?()/-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        patterns = {
            "contact": r"(?i)(contact|email|phone|linkedin|github)",
            "summary": r"(?i)(summary|profile|objective|about me)",
            "experience": r"(?i)(experience|work|employment|professional background)",
            "education": r"(?i)(education|academic|qualification|degree)",
            "skills": r"(?i)(skills|technologies|competencies|technical skills)",
            "projects": r"(?i)(projects|portfolio|personal projects)",
            "certifications": r"(?i)(certifications|certificates|licenses)",
            "achievements": r"(?i)(achievements|awards|honors)",
            "publications": r"(?i)(publications|papers|research)",
        }

        lines = text.split("\n")
        current_section = "header"
        sections["header"] = ""

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            matched = False
            for section_name, pattern in patterns.items():
                if re.match(pattern, line_stripped) and len(line_stripped) < 100:
                    current_section = section_name
                    if current_section not in sections:
                        sections[current_section] = ""
                    matched = True
                    break
            if not matched:
                sections.setdefault(current_section, "")
                sections[current_section] += line_stripped + "\n"

        return sections
