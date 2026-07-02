import pytest
import numpy as np
from src.chatbot.knowledge_base import CareerKnowledgeBase, KnowledgeBase


class TestKnowledgeBase:
    def test_add_documents(self):
        kb = KnowledgeBase(base_path="/tmp/test_kb")
        doc = {"id": "test1", "text": "This is a test document about data science careers.", "metadata": {"cat": "test"}, "source": "test"}
        kb.add_documents([doc])
        assert len(kb.documents) == 1
        assert len(kb.chunks) >= 1

    def test_chunk_large_document(self):
        kb = KnowledgeBase(base_path="/tmp/test_kb")
        long_text = "word " * 1000
        doc = {"id": "long", "text": long_text, "metadata": {}, "source": "test"}
        kb.add_documents([doc])
        assert len(kb.chunks) > 1

    def test_save_load_roundtrip(self, tmp_path):
        kb = KnowledgeBase(base_path="/tmp/test_kb")
        doc = {"id": "rt1", "text": "Round trip test", "metadata": {}, "source": "test"}
        kb.add_documents([doc])
        path = str(tmp_path / "kb.json")
        kb.save(path)
        loaded = KnowledgeBase(base_path="/tmp/test_kb")
        loaded.load(path)
        assert len(loaded.documents) == 1
        assert loaded.documents[0]["text"] == "Round trip test"


class TestCareerKnowledgeBase:
    def test_init_loads_career_data(self):
        kb = CareerKnowledgeBase(base_path="/tmp/test_career_kb")
        assert len(kb.documents) > 0
        assert len(kb.chunks) > 0

    def test_contains_career_topics(self):
        kb = CareerKnowledgeBase(base_path="/tmp/test_career_kb")
        all_text = " ".join(d["text"] for d in kb.documents).lower()
        assert "data science" in all_text
        assert "interview" in all_text
        assert "salary" in all_text
        assert "roadmap" in all_text
