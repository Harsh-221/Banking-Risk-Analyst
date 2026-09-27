"""Small local TF-IDF retriever for the synthetic Markdown policy corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class PolicyChunk:
    source: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} — {self.section}"


class PolicyRetriever:
    def __init__(self, documents_dir: Path):
        self.documents_dir = documents_dir
        self.chunks = self._load_chunks()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks]) if self.chunks else None

    def _load_chunks(self) -> list[PolicyChunk]:
        chunks = []
        for path in sorted(self.documents_dir.rglob("*.md")):
            section = path.stem.replace("_", " ").title()
            paragraph = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    if paragraph:
                        text = " ".join(paragraph).strip()
                        if text and not text.startswith(">"):
                            chunks.append(PolicyChunk(path.name, section, text))
                        paragraph = []
                    section = re.sub(r"^#+\s*", "", line).strip() or section
                elif not line.strip():
                    if paragraph:
                        text = " ".join(paragraph).strip()
                        if text and not text.startswith(">"):
                            chunks.append(PolicyChunk(path.name, section, text))
                        paragraph = []
                else:
                    paragraph.append(line.strip())
            if paragraph:
                text = " ".join(paragraph).strip()
                if text and not text.startswith(">"):
                    chunks.append(PolicyChunk(path.name, section, text))
        return chunks

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        if self.matrix is None or not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        best = scores.argsort()[::-1][:top_k]
        results = []
        for index in best:
            score = float(scores[index])
            if score < 0.015:
                continue
            chunk = self.chunks[index]
            results.append({"source": chunk.source, "section": chunk.section, "citation": chunk.citation, "text": chunk.text, "relevance": round(score, 4)})
        return results
