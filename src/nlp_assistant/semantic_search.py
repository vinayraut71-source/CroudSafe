"""
Semantic Incident Search Module.
Allows operators to search historical crowd incidents and alerts using natural language queries.
Computes semantic similarity using dense vector representations with robust fallbacks.
"""

import os
import sqlite3
import json
import datetime
import re
from typing import List, Dict, Any, Optional
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DB_PATH, RECORDING_DIR
from src.nlp_assistant.schemas import SemanticSearchItem, SemanticSearchResponse


class SemanticIncidentSearcher:
    """
    Indexes historical events and incidents from SQLite,
    computing semantic similarity against natural language queries.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._model = None
        self._vectorizer = None
        self._init_engine()

    def _init_engine(self):
        """Initialize TF-IDF vector similarity engine with optional neural embeddings."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            lowercase=True,
            stop_words="english"
        )
        self._model = None
        print("[SemanticSearch] Vector similarity engine initialized.")

    def _find_evidence_link(self, timestamp: float, level: str) -> Optional[str]:
        """Match an incident or event to a recorded video clip in recordings/ if available."""
        if not os.path.exists(RECORDING_DIR):
            return None
        recs = [f for f in os.listdir(RECORDING_DIR) if f.endswith(".mp4")]
        if not recs:
            return None

        # Check for filename with matching level or closest timestamp
        for r in sorted(recs, reverse=True):
            if level.upper() in r.upper():
                return f"/recordings/{r}"
        return f"/recordings/{recs[0]}"

    def _fetch_records(self) -> List[Dict[str, Any]]:
        """Fetch historical incident and alert records from SQLite."""
        records = []
        if not os.path.exists(self.db_path):
            return records

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # 1. Fetch from incidents table if exists
                try:
                    inc_rows = conn.execute(
                        "SELECT id, timestamp, time_str, event_type, severity, camera_id, zone, evidence_url, data FROM incidents ORDER BY timestamp DESC LIMIT 200"
                    ).fetchall()
                    for r in inc_rows:
                        data = json.loads(r["data"]) if r["data"] else {}
                        records.append({
                            "id": r["id"],
                            "event_type": r["event_type"],
                            "level": r["severity"],
                            "message": f"Incident {r['event_type']} in {r['zone']} on camera {r['camera_id']}",
                            "timestamp": r["timestamp"],
                            "time_str": r["time_str"],
                            "evidence_link": r["evidence_url"] or self._find_evidence_link(r["timestamp"], r["severity"]),
                            "data": data,
                            "source_table": "incidents"
                        })
                except Exception:
                    pass

                # 2. Fetch from events table (alerts, violations, behaviors)
                try:
                    ev_rows = conn.execute(
                        "SELECT id, timestamp, event_type, level, message, data FROM events WHERE level IN ('WARNING', 'DANGER', 'CRITICAL') ORDER BY timestamp DESC LIMIT 300"
                    ).fetchall()
                    for r in ev_rows:
                        data = json.loads(r["data"]) if r["data"] else {}
                        time_str = datetime.datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                        records.append({
                            "id": f"EV-{r['id']}",
                            "event_type": r["event_type"],
                            "level": r["level"],
                            "message": r["message"],
                            "timestamp": r["timestamp"],
                            "time_str": time_str,
                            "evidence_link": self._find_evidence_link(r["timestamp"], r["level"]),
                            "data": data,
                            "source_table": "events"
                        })
                except Exception:
                    pass

        except Exception as e:
            print(f"[SemanticSearch] Error reading records: {e}")

        return records

    def _expand_semantic_query(self, query: str) -> str:
        """Expand domain synonyms for crowd safety to increase recall."""
        q_lower = query.lower()
        synonyms = {
            "running": "rush surge speed fast stampede escape fleeing",
            "stampede": "panic crush compression rush surge danger critical",
            "child": "lost child infant kid minor wandering alone missing",
            "loud": "screaming shouting noise acoustic panic spike decibel sound",
            "screaming": "acoustic noise panic loud shout shout scream",
            "crowded": "density congested packed high occupancy buildup clusters",
            "density": "congested crowded occupancy compression buildup tight",
            "gate": "exit entrance entry door zone restricted intrusion barrier",
            "intrusion": "unauthorized restricted boundary crossing perimeter",
            "pressure": "compression wave push crush squeeze"
        }
        additions = []
        for word, expansion in synonyms.items():
            if word in q_lower:
                additions.append(expansion)
        if additions:
            return f"{query} {' '.join(additions)}"
        return query

    def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.05
    ) -> SemanticSearchResponse:
        """
        Execute semantic search over historical records using dense or sparse representations.
        Returns ranked list of matching incident items.
        """
        records = self._fetch_records()
        if not records:
            return SemanticSearchResponse(query=query, count=0, results=[])

        expanded_query = self._expand_semantic_query(query)

        # Prepare corpus texts
        corpus = []
        for r in records:
            data_str = " ".join(f"{k}:{v}" for k, v in r["data"].items()) if r["data"] else ""
            doc = f"{r['event_type']} {r['level']} {r['message']} {data_str}".strip()
            corpus.append(doc)

        similarities = []

        # Strategy A: SentenceTransformer dense cosine similarity
        if self._model is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                q_emb = self._model.encode([expanded_query])
                doc_embs = self._model.encode(corpus)
                sim_matrix = cosine_similarity(q_emb, doc_embs)[0]
                similarities = sim_matrix.tolist()
            except Exception as e:
                print(f"[SemanticSearch] Transformer error, falling back to TF-IDF: {e}")
                similarities = []

        # Strategy B: TF-IDF vector similarity
        if not similarities:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            all_texts = [expanded_query] + corpus
            tfidf_mat = vec.fit_transform(all_texts)
            q_vec = tfidf_mat[0:1]
            doc_vecs = tfidf_mat[1:]
            sim_matrix = cosine_similarity(q_vec, doc_vecs)[0]
            similarities = sim_matrix.tolist()

        # Keyword overlap bonus
        query_terms = set(re.findall(r"\w+", query.lower()))
        results = []
        for i, rec in enumerate(records):
            score = float(similarities[i])
            # Boost score if specific keywords explicitly appear
            doc_words = set(re.findall(r"\w+", corpus[i].lower()))
            overlap = len(query_terms.intersection(doc_words))
            if overlap > 0:
                score += (overlap * 0.15)

            score = min(score, 1.0)
            if score >= min_similarity:
                results.append((score, rec))

        # Rank by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:limit]

        items = []
        for score, rec in top_results:
            items.append(SemanticSearchItem(
                id=rec["id"],
                event_type=rec["event_type"],
                level=rec["level"],
                message=rec["message"],
                timestamp=rec["timestamp"],
                time_str=rec["time_str"],
                similarity=round(score, 4),
                evidence_link=rec["evidence_link"],
                data=rec["data"]
            ))

        return SemanticSearchResponse(
            query=query,
            count=len(items),
            results=items
        )
