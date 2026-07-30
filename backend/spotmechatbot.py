import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.llm import GroqLLM

SPORT_KEYWORDS = {
    "football": ["كرة قدم", "كورة قدم", "كورة", "فوتبول", "football"],
    "basketball": ["كرة سلة", "كورة سلة", "باسكت", "basketball"],
    "handball": ["كرة يد", "كورة يد", "handball"],
    "volleyball": ["كرة طائرة", "فوليبول", "volleyball"],
}


def _normalize_arabic(text: str) -> str:
    """بيوحد أشكال الهمزة والألف المختلفة (أ/إ/آ -> ا) ويشيل التشكيل، عشان
    مطابقة أسماء الأندية تنجح حتى لو المستخدم كتبها بشكل مختلف شوية."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    return text


class SpotMeChatbot:
    """نفس الكلاس بالظبط WorldCupChatbot الأصلي (retriever + llm + session history)،
    بس هنا بيعمل استرجاع Hybrid: لو السؤال فيه اسم نادي أو رياضة واضح، بيبني
    metadata filter (where) حقيقي قبل البحث الدلالي، بدل ما يعتمد على الـ
    embedding similarity وحده اللي بيدوب فيه اسم النادي وسط باقي أرقام اللاعب."""

    def __init__(self, retriever, llm: GroqLLM, players_json_path: str = "data/players.json"):
        self.retriever = retriever
        self.llm = llm
        self.histories: Dict[str, List[Dict[str, str]]] = {}
        self._club_lookup = self._build_club_lookup(players_json_path)

    @staticmethod
    def _build_club_lookup(players_json_path: str) -> Dict[str, str]:
        """بيبني قاموس: (اسم النادي بعد التطبيع) -> (اسم النادي الأصلي زي ما
        مخزن في الميتاداتا)، عشان نقدر نطابق أي صيغة كتابة مع القيمة الصح."""
        path = Path(players_json_path)
        lookup: Dict[str, str] = {}
        if not path.exists():
            return lookup
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return lookup

        clubs = set()
        for players in data.values():
            for p in players:
                club = p.get("current_club")
                if club:
                    clubs.add(club)

        for club in clubs:
            lookup[_normalize_arabic(club)] = club
        return lookup

    def _detect_filters(self, question: str) -> Optional[dict]:
        normalized_q = _normalize_arabic(question)

        club_match = None
        for normalized_club, original_club in self._club_lookup.items():
            if normalized_club and normalized_club in normalized_q:
                club_match = original_club
                break

        sport_match = None
        for sport_key, keywords in SPORT_KEYWORDS.items():
            if any(_normalize_arabic(kw) in normalized_q for kw in keywords):
                sport_match = sport_key
                break

        conditions = []
        if club_match:
            conditions.append({"club": club_match})
        if sport_match:
            conditions.append({"sport": sport_match})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def chat(self, session_id: str, question: str) -> Dict[str, Any]:
        history = self.histories.setdefault(session_id, [])

        where = self._detect_filters(question)
        results = self.retriever.retrieve(question, top_k=8, where=where)

        # لو الفلترة رجعت مفيش نتايج (اسم اتعرف بس مفيش لاعبين مطابقين فعلاً)،
        # نرجع نجرب بحث دلالي عادي من غير فلتر بدل ما نرجع سياق فاضي بالكامل.
        if where and not results:
            results = self.retriever.retrieve(question, top_k=8, where=None)

        context = "\n".join([doc["content"] for doc in results]) if results else "لا يوجد سياق مطابق."

        response_text = self.llm.invoke(question, context, history=history)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response_text})

        def clean_source(raw_source_file: str) -> str:
            name = raw_source_file.rsplit(".", 1)[0]
            if name.endswith("_players"):
                name = name[: -len("_players")]
            return name

        sources = [
            {
                "source_file": clean_source(doc.get("metadata", {}).get("source_file", "unknown")),
                "content_preview": doc.get("content", "")[:120],
            }
            for doc in results
        ]

        return {"answer": response_text, "sources": sources}

    def reset(self, session_id: str):
        self.histories.pop(session_id, None)
        print(f"History cleared for session {session_id}.")
