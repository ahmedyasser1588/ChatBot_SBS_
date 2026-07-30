"""
data_preparation.py
--------------------
بيحول players.json لفقرات نصية، يعمل لها embeddings، ويخزنها في ChromaDB — بس
دلوقتي بيحتفظ كمان بميتاداتا حقيقية (sport, club, name, position) جنب كل لاعب،
عشان نقدر نعمل فلترة دقيقة (metadata filter) قبل البحث الدلالي، مش بس نعتمد
على الـ embedding similarity وحده اللي بيدوب فيه اسم النادي وسط باقي الأرقام.

يعني الاسترجاع بقى Hybrid: فلترة حقيقية (لو السؤال فيه اسم نادي/رياضة واضح) +
بحث دلالي فوقها. ده أفضل بكتير من semantic search وحده على بيانات structured.
"""

import json
import os
import uuid
from pathlib import Path

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

DATA_JSON_PATH = "data/players.json"
TEXT_FILES_DIR = "data/text_files"
VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME = "spotme_players"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

SPORT_AR = {
    "football": "كرة القدم",
    "basketball": "كرة السلة",
    "handball": "كرة اليد",
    "volleyball": "الكرة الطائرة",
}


def player_to_text(p: dict, sport_key: str) -> str:
    sport_ar = SPORT_AR.get(sport_key, sport_key)
    name = p.get("name", "غير معروف")

    parts = [
        f"اللاعب {name}، رقمه التعريفي {p.get('player_id')}.",
        f"يلعب رياضة {sport_ar} في مركز {p.get('position')}.",
        f"عمره {p.get('age')} سنة، طوله {p.get('height_cm')} سم، ووزنه {p.get('weight_kg')} كجم.",
    ]
    if p.get("preferred_foot"):
        parts.append(f"قدمه المفضلة هي {p.get('preferred_foot')}.")
    parts.append(f"ناديه الحالي هو {p.get('current_club')}.")
    parts.append(f"درجة الذكاء الاصطناعي (AI Score) الخاصة به هي {p.get('ai_score')} من 100.")
    parts.append(
        f"عدد إصاباته خلال آخر سنتين هو {p.get('injuries_last_2y')}، "
        f"ونسبة تعافيه {p.get('recovery_percentage')}%."
    )
    parts.append(f"نسبة تحسنه الشهري هي {p.get('monthly_improvement_pct')}%.")
    if p.get("speed_kmh") is not None:
        parts.append(f"سرعته القصوى {p.get('speed_kmh')} كم/س.")
    if p.get("pass_accuracy_pct") is not None:
        parts.append(f"دقة تمريراته {p.get('pass_accuracy_pct')}%.")
    if p.get("shot_accuracy_pct") is not None:
        parts.append(f"دقة تسديداته {p.get('shot_accuracy_pct')}%.")
    parts.append(f"عدد مشاهدات ملفه الشخصي خلال آخر أسبوع {p.get('profile_views_last_week')}.")

    return " ".join(parts)


def load_players():
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_files(data):
    """بنكتب كمان نسخة .txt لكل رياضة للمراجعة اليدوية، زي الأصل بالظبط."""
    os.makedirs(TEXT_FILES_DIR, exist_ok=True)
    for sport_key, players in data.items():
        lines = [player_to_text(p, sport_key) for p in players]
        out_path = Path(TEXT_FILES_DIR) / f"{sport_key}_players.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote {out_path} ({len(players)} players)")


def build_chunks_with_metadata(data):
    """كل لاعب = chunk واحد، ومعاه ميتاداتا حقيقية (sport/club/name/position)
    عشان نقدر نفلتر عليها exact-match وقت الاسترجاع."""
    all_chunks = []
    for sport_key, players in data.items():
        for p in players:
            all_chunks.append({
                "text": player_to_text(p, sport_key),
                "sport": sport_key,
                "club": p.get("current_club", ""),
                "name": p.get("name", ""),
                "position": p.get("position", ""),
                "source_file": f"{sport_key}_players.txt",
            })
    print(f"Total chunks (= total players): {len(all_chunks)}")
    return all_chunks


def embed_and_store(all_chunks):
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print("Embeddings generated. Shape:", embeddings.shape)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

    # نمسح الكوليكشن القديمة لو موجودة (عشان مفيش تكرار لو شغلت السكريبت أكتر من مرة)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection to rebuild it fresh.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "SpotMe player profiles embeddings",
            "hnsw:space": "cosine",
        },
    )
    print("Collection ready. Current count:", collection.count())

    ids = [f"player_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(all_chunks))]
    metadatas = [
        {
            "source_file": c["source_file"],
            "content_length": len(c["text"]),
            "sport": c["sport"],
            "club": c["club"],
            "name": c["name"],
            "position": c["position"],
        }
        for c in all_chunks
    ]

    normalized_embeddings = []
    for vector in embeddings:
        arr = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        normalized_embeddings.append(arr.tolist())

    collection.add(
        ids=ids,
        embeddings=normalized_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print("Stored", len(ids), "player chunks in the vector store.")
    print("Final count:", collection.count())


if __name__ == "__main__":
    print("1) Loading players.json")
    data = load_players()

    print("\n2) Writing per-sport text files (for manual review)")
    build_text_files(data)

    print("\n3) Building chunks with metadata (sport/club/name/position)")
    chunks = build_chunks_with_metadata(data)

    print("\n4) Embedding + storing in ChromaDB")
    embed_and_store(chunks)

    print("\nDone. Vector store ready at:", VECTOR_STORE_DIR)
