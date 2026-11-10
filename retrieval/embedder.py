import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

# Konfigurasi backend ML: paksa PyTorch & mode offline HuggingFace agar cepat dan stabil
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "financial_rag_data.json")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "data", "processed", "embeddings_minilm.npy")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_RAW_DOCUMENTS = None
_EMBEDDING_MODEL = None
_DOC_EMBEDDINGS = None


def get_raw_documents() -> List[Dict[str, Any]]:
    """Memuat seluruh dokumen chunks dari file JSON hasil ingestion."""
    global _RAW_DOCUMENTS
    if _RAW_DOCUMENTS is None:
        if os.path.exists(JSON_DATA_PATH):
            with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
                _RAW_DOCUMENTS = json.load(f)
        else:
            _RAW_DOCUMENTS = []
    return _RAW_DOCUMENTS


def get_embedding_model():
    """Memuat model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", local_files_only=True)
        except Exception:
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _EMBEDDING_MODEL


def get_doc_embeddings() -> np.ndarray:
    """Mengambil atau menghitung matriks embedding vektor untuk seluruh chunks."""
    global _DOC_EMBEDDINGS
    if _DOC_EMBEDDINGS is not None:
        return _DOC_EMBEDDINGS

    docs = get_raw_documents()
    if not docs:
        return np.empty((0, 384))

    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            cached = np.load(EMBEDDINGS_CACHE_PATH)
            if cached.shape[0] == len(docs):
                _DOC_EMBEDDINGS = cached
                return _DOC_EMBEDDINGS
        except Exception:
            pass

    print(f"[RETRIEVAL] Menghitung embedding untuk {len(docs)} dokumen menggunakan {EMBEDDING_MODEL_NAME}...")
    model = get_embedding_model()
    texts = [d["page_content"] for d in docs]
    _DOC_EMBEDDINGS = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH), exist_ok=True)
    np.save(EMBEDDINGS_CACHE_PATH, _DOC_EMBEDDINGS)
    print(f"[RETRIEVAL] Embedding berhasil dihitung dan disimpan di '{EMBEDDINGS_CACHE_PATH}'.")
    return _DOC_EMBEDDINGS


def reload_embeddings_cache():
    """Reset cache di memori dan hitung ulang embedding jika ada dokumen baru dari Admin."""
    global _RAW_DOCUMENTS, _DOC_EMBEDDINGS
    _RAW_DOCUMENTS = None
    _DOC_EMBEDDINGS = None
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            os.remove(EMBEDDINGS_CACHE_PATH)
        except Exception:
            pass
    return get_doc_embeddings()


if __name__ == "__main__":
    docs = get_raw_documents()
    emb = get_doc_embeddings()
    print("=" * 70)
    print(f"MODEL EMBEDDING : {EMBEDDING_MODEL_NAME}")
    print(f"FILE VEKTOR     : {EMBEDDINGS_CACHE_PATH}")
    print(f"UKURAN MATRIKS  : {emb.shape} ({emb.shape[0]} dokumen x {emb.shape[1]} dimensi)")
    print("=" * 70)
    for i in range(min(3, len(docs))):
        d = docs[i]
        m = d.get("metadata", {})
        print(f"\n[Dokumen #{i+1}] ID: {d['id']}")
        print(f"  Emiten & Periode : {m.get('ticker')} | {m.get('year')} {m.get('quarter')} ({m.get('period')})")
        print(f"  Sheet            : [{m.get('sheet_code')}] {m.get('sheet_title_id')}")
        snippet = d.get("page_content", "").replace("\n", " ")[:120]
        print(f"  Cuplikan Teks    : {snippet}...")
        print(f"  Vektor (10 dim)  : {np.round(emb[i][:10], 4)}")

