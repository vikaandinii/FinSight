import os
import sys
import types
from typing import Optional
from .embedder import BASE_DIR, get_raw_documents, get_doc_embeddings

# Stub opentelemetry exporter agar ChromaDB berjalan lancar tanpa dependensi eksternal telemetry
if "opentelemetry.exporter.otlp.proto.grpc.trace_exporter" not in sys.modules:
    for mod_name in [
        "opentelemetry.exporter",
        "opentelemetry.exporter.otlp",
        "opentelemetry.exporter.otlp.proto",
        "opentelemetry.exporter.otlp.proto.grpc",
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    ]:
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    sys.modules["opentelemetry.exporter.otlp.proto.grpc.trace_exporter"].OTLPSpanExporter = object

CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None


def get_chroma_collection():
    """Membuat atau memuat koleksi Vector Database ChromaDB di data/chroma_db."""
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    try:
        import chromadb
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
            name="financial_docs",
            metadata={"hnsw:space": "cosine"}
        )
        docs = get_raw_documents()
        if docs and _CHROMA_COLLECTION.count() < len(docs):
            print(f"[CHROMADB] Menyimpan {len(docs)} dokumen & vektor ke ChromaDB ({CHROMA_DB_PATH})...")
            ids = [doc["id"] for doc in docs]
            embeddings = get_doc_embeddings().tolist()
            metadatas = [doc.get("metadata", {}) for doc in docs]
            contents = [doc.get("page_content", "") for doc in docs]

            batch_size = 100
            for i in range(0, len(docs), batch_size):
                _CHROMA_COLLECTION.upsert(
                    ids=ids[i:i + batch_size],
                    documents=contents[i:i + batch_size],
                    embeddings=embeddings[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                )
            print(f"[CHROMADB] Berhasil menyimpan {_CHROMA_COLLECTION.count()} item di ChromaDB.")
        return _CHROMA_COLLECTION
    except Exception as e:
        print(f"[RETRIEVAL WARN] Gagal memuat ChromaDB: {e}")
        return None


def sync_chroma_collection():
    """Sinkronisasi ulang dokumen & vektor terbaru ke ChromaDB."""
    global _CHROMA_COLLECTION
    _CHROMA_COLLECTION = None
    return get_chroma_collection()
