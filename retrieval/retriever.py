import os
import re
import sys
import numpy as np
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from .embedder import (
        EMBEDDING_MODEL_NAME,
        get_raw_documents,
        get_embedding_model,
        get_doc_embeddings,
    )
    from .vector_store import get_chroma_collection
    from .query_understanding import analyze_query
    from .reranker import rerank_documents
except ImportError:
    from retrieval.embedder import (
        EMBEDDING_MODEL_NAME,
        get_raw_documents,
        get_embedding_model,
        get_doc_embeddings,
    )
    from retrieval.vector_store import get_chroma_collection
    from retrieval.query_understanding import analyze_query
    from retrieval.reranker import rerank_documents

PRIMARY_SHEETS = {"4220000", "4312000", "4322000", "4410000", "4410000PY", "4510000", "4520000"}


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def search_bm25(query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 15) -> List[Dict[str, Any]]:
    """Pencarian leksikal cepat berbasis BM25."""
    if not candidate_docs:
        return []
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [tokenize(doc.get("page_content", "")) for doc in candidate_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        q_tokens = tokenize(query)
        scores = bm25.get_scores(q_tokens)

        scored_docs = []
        for doc, score in zip(candidate_docs, scores):
            scored_docs.append({"doc": doc, "score": float(score)})
    except Exception:
        q_tokens = set(tokenize(query))
        scored_docs = []
        for doc in candidate_docs:
            doc_tokens = set(tokenize(doc.get("page_content", "")))
            overlap = len(q_tokens.intersection(doc_tokens))
            scored_docs.append({"doc": doc, "score": float(overlap)})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]


def search_vector(
    query: str,
    candidate_docs: List[Dict[str, Any]],
    all_docs: List[Dict[str, Any]],
    doc_matrix: np.ndarray,
    top_k: int = 15,
    ticker: Optional[str] = None,
    year: Optional[str] = None,
    quarter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Pencarian kemiripan kosinus semantik menggunakan ChromaDB & model MiniLM."""
    if not candidate_docs:
        return []

    augmented_query = query
    if ticker and ticker.upper() not in query.upper():
        augmented_query = f"{augmented_query} ticker {ticker}"
    if year and str(year) not in query:
        augmented_query = f"{augmented_query} year {year}"
    if quarter and quarter.upper() not in query.upper():
        augmented_query = f"{augmented_query} quarter {quarter}"

    model = get_embedding_model()
    q_vec = model.encode([augmented_query], normalize_embeddings=True)[0]

    collection = get_chroma_collection()
    if collection is not None:
        try:
            conditions = []
            if ticker:
                conditions.append({"ticker": ticker.upper()})
            if year:
                conditions.append({"year": str(year)})
            if quarter:
                conditions.append({"quarter": quarter})

            where_filter = None
            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}

            res = collection.query(
                query_embeddings=[q_vec.tolist()],
                n_results=min(top_k, len(candidate_docs)),
                where=where_filter
            )
            doc_map = {d["id"]: d for d in candidate_docs}
            scored_docs = []
            if res and res.get("ids") and res["ids"][0]:
                ids = res["ids"][0]
                distances = res.get("distances", [[]])[0]
                for doc_id, dist in zip(ids, distances):
                    if doc_id in doc_map:
                        scored_docs.append({"doc": doc_map[doc_id], "score": float(1.0 - dist)})
            if scored_docs:
                return scored_docs
        except Exception as e:
            print(f"[CHROMADB WARN] Query fallback: {e}")

    if doc_matrix.size == 0:
        return []
    candidate_indices = [all_docs.index(d) for d in candidate_docs if d in all_docs]
    if not candidate_indices:
        candidate_indices = list(range(len(candidate_docs)))

    sub_matrix = doc_matrix[candidate_indices]
    sims = np.dot(sub_matrix, q_vec)

    scored_docs = []
    for doc, sim in zip(candidate_docs, sims):
        scored_docs.append({"doc": doc, "score": float(sim)})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]


def extract_metadata_from_query(query: str):
    """Deteksi otomatis filter metadata (ticker, year, quarter) dari pertanyaan user."""
    info = analyze_query(query)
    return info["ticker"], info["year"], info["quarter"]


def search_financial_docs(
    query: str,
    ticker: Optional[str] = None,
    year: Optional[str] = None,
    quarter: Optional[str] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Pipeline Pencarian Lengkap:
    1. Query Understanding (Intent, Target Sheets, Metadata, Query Expansion)
    2. Metadata Filtering (Ticker, Tahun, Kuartal)
    3. Hybrid Search (BM25 + Dense MiniLM Vector via ChromaDB) + RRF
    4. Cross-Feature Reranking untuk mengurutkan dokumen paling relevan di posisi teratas
    """
    all_docs = get_raw_documents()
    if not all_docs:
        return []

    # 1. Query Understanding
    q_info = analyze_query(query, ticker=ticker, year=year, quarter=quarter)
    eff_ticker = q_info["ticker"]
    eff_year = q_info["year"]
    eff_quarter = q_info["quarter"]
    target_sheets = set(q_info["target_sheets"])
    expanded_query = q_info["expanded_query"]

    # 2. Filter awal metadata (Ticker, Tahun, Kuartal)
    candidate_docs = []
    for d in all_docs:
        m = d.get("metadata", {})
        if eff_ticker and m.get("ticker", "").upper() != eff_ticker.upper():
            continue
        if eff_year and str(m.get("year", "")) != str(eff_year):
            continue
        if eff_quarter and str(m.get("quarter", "")).upper() != eff_quarter.upper():
            continue
        candidate_docs.append(d)

    if not candidate_docs:
        candidate_docs = all_docs

    pool_size = max(top_k * 5, 20)

    # 3. BM25 Search menggunakan expanded_query
    bm25_hits = search_bm25(expanded_query, candidate_docs, top_k=pool_size)
    bm25_ranks = {h["doc"]["id"]: rank for rank, h in enumerate(bm25_hits, 1)}

    # 4. Dense Vector Search (MiniLM + ChromaDB)
    try:
        doc_matrix = get_doc_embeddings()
        vector_hits = search_vector(
            expanded_query, candidate_docs, all_docs, doc_matrix,
            top_k=pool_size, ticker=eff_ticker, year=eff_year, quarter=eff_quarter
        )
        vector_ranks = {h["doc"]["id"]: rank for rank, h in enumerate(vector_hits, 1)}
    except Exception as e:
        print(f"[RETRIEVAL WARN] Vector search fallback: {e}")
        vector_ranks = {}

    # 5. RRF (Reciprocal Rank Fusion) + Intent Sheet Prioritization
    doc_map = {d["id"]: d for d in candidate_docs}
    combined_scores = {}
    all_ids = set(bm25_ranks.keys()).union(set(vector_ranks.keys()))
    for d in candidate_docs:
        sh = str(d.get("metadata", {}).get("sheet_code", ""))
        if sh in PRIMARY_SHEETS or sh in target_sheets:
            all_ids.add(d["id"])
    if not all_ids:
        all_ids = set([d["id"] for d in candidate_docs[:pool_size]])

    for doc_id in all_ids:
        doc = doc_map.get(doc_id)
        if not doc:
            continue

        b_rank = bm25_ranks.get(doc_id, 100)
        v_rank = vector_ranks.get(doc_id, 100)
        rrf = (1.0 / (60 + b_rank)) + (1.0 / (60 + v_rank))

        sheet_code = str(doc.get("metadata", {}).get("sheet_code", ""))
        quarter_val = str(doc.get("metadata", {}).get("quarter", ""))
        boost = 0.0
        if sheet_code in target_sheets:
            boost += 0.025
        elif sheet_code in PRIMARY_SHEETS:
            boost += 0.008

        if not eff_quarter and quarter_val == "Tahunan" and sheet_code in PRIMARY_SHEETS:
            boost += 0.005

        combined_scores[doc_id] = rrf + boost

    sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:pool_size]

    initial_results = []
    for doc_id, score in sorted_candidates:
        doc = doc_map[doc_id]
        initial_results.append({
            "id": doc["id"],
            "content": doc["page_content"],
            "metadata": doc["metadata"],
            "hybrid_score": round(score, 5),
            "query_understanding": q_info
        })

    # 6. Reranking
    reranked_results = rerank_documents(q_info, initial_results, top_k=top_k)
    return reranked_results


def run_interactive_retrieval():
    """Mode uji coba Retrieval di terminal: ketik pertanyaan -> tampilkan Query Understanding & Dokumen Reranked."""
    print("=" * 75)
    print("UJI COBA RETRIEVAL (QUERY UNDERSTANDING + HYBRID SEARCH + RERANKING)")
    print(f"Model Embedding : {EMBEDDING_MODEL_NAME}")
    print("Ketik pertanyaan untuk melihat dokumen/sheet mana yang ditemukan.")
    print("Ketik 'exit' atau 'keluar' untuk berhenti.")
    print("=" * 75)

    while True:
        try:
            q = input("\nMasukkan pertanyaan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSelesai.")
            break
        if not q:
            continue
        if q.lower() in ["exit", "quit", "keluar", "q"]:
            print("Selesai.")
            break

        q_info = analyze_query(q)
        print(f"   [Query Understanding]")
        print(f"   - Filter Metadata : Ticker={q_info['ticker'] or 'Semua'} | Tahun={q_info['year'] or 'Semua'} | Kuartal={q_info['quarter'] or 'Semua'}")
        print(f"   - Intent Terdeteksi: {', '.join(q_info['intents'])}")
        print(f"   - Sheet Prioritas  : {', '.join(q_info['target_sheets'])}")

        hits = search_financial_docs(q, top_k=3)
        print(f"\nDitemukan {len(hits)} Dokumen Teratas (Setelah Reranking):")
        print("-" * 75)
        for idx, h in enumerate(hits, 1):
            m = h["metadata"]
            snippet = h["content"].strip().replace("\n", " ")
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            print(f"#{idx} | ID Dokumen   : {h['id']}")
            print(f"   | Asal Sheet   : [{m.get('sheet_code')}] {m.get('sheet_title_id')}")
            print(f"   | Metadata     : Emiten={m.get('ticker')} | Periode={m.get('year')} {m.get('quarter')} ({m.get('period')}) | Tipe={m.get('doc_type')}")
            print(f"   | Skor Hybrid  : {h['hybrid_score']} | Skor Rerank: {h.get('rerank_score')}")
            print(f"   | Cuplikan     : {snippet}")
            print("-" * 75)


if __name__ == "__main__":
    run_interactive_retrieval()
