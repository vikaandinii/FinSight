import re
import numpy as np
from typing import List, Dict, Any
from .embedder import get_embedding_model


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def rerank_documents(
    query_info: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Melakukan Reranking terhadap kandidat dokumen hasil Hybrid Search (BM25 + ChromaDB):
    1. Semantic Similarity langsung antara pertanyaan asli dan potongan teks/judul dokumen
    2. Exact Account/Keyword Overlap pada baris tabel atau topik pengungkapan
    3. Kesesuaian Sheet Prioritas (Target Sheets dari Query Understanding)
    4. Kesesuaian Jenis Dokumen (financial_table vs narrative_disclosure) & Periode (Tahunan/Kuartal)
    """
    if not candidates:
        return []

    original_query = query_info.get("original_query", "")
    target_sheets = set(query_info.get("target_sheets", []))
    matched_keywords = query_info.get("matched_keywords", [])
    preferred_doc_type = query_info.get("preferred_doc_type", "financial_table")
    explicit_quarter = query_info.get("quarter")

    q_tokens = set(tokenize_simple(original_query))

    # Hitung semantic similarity fokus pada header + 1000 karakter pertama tiap kandidat
    try:
        model = get_embedding_model()
        q_vec = model.encode([original_query], normalize_embeddings=True)[0]
        focused_texts = [
            f"{c['metadata'].get('sheet_title_id', '')} {c['content'][:1000]}"
            for c in candidates
        ]
        cand_vecs = model.encode(focused_texts, normalize_embeddings=True, show_progress_bar=False)
        sem_sims = np.dot(cand_vecs, q_vec)
    except Exception:
        sem_sims = np.zeros(len(candidates))

    reranked = []
    for idx, item in enumerate(candidates):
        meta = item.get("metadata", {})
        content_lower = item.get("content", "").lower()
        sheet_code = str(meta.get("sheet_code", ""))
        doc_type = str(meta.get("doc_type", ""))
        quarter_val = str(meta.get("quarter", ""))

        base_rrf = float(item.get("hybrid_score", 0.0))
        sem_score = float(sem_sims[idx]) if idx < len(sem_sims) else 0.0

        # 1. Bonus kesesuaian Sheet dengan Intent Pertanyaan
        sheet_bonus = 0.0
        if sheet_code in target_sheets:
            sheet_bonus += 0.25
        elif any(sheet_code.startswith(ts) for ts in target_sheets):
            sheet_bonus += 0.15

        # 2. Bonus kecocokan frasa akuntansi eksak di dalam isi dokumen
        kw_bonus = 0.0
        for kw in matched_keywords:
            if kw in content_lower:
                kw_bonus += 0.08
        kw_bonus = min(kw_bonus, 0.24)

        # 3. Token overlap rasio
        doc_tokens = set(tokenize_simple(content_lower))
        overlap_ratio = len(q_tokens.intersection(doc_tokens)) / max(len(q_tokens), 1)
        lexical_bonus = 0.10 * overlap_ratio

        # 4. Bonus kesesuaian tipe dokumen dan periode
        type_period_bonus = 0.0
        if doc_type == preferred_doc_type:
            type_period_bonus += 0.08
        if not explicit_quarter and quarter_val == "Tahunan":
            type_period_bonus += 0.06
        elif explicit_quarter and quarter_val.upper() == explicit_quarter.upper():
            type_period_bonus += 0.10

        rerank_score = (base_rrf * 2.0) + (0.30 * max(sem_score, 0.0)) + sheet_bonus + kw_bonus + lexical_bonus + type_period_bonus

        updated_item = dict(item)
        updated_item["rerank_score"] = round(float(rerank_score), 5)
        reranked.append(updated_item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
