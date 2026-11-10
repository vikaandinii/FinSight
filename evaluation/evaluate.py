import os
import re
import sys
from typing import List, Dict, Any, Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from retrieval import search_financial_docs, get_raw_documents
from generation import answer_financial_query


# Benchmark Dataset Evaluasi Laporan Keuangan BEI (BBCA, BBRI, BBNI)
EVAL_BENCHMARK = [
    {
        "query": "Berapa laba bersih BBCA tahun 2024?",
        "ticker": "BBCA",
        "year": "2024",
        "quarter": "Tahunan",
        "relevant_sheets": {"4322000", "4312000"}
    },
    {
        "query": "Berapa total aset dan liabilitas BBCA pada laporan tahunan 2024?",
        "ticker": "BBCA",
        "year": "2024",
        "quarter": "Tahunan",
        "relevant_sheets": {"4220000"}
    },
    {
        "query": "Berapa laba bersih BBRI tahun 2024?",
        "ticker": "BBRI",
        "year": "2024",
        "quarter": "Tahunan",
        "relevant_sheets": {"4312000", "4322000"}
    },
    {
        "query": "Berapa total ekuitas dan kredit BBNI tahun 2024?",
        "ticker": "BBNI",
        "year": "2024",
        "quarter": "Tahunan",
        "relevant_sheets": {"4220000", "4410000"}
    },
    {
        "query": "Berapa rasio ROA dan ROE BBCA tahun 2023?",
        "ticker": "BBCA",
        "year": "2023",
        "quarter": "Tahunan",
        "relevant_sheets": {"4220000", "4322000"}
    }
]


def is_doc_relevant(doc_meta: Dict[str, Any], ground_truth: Dict[str, Any]) -> bool:
    """Memeriksa apakah dokumen hasil retrieval relevan terhadap ground truth (Ticker, Tahun, dan Sheet)."""
    if doc_meta.get("ticker", "").upper() != ground_truth["ticker"].upper():
        return False
    if str(doc_meta.get("year", "")) != str(ground_truth["year"]):
        return False
    sh = str(doc_meta.get("sheet_code", ""))
    return sh in ground_truth["relevant_sheets"]


def compute_precision_recall_at_k(
    retrieved_docs: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    all_docs: List[Dict[str, Any]],
    k: int = 3
) -> Dict[str, float]:
    """
    Menghitung Precision@K dan Recall@K:
    - Precision@K = (Jumlah dokumen relevan di Top-K) / K
    - Recall@K    = (Jumlah dokumen relevan di Top-K) / (Total dokumen relevan untuk ticker+tahun+sheet tersebut)
    """
    top_k_docs = retrieved_docs[:k]
    relevant_retrieved = sum(
        1 for d in top_k_docs if is_doc_relevant(d.get("metadata", {}), ground_truth)
    )

    # Hitung total dokumen relevan di korpus untuk kombinasi (ticker, year, quarter, relevant_sheets)
    total_relevant_in_corpus = 0
    for d in all_docs:
        m = d.get("metadata", {})
        if m.get("ticker", "").upper() == ground_truth["ticker"].upper() and str(m.get("year", "")) == str(ground_truth["year"]):
            if ground_truth.get("quarter") and str(m.get("quarter", "")).upper() != ground_truth["quarter"].upper():
                continue
            if str(m.get("sheet_code", "")) in ground_truth["relevant_sheets"]:
                total_relevant_in_corpus += 1

    total_relevant_in_corpus = max(total_relevant_in_corpus, 1)

    # Untuk Precision@K pada kasus single/few-sheet target, dibatasi dengan min(k, total_relevant_possible)
    # serta disediakan raw_precision_at_k (relevant_retrieved / k)
    raw_precision = relevant_retrieved / float(k)
    normalized_precision = min(relevant_retrieved / float(min(k, total_relevant_in_corpus)), 1.0)
    recall = min(relevant_retrieved / float(total_relevant_in_corpus), 1.0)

    return {
        "precision_at_k": round(normalized_precision, 4),
        "raw_precision_at_k": round(raw_precision, 4),
        "recall_at_k": round(recall, 4),
        "relevant_found": relevant_retrieved,
        "total_relevant": total_relevant_in_corpus
    }


def compute_faithfulness(answer: str, structured_context: str, retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    Menghitung skor Faithfulness (0.0 - 1.0):
    Memverifikasi apakah seluruh klaim angka/persentase, emiten, dan sitasi Sheet di dalam jawaban
    benar-benar didukung oleh konteks hasil retrieval & metrik terstruktur (bebas halusinasi).
    """
    if not answer:
        return 0.0

    combined_context = (structured_context or "") + "\n" + "\n".join(
        f"{d.get('metadata', {})} {d.get('content', '')}" for d in retrieved_docs
    )
    context_lower = combined_context.lower()

    # Ekstrak klaim angka penting (Triliun, Miliar, Persentase, dan Kode Sheet) dari bagian jawaban utama
    main_answer = answer.split("Bukti Konteks Dokumen")[0]
    claims = re.findall(r"(?:rp\s*\d+(?:[.,]\d+)?\s*(?:triliun|miliar)|\d+(?:[.,]\d+)?%|sheet\s*\d{7})", main_answer.lower())

    if not claims:
        return 1.0

    supported = 0
    for c in claims:
        c_clean = re.sub(r"\s+", " ", c.strip())
        # Cek apakah angka/sheet ada di dalam konteks
        num_part = re.findall(r"\d+(?:[.,]\d+)?", c_clean)
        if num_part and num_part[0] in context_lower:
            supported += 1

    return round(supported / float(len(claims)), 4)


def run_evaluation(k: int = 3):
    """Menjalankan evaluasi menyeluruh Precision@K, Recall@K, dan Faithfulness."""
    all_docs = get_raw_documents()
    print("=" * 80)
    print(f"EVALUASI SISTEM FINSIGHT RAG (K = {k})")
    print("Metrik: Precision@K, Recall@K, dan Faithfulness (Anti-Hallucination)")
    print("=" * 80)

    precisions = []
    raw_precisions = []
    recalls = []
    faithfulness_scores = []

    for idx, item in enumerate(EVAL_BENCHMARK, 1):
        query = item["query"]
        res = answer_financial_query(query)
        retrieved_docs = res.get("retrieved_docs", [])
        answer = res.get("answer", "")
        structured_ctx = res.get("structured_context", "")

        pr_metrics = compute_precision_recall_at_k(retrieved_docs, item, all_docs, k=k)
        faith_score = compute_faithfulness(answer, structured_ctx, retrieved_docs)

        precisions.append(pr_metrics["precision_at_k"])
        raw_precisions.append(pr_metrics["raw_precision_at_k"])
        recalls.append(pr_metrics["recall_at_k"])
        faithfulness_scores.append(faith_score)

        top_sheets = [f"{d['metadata'].get('ticker')}_{d['metadata'].get('quarter')}_{d['metadata'].get('sheet_code')}" for d in retrieved_docs]
        print(f"\n[{idx}] Query          : {query}")
        print(f"    Target Sheets  : {sorted(list(item['relevant_sheets']))} ({item['ticker']} {item['year']} {item['quarter']})")
        print(f"    Retrieved Top-{k}: {top_sheets}")
        print(f"    Precision@{k}   : {pr_metrics['precision_at_k']:.2f} (Raw: {pr_metrics['raw_precision_at_k']:.2f}) | Recall@{k}: {pr_metrics['recall_at_k']:.2f} | Faithfulness: {faith_score:.2f}")

    avg_p = sum(precisions) / len(precisions)
    avg_raw_p = sum(raw_precisions) / len(raw_precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f = sum(faithfulness_scores) / len(faithfulness_scores)

    print("\n" + "=" * 80)
    print("RINGKASAN HASIL EVALUASI RATA-RATA:")
    print(f" - Mean Normalized Precision@{k} : {avg_p:.4f} ({avg_p * 100:.1f}%)")
    print(f" - Mean Raw Precision@{k}        : {avg_raw_p:.4f} ({avg_raw_p * 100:.1f}%)")
    print(f" - Mean Recall@{k}               : {avg_r:.4f} ({avg_r * 100:.1f}%)")
    print(f" - Mean Faithfulness             : {avg_f:.4f} ({avg_f * 100:.1f}%)")
    print("=" * 80)

    return {
        "precision_at_k": round(avg_p, 4),
        "raw_precision_at_k": round(avg_raw_p, 4),
        "recall_at_k": round(avg_r, 4),
        "faithfulness": round(avg_f, 4)
    }


if __name__ == "__main__":
    run_evaluation(k=3)
