import re
from typing import Dict, Any, Optional, List

# Pemetaan intent keuangan ke kata kunci dan sheet prioritas BEI/IDX
INTENT_PATTERNS = {
    "net_income": {
        "keywords": ["laba bersih", "labar bersih", "laba tahun berjalan", "rugi bersih", "kerugian", "profit", "net income", "laba atribusi", "laba", "labar", "keuntungan"],
        "target_sheets": ["4312000", "4322000"],
        "metric_key": "laba_bersih_berjalan",
        "label_id": "Laba Bersih Tahun Berjalan"
    },
    "income_statement": {
        "keywords": ["pendapatan", "beban", "bunga bersih", "laba sebelum pajak", "laba operasional", "laba rugi", "penghasilan komprehensif"],
        "target_sheets": ["4312000", "4322000"],
        "metric_key": "pendapatan_bunga_bersih",
        "label_id": "Laporan Laba Rugi"
    },
    "total_assets": {
        "keywords": ["total aset", "jumlah aset", "aset", "assets", "harta"],
        "target_sheets": ["4220000"],
        "metric_key": "total_aset",
        "label_id": "Total Aset"
    },
    "liabilities": {
        "keywords": ["liabilitas", "kewajiban", "utang", "hutang", "dpk", "simpanan nasabah", "dana pihak ketiga", "giro", "tabungan", "deposito"],
        "target_sheets": ["4220000"],
        "metric_key": "total_liabilitas",
        "label_id": "Total Liabilitas & DPK"
    },
    "equity": {
        "keywords": ["ekuitas", "modal", "perubahan ekuitas", "saldo laba"],
        "target_sheets": ["4220000", "4410000", "4410000PY"],
        "metric_key": "total_ekuitas",
        "label_id": "Total Ekuitas"
    },
    "loans": {
        "keywords": ["kredit", "pinjaman", "pembiayaan", "loans"],
        "target_sheets": ["4220000"],
        "metric_key": "pinjaman_kredit",
        "label_id": "Pinjaman / Kredit yang Diberikan"
    },
    "cash_flow": {
        "keywords": ["arus kas", "kas dan setara kas", "cash flow", "aktivitas operasi", "aktivitas investasi", "aktivitas pendanaan"],
        "target_sheets": ["4510000", "4520000", "4220000"],
        "metric_key": "kas_dan_setara_kas",
        "label_id": "Arus Kas & Setara Kas"
    },
    "financial_ratios": {
        "keywords": ["rasio", "roa", "roe", "ldr", "nim", "npl", "car", "return on asset", "return on equity", "loan to deposit"],
        "target_sheets": ["4220000", "4312000", "4322000"],
        "metric_key": "ratios",
        "label_id": "Rasio Keuangan (ROA, ROE, LDR)"
    },
    "accounting_policy": {
        "keywords": ["kebijakan akuntansi", "ckpn", "cadangan kerugian", "penurunan nilai", "aset tetap", "pihak berelasi", "catatan", "calk", "pengungkapan"],
        "target_sheets": ["4610000", "4611100"],
        "metric_key": None,
        "label_id": "Catatan atas Laporan Keuangan (CALK)"
    }
}


def analyze_query(
    query: str,
    ticker: Optional[str] = None,
    year: Optional[str] = None,
    quarter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Menganalisis pertanyaan pengguna (Query Understanding):
    - Mendeteksi emiten (ticker), periode tahun, dan kuartal
    - Mengklasifikasikan maksud pertanyaan (intent: laba bersih, aset, liabilitas, ekuitas, rasio, arus kas, kebijakan akuntansi)
    - Menentukan kode sheet prioritas dan kata kunci ekspansi untuk Hybrid Retrieval & Reranking
    """
    q_lower = query.lower()
    q_upper = query.upper()

    # 1. Deteksi Ticker (Tunggal maupun Multi-Ticker untuk perbandingan)
    detected_tickers: List[str] = []
    if ticker:
        detected_tickers.append(ticker.upper())
    else:
        for t, pattern in [("BBCA", r"\b(bbca|bca)\b"), ("BBRI", r"\b(bbri|bri)\b"), ("BBNI", r"\b(bbni|bni)\b")]:
            if re.search(pattern, q_lower):
                detected_tickers.append(t)

    is_comparison = any(w in q_lower for w in [
        "banding", "perbandingan", "tertinggi", "terbesar", "terbanyak", "terendah", "terkecil",
        "paling tinggi", "paling besar", "bank apa", "bank mana", "emiten apa", "emiten mana", "vs", "versus", "antara"
    ]) or len(detected_tickers) > 1

    detected_ticker = detected_tickers[0] if len(detected_tickers) == 1 and not is_comparison else (ticker.upper() if ticker else None)
    if not is_comparison and not detected_ticker and len(detected_tickers) == 1:
        detected_ticker = detected_tickers[0]

    # 2. Deteksi Tahun
    detected_year = str(year) if year else None
    if not detected_year:
        m_year = re.search(r"\b(202[0-9])\b", query)
        if m_year:
            detected_year = m_year.group(1)

    # 3. Deteksi Kuartal / Periode
    detected_quarter = quarter
    if not detected_quarter:
        if any(w in q_upper for w in ["Q1", "TW1", "TRIWULAN 1", "TRIWULAN I", "KUARTAL 1", "KUARTAL I", "MARET"]):
            detected_quarter = "Q1"
        elif any(w in q_upper for w in ["Q2", "TW2", "TRIWULAN 2", "TRIWULAN II", "KUARTAL 2", "KUARTAL II", "JUNI", "SEMESTER 1"]):
            detected_quarter = "Q2"
        elif any(w in q_upper for w in ["Q3", "TW3", "TRIWULAN 3", "TRIWULAN III", "KUARTAL 3", "KUARTAL III", "SEPTEMBER"]):
            detected_quarter = "Q3"
        elif any(w in q_upper for w in ["TAHUNAN", "FULL YEAR", "FY", "AKHIR TAHUN", "DESEMBER"]):
            detected_quarter = "Tahunan"

    # 4. Deteksi Intent & Target Sheets
    detected_intents: List[str] = []
    target_sheets: List[str] = []
    matched_keywords: List[str] = []
    metric_keys: List[str] = []

    for intent_name, cfg in INTENT_PATTERNS.items():
        for kw in cfg["keywords"]:
            if kw in q_lower:
                if intent_name not in detected_intents:
                    detected_intents.append(intent_name)
                for sh in cfg["target_sheets"]:
                    if sh not in target_sheets:
                        target_sheets.append(sh)
                if kw not in matched_keywords:
                    matched_keywords.append(kw)
                if cfg["metric_key"] and cfg["metric_key"] not in metric_keys:
                    metric_keys.append(cfg["metric_key"])

    if not detected_intents:
        detected_intents = ["general_financial"]
        target_sheets = ["4220000", "4312000", "4322000"]

    # Menentukan jenis dokumen yang diprioritaskan
    preferred_doc_type = "narrative_disclosure" if "accounting_policy" in detected_intents and len(detected_intents) == 1 else "financial_table"

    # 5. Ekspansi Query untuk meningkatkan akurasi vektor dan BM25
    expansion_terms = list(matched_keywords)
    if "net_income" in detected_intents or "income_statement" in detected_intents:
        expansion_terms.extend(["laporan laba rugi", "jumlah laba rugi tahun berjalan"])
    if "total_assets" in detected_intents or "liabilities" in detected_intents or "equity" in detected_intents or "loans" in detected_intents:
        expansion_terms.extend(["laporan posisi keuangan", "jumlah aset", "jumlah liabilitas", "jumlah ekuitas"])

    expanded_query = query
    if expansion_terms:
        extra = " ".join(dict.fromkeys(expansion_terms))
        expanded_query = f"{query} ({extra})"

    return {
        "original_query": query,
        "expanded_query": expanded_query,
        "ticker": detected_ticker,
        "tickers": detected_tickers if detected_tickers else ["BBCA", "BBRI", "BBNI"],
        "is_comparison": is_comparison,
        "year": detected_year,
        "quarter": detected_quarter,
        "intents": detected_intents,
        "target_sheets": target_sheets,
        "metric_keys": metric_keys,
        "matched_keywords": matched_keywords,
        "preferred_doc_type": preferred_doc_type
    }
