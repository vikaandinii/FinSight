import os
import re
import sys
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from retrieval import search_financial_docs, analyze_query
try:
    from .metrics import get_multi_year_trend_table, calculate_key_ratios, format_rupiah_readable
except ImportError:
    from generation.metrics import get_multi_year_trend_table, calculate_key_ratios, format_rupiah_readable


def df_to_markdown_simple(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) if v is not None else "" for v in row.values) + " |")
        return "\n".join(lines)


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """Anda adalah FinSight AI, asisten analis keuangan dan akuntansi cerdas profesional khusus emiten Bursa Efek Indonesia (BEI / IDX).

ATURAN KETAT ANALISIS:
1. Jawab pertanyaan secara SPESIFIK sesuai maksud pertanyaan pengguna (misal: laba bersih, aset, liabilitas, ekuitas, kredit, atau rasio keuangan) HANYA berdasarkan [METRIK TERSTRUKTUR] dan [KONTEKS DOKUMEN LAPORAN KEUANGAN] yang disediakan.
2. DILARANG KERAS MENGARANG ANGKA (Zero-Hallucination Policy). Jika pos akun atau data tahun yang ditanyakan tidak tercantum di konteks, nyatakan secara eksplisit: "Maaf, informasi/pos akun tersebut tidak tersedia di laporan keuangan yang tersedia."
3. SATUAN ANGKA:
   - Selalu perhatikan satuan angka laporan keuangan (misal: Jutaan / In Million, Ribuan, atau Rupiah penuh).
   - Tuliskan nilai asli laporan beserta satuannya, dan sediakan konversi ringkas (misal: Miliar / Triliun Rupiah) agar mudah dipahami.
4. PERIODE WAKTU:
   - Perhatikan dengan seksama Ticker (misal: BBCA, BBRI, BBNI) dan Periode Tahun/Kuartal.
   - Bedakan antara Periode Berjalan (Current Period) dan Periode Pembanding Sebelumnya (Prior Period).
5. KUTIPAN SUMBER:
   - Di akhir penjelasan setiap angka atau temuan, wajib cantumkan sumber sheet laporan keuangan (contoh: [Sumber: Sheet 4322000 - Laporan Laba Rugi] atau [Sumber: Sheet 4220000 - Laporan Posisi Keuangan]).
"""


def get_gemini_client(api_key: str = None):
    key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as e:
        print(f"[GENERATOR ERROR] Gagal inisialisasi Gemini Client: {e}")
        return None


def extract_relevant_table_rows(content: str, keywords: list, max_rows: int = 8) -> str:
    """Mengekstrak baris-baris tabel Markdown yang cocok dengan kata kunci pertanyaan."""
    lines = content.splitlines()
    matched_lines = []
    header_lines = []
    for line in lines:
        if line.strip().startswith("|") and ("Pos_Akun" in line or "---" in line or "Kolom" in line):
            header_lines.append(line)
        elif line.strip().startswith("|"):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                matched_lines.append(line)

    if matched_lines:
        result_lines = header_lines[:2] + matched_lines[:max_rows]
        return "\n".join(result_lines)
    return "\n".join(lines[:12])


def _select_period_row(df_trend, year=None, quarter=None):
    if df_trend.empty:
        return None
    target_df = df_trend
    if year:
        filtered_y = target_df[target_df["Periode"].str.contains(str(year), na=False)]
        if not filtered_y.empty:
            target_df = filtered_y
    if quarter:
        filtered_q = target_df[target_df["Periode"].str.contains(str(quarter), case=False, na=False)]
        if not filtered_q.empty:
            target_df = filtered_q
    else:
        # Prioritaskan laporan Tahunan (FY) jika kuartal tidak disebut secara spesifik
        filtered_fy = target_df[target_df["Periode"].str.contains("Tahunan", case=False, na=False)]
        if not filtered_fy.empty:
            target_df = filtered_fy
    return target_df.iloc[-1]


def build_local_answer(query_info: dict, retrieved_docs: list) -> str:
    """
    Menyusun jawaban naratif alami yang fokus dan spesifik sesuai intent pertanyaan pengguna
    berdasarkan hasil retrieval dan metrik, lengkap dengan sitasi sumber dokumen.
    Mendukung analisis bank tunggal maupun perbandingan lintas bank (BBCA, BBRI, BBNI).
    """
    ticker = query_info.get("ticker")
    tickers = query_info.get("tickers", ["BBCA", "BBRI", "BBNI"])
    is_comparison = query_info.get("is_comparison", False)
    year = query_info.get("year")
    quarter = query_info.get("quarter")
    intents = query_info.get("intents", [])
    matched_keywords = query_info.get("matched_keywords", [])
    q_lower = query_info.get("original_query", "").lower()

    paragraphs = []

    # 1. Penanganan Pertanyaan Perbandingan / Peringkat Lintas Bank (misal: "bank apa laba bersih tertinggi di 2024?")
    if is_comparison or not ticker:
        target_banks = tickers if len(tickers) > 1 else ["BBCA", "BBRI", "BBNI"]
        bank_rows = []
        for b in target_banks:
            df_b = get_multi_year_trend_table(b)
            r_b = _select_period_row(df_b, year=year, quarter=quarter)
            if r_b is not None:
                ratios_b = calculate_key_ratios(b, year, quarter)
                bank_rows.append((b, r_b, ratios_b))

        if bank_rows and "accounting_policy" not in intents:
            wants_lowest = any(w in q_lower for w in ["terendah", "terkecil", "paling rendah", "paling kecil"])
            wants_highest = any(w in q_lower for w in ["tertinggi", "terbesar", "terbanyak", "paling tinggi", "paling besar", "unggul", "terbaik"])

            if "net_income" in intents or "income_statement" in intents:
                valid = [x for x in bank_rows if x[1].get("_raw_laba_bersih") is not None]
                valid.sort(key=lambda x: x[1]["_raw_laba_bersih"], reverse=not wants_lowest)
                if valid:
                    top_b, top_r, _ = valid[0]
                    label_rank = "terendah" if wants_lowest else "tertinggi"
                    if wants_highest or wants_lowest or "bank apa" in q_lower or "bank mana" in q_lower:
                        intro = (
                            f"Bank dengan **laba bersih {label_rank}** pada periode **{top_r['Periode']}** adalah **{top_b}** "
                            f"dengan capaian laba bersih tahun berjalan sebesar **{top_r['Laba Bersih']}** *[Sumber: Sheet 4322000 - Laporan Laba Rugi]*."
                        )
                    else:
                        intro = f"Berikut perbandingan **laba bersih tahun berjalan** antar bank pada periode **{top_r['Periode']}**:"
                    detail_lines = [
                        f"- **{b}** ({r['Periode']}): **{r['Laba Bersih']}** *[Sumber: Sheet 4322000]*"
                        for b, r, _ in valid
                    ]
                    paragraphs.append(intro + "\n\n" + "\n".join(detail_lines))

            elif "total_assets" in intents:
                valid = [x for x in bank_rows if x[1].get("_raw_total_aset") is not None]
                valid.sort(key=lambda x: x[1]["_raw_total_aset"], reverse=not wants_lowest)
                if valid:
                    top_b, top_r, _ = valid[0]
                    label_rank = "terkecil" if wants_lowest else "terbesar"
                    if wants_highest or wants_lowest or "bank apa" in q_lower or "bank mana" in q_lower:
                        intro = (
                            f"Bank dengan **total aset {label_rank}** pada periode **{top_r['Periode']}** adalah **{top_b}** "
                            f"dengan total aset mencapai **{top_r['Total Aset']}** *[Sumber: Sheet 4220000 - Laporan Posisi Keuangan]*."
                        )
                    else:
                        intro = f"Berikut perbandingan **total aset** antar bank pada periode **{top_r['Periode']}**:"
                    detail_lines = [
                        f"- **{b}** ({r['Periode']}): **{r['Total Aset']}** *[Sumber: Sheet 4220000]*"
                        for b, r, _ in valid
                    ]
                    paragraphs.append(intro + "\n\n" + "\n".join(detail_lines))

            elif "equity" in intents:
                valid = [x for x in bank_rows if x[1].get("_raw_total_ekuitas") is not None]
                valid.sort(key=lambda x: x[1]["_raw_total_ekuitas"], reverse=not wants_lowest)
                if valid:
                    top_b, top_r, _ = valid[0]
                    detail_lines = [
                        f"- **{b}** ({r['Periode']}): **{r['Total Ekuitas']}** *[Sumber: Sheet 4220000]*"
                        for b, r, _ in valid
                    ]
                    paragraphs.append(
                        f"Berikut perbandingan **total ekuitas** pada periode **{top_r['Periode']}** (tertinggi: **{top_b}** sebesar **{top_r['Total Ekuitas']}**):\n\n"
                        + "\n".join(detail_lines)
                    )

            elif "loans" in intents:
                valid = [x for x in bank_rows if x[1].get("_raw_kredit") is not None]
                valid.sort(key=lambda x: x[1]["_raw_kredit"], reverse=not wants_lowest)
                if valid:
                    top_b, top_r, _ = valid[0]
                    detail_lines = [
                        f"- **{b}** ({r['Periode']}): **{r['Kredit (Loans)']}** *[Sumber: Sheet 4220000]*"
                        for b, r, _ in valid
                    ]
                    paragraphs.append(
                        f"Berikut perbandingan **penyaluran kredit / pinjaman** pada periode **{top_r['Periode']}** (terbesar: **{top_b}** sebesar **{top_r['Kredit (Loans)']}**):\n\n"
                        + "\n".join(detail_lines)
                    )

            elif "financial_ratios" in intents:
                detail_lines = []
                for b, r, rat in bank_rows:
                    if rat:
                        r_str = ", ".join([f"{k}: **{v}**" for k, v in rat.items() if k not in ["ticker", "period", "date"]])
                        detail_lines.append(f"- **{b}** ({rat.get('period')}): {r_str} *[Sumber: Sheet 4220000 & Sheet 4322000]*")
                if detail_lines:
                    paragraphs.append("Berikut perbandingan **rasio profitabilitas dan likuiditas** antar bank:\n\n" + "\n".join(detail_lines))

            else:
                detail_lines = [
                    f"- **{b}** ({r['Periode']}): Laba Bersih **{r['Laba Bersih']}** | Total Aset **{r['Total Aset']}** | Ekuitas **{r['Total Ekuitas']}** *[Sumber: Sheet 4220000 & Sheet 4322000]*"
                    for b, r, _ in bank_rows
                ]
                paragraphs.append("Berikut ringkasan kinerja keuangan masing-masing bank:\n\n" + "\n".join(detail_lines))

    # 2. Penanganan Pertanyaan Spesifik Satu Bank (Ticker Tunggal)
    elif ticker:
        df_trend = get_multi_year_trend_table(ticker)
        ratios = calculate_key_ratios(ticker, year, quarter)
        row = _select_period_row(df_trend, year=year, quarter=quarter)
        if row is not None:
            periode_str = row["Periode"]
            tanggal_str = row["Tanggal"]

            sentences = []
            if "net_income" in intents or "income_statement" in intents:
                sentences.append(
                    f"Berdasarkan Laporan Laba Rugi **{ticker.upper()}** pada periode **{periode_str}** (per {tanggal_str}), "
                    f"**laba bersih tahun berjalan** tercatat sebesar **{row['Laba Bersih']}** *[Sumber: Sheet 4322000 - Laporan Laba Rugi]*."
                )
            if "total_assets" in intents:
                sentences.append(
                    f"Berdasarkan Laporan Posisi Keuangan **{ticker.upper()}** pada periode **{periode_str}** (per {tanggal_str}), "
                    f"**total aset** tercatat sebesar **{row['Total Aset']}** *[Sumber: Sheet 4220000 - Laporan Posisi Keuangan]*."
                )
            if "liabilities" in intents:
                sentences.append(
                    f"Pada periode **{periode_str}** (per {tanggal_str}), **total liabilitas** **{ticker.upper()}** "
                    f"tercatat sebesar **{row['Total Liabilitas']}** *[Sumber: Sheet 4220000 - Laporan Posisi Keuangan]*."
                )
            if "equity" in intents:
                sentences.append(
                    f"Pada periode **{periode_str}** (per {tanggal_str}), **total ekuitas** **{ticker.upper()}** "
                    f"tercatat sebesar **{row['Total Ekuitas']}** *[Sumber: Sheet 4220000 - Laporan Posisi Keuangan]*."
                )
            if "loans" in intents:
                sentences.append(
                    f"Total **pinjaman / kredit yang diberikan** oleh **{ticker.upper()}** pada periode **{periode_str}** (per {tanggal_str}) "
                    f"mencapai **{row['Kredit (Loans)']}** *[Sumber: Sheet 4220000 - Laporan Posisi Keuangan]*."
                )
            if "financial_ratios" in intents and ratios:
                r_str = ", ".join([f"**{k}** sebesar **{v}**" for k, v in ratios.items() if k not in ["ticker", "period", "date"]])
                if r_str:
                    sentences.append(
                        f"Untuk rasio profitabilitas **{ticker.upper()}** pada periode **{ratios.get('period')}**, tercatat {r_str} "
                        f"*[Sumber: Sheet 4220000 & Sheet 4322000]*."
                    )

            if sentences:
                paragraphs.append(" ".join(sentences))
            elif "accounting_policy" not in intents:
                paragraphs.append(
                    f"Pada periode **{periode_str}** (per {tanggal_str}), **{ticker.upper()}** mencatatkan **laba bersih** sebesar **{row['Laba Bersih']}**, "
                    f"**total aset** sebesar **{row['Total Aset']}**, **total liabilitas** sebesar **{row['Total Liabilitas']}**, dan "
                    f"**total ekuitas** sebesar **{row['Total Ekuitas']}** *[Sumber: Sheet 4220000 & Sheet 4322000]*."
                )

    # Jika pertanyaan mengenai kebijakan akuntansi / CKPN / narasi atau belum terjawab oleh metrik di atas
    if retrieved_docs and ("accounting_policy" in intents or not paragraphs):
        top_doc = retrieved_docs[0]
        m = top_doc.get("metadata", {})
        content = top_doc.get("content", "").strip()
        if m.get("doc_type") == "financial_table" and matched_keywords:
            excerpt = extract_relevant_table_rows(content, matched_keywords, max_rows=5)
            paragraphs.append(
                f"Berikut rincian pos terkait dari **{m.get('sheet_title_id')}** ({m.get('ticker')} {m.get('year')} {m.get('quarter')}):\n\n{excerpt}\n\n"
                f"*[Sumber: Sheet {m.get('sheet_code')}]*"
            )
        else:
            # Bersihkan baris header metadata dari page_content
            body_lines = [
                ln for ln in content.splitlines()
                if not ln.startswith("Emiten:") and not ln.startswith("Satuan Angka:")
            ]
            clean_body = "\n".join(body_lines).strip()
            if len(clean_body) > 600:
                clean_body = clean_body[:600] + "..."
            paragraphs.append(
                f"Berdasarkan **{m.get('sheet_title_id')}** ({m.get('ticker')} {m.get('year')} {m.get('quarter')}):\n\n{clean_body}\n\n"
                f"*[Sumber: Sheet {m.get('sheet_code')}]*"
            )

    if not paragraphs:
        return "Maaf, informasi yang Anda tanyakan tidak ditemukan di laporan keuangan yang tersedia."

    return "\n\n".join(paragraphs)


def answer_financial_query(
    query: str,
    ticker: str = None,
    year: str = None,
    quarter: str = None,
    api_key: str = None,
    model_name: str = None
):
    """
    Menjawab pertanyaan finansial dengan menggabungkan:
    1. Query Understanding (Intent, Target Sheet, Ticker, Tahun, Kuartal)
    2. Hybrid Retrieval + Cross-Feature Reranking
    3. Metrik keuangan terstruktur (anti-halusinasi)
    4. Gemini Reasoning LLM (atau ekstraksi spesifik berbasis konteks lokal jika tanpa API Key)
    """
    q_info = analyze_query(query, ticker=ticker, year=year, quarter=quarter)
    eff_ticker = q_info["ticker"]
    eff_year = q_info["year"]
    eff_quarter = q_info["quarter"]

    client = get_gemini_client(api_key)
    active_model = model_name or DEFAULT_MODEL_NAME

    # 1. Ambil dokumen relevan via Hybrid Search + Reranking
    retrieved_docs = search_financial_docs(
        query=query,
        ticker=eff_ticker,
        year=eff_year,
        quarter=eff_quarter,
        top_k=3
    )

    # 2. Ambil metrik terstruktur (baik untuk emiten tunggal maupun perbandingan semua bank)
    structured_context = ""
    target_tickers_ctx = [eff_ticker] if eff_ticker else q_info.get("tickers", ["BBCA", "BBRI", "BBNI"])
    for tk in target_tickers_ctx:
        df_trend = get_multi_year_trend_table(tk)
        if not df_trend.empty:
            summary_cols = ["Periode", "Total Aset", "Total Liabilitas", "Total Ekuitas", "Laba Bersih", "Kredit (Loans)", "Simpanan (DPK)"]
            available_cols = [c for c in summary_cols if c in df_trend.columns]
            structured_context += f"\n[METRIK TERSTRUKTUR MULTI-TAHUN {tk.upper()}]\n"
            structured_context += df_to_markdown_simple(df_trend[available_cols]) + "\n"

        ratios = calculate_key_ratios(tk, eff_year, eff_quarter)
        if ratios:
            structured_context += f"\n[RASIO KEUANGAN {tk.upper()} PERIODE {ratios.get('period', '')}]\n"
            for r_name, r_val in ratios.items():
                if r_name not in ["ticker", "period", "date"]:
                    structured_context += f"- {r_name}: {r_val}\n"
            structured_context += "\n"

    # 3. Format konteks dokumen menggunakan LangChain Document & PromptTemplate
    lc_documents = [
        Document(page_content=doc.get("content", ""), metadata=doc.get("metadata", {}))
        for doc in retrieved_docs
    ]
    doc_context = ""
    sources = []
    for idx, lc_doc in enumerate(lc_documents, 1):
        meta = lc_doc.metadata
        sheet_info = f"Sheet [{meta.get('sheet_code')}] {meta.get('sheet_title_id', '')} ({meta.get('ticker')} {meta.get('year')} {meta.get('quarter')})"
        sources.append(sheet_info)
        doc_context += f"\n--- DOKUMEN {idx} ({sheet_info}) ---\n"
        doc_context += lc_doc.page_content + "\n"

    rag_prompt_template = PromptTemplate.from_template(
        """[PERTANYAAN PENGGUNA]
{query}

[ANALISIS MAKSUD PERTANYAAN (QUERY UNDERSTANDING)]
- Emiten: {ticker}
- Periode: {year} {quarter}
- Fokus Topik (Intent): {intents}
- Sheet Prioritas: {target_sheets}

{structured_context}

[KONTEKS LAPORAN KEUANGAN TERKAIT (HASIL HYBRID RETRIEVAL & RERANKING)]
{doc_context}

Instruksi: Jawab secara spesifik sesuai fokus topik pertanyaan di atas, gunakan angka eksak dari konteks, dan wajib cantumkan kutipan sumber Sheet pada setiap klaim angka.
"""
    )
    full_prompt = rag_prompt_template.format(
        query=query,
        ticker=eff_ticker or "Semua",
        year=eff_year or "Semua",
        quarter=eff_quarter or "Semua",
        intents=", ".join(q_info["intents"]),
        target_sheets=", ".join(q_info["target_sheets"]),
        structured_context=structured_context,
        doc_context=doc_context if doc_context else "Tidak ada dokumen teks tambahan yang spesifik."
    )

    if not client:
        local_ans = build_local_answer(q_info, retrieved_docs)
        return {
            "answer": local_ans,
            "sources": list(dict.fromkeys(sources)),
            "structured_context": structured_context,
            "retrieved_docs": retrieved_docs,
            "query_understanding": q_info
        }

    try:
        from google.genai import types
        response = client.models.generate_content(
            model=active_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1
            )
        )
        return {
            "answer": response.text,
            "sources": list(dict.fromkeys(sources)),
            "structured_context": structured_context,
            "retrieved_docs": retrieved_docs,
            "query_understanding": q_info
        }
    except Exception:
        local_ans = build_local_answer(q_info, retrieved_docs)
        return {
            "answer": local_ans,
            "sources": list(dict.fromkeys(sources)),
            "structured_context": structured_context,
            "retrieved_docs": retrieved_docs,
            "query_understanding": q_info
        }


def run_interactive_cli():
    """Mode Tanya Jawab Interaktif langsung di Terminal."""
    print("=" * 72)
    print("FINSIGHT AI - MODE TANYA JAWAB INTERAKTIF DI TERMINAL")
    print("Ketik pertanyaan keuangan (contoh: 'Berapa laba bersih BBCA tahun 2024?')")
    print("Ketik 'exit' atau 'keluar' untuk berhenti.")
    print("=" * 72)
    while True:
        try:
            user_q = input("\nPertanyaan kamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari mode interaktif.")
            break
        if not user_q:
            continue
        if user_q.lower() in ["exit", "quit", "keluar", "q"]:
            print("Sampai jumpa!")
            break

        res = answer_financial_query(user_q)
        q_info = res.get("query_understanding", {})
        print("\n" + "-" * 72)
        print(f"QUERY UNDERSTANDING: Intent={q_info.get('intents')} | Ticker={q_info.get('ticker')} | Periode={q_info.get('year')} {q_info.get('quarter') or ''}")
        print("-" * 72)
        print("JAWABAN FINSIGHT AI:")
        print("-" * 72)
        print(res["answer"])
        if res.get("sources"):
            print("\nSumber Sheet:")
            for s in res["sources"]:
                print(f" - {s}")
        print("-" * 72)


if __name__ == "__main__":
    run_interactive_cli()
