import os
import re
import json
import glob
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dataframe_to_markdown(df):
    """Konversi DataFrame ke tabel Markdown dengan fallback mandiri tanpa dependensi tabulate."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(c) for c in df.columns]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines = [header_row, sep_row]
        for _, row in df.iterrows():
            row_vals = [str(v) if not pd.isna(v) else "" for v in row.values]
            lines.append("| " + " | ".join(row_vals) + " |")
        return "\n".join(lines)


def format_val(val):
    """Memformat nilai numerik akuntansi agar konsisten."""
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return ""
        if 0 < abs(val) < 1:
            return f"{val * 100:.2f}%"
        if float(val).is_integer():
            return f"{int(val):,}".replace(",", ".")
        return f"{val:,.2f}".replace(",", ".")
    return str(val).strip()


def clean_sheet_title(df_raw, sheet_code):
    """Membersihkan judul sheet dari header XBRL dan baris awal."""
    col0_header = str(df_raw.columns[0]).strip()
    en_header = re.sub(r"^\[.*?\]\s*", "", col0_header)
    en_header = re.sub(r"\s*-\s*Financial and Sharia Industry$", "", en_header, flags=re.IGNORECASE)

    id_title = ""
    for r in [1, 2, 3]:
        if r < len(df_raw):
            candidate = str(df_raw.iloc[r, 0]).strip()
            if candidate and candidate not in ["nan", "CurrentYearInstant", "CurrentYearDuration"]:
                id_title = candidate
                break

    if not id_title or id_title.lower() == "pengungkapan":
        return f"{id_title} - {en_header}" if id_title else en_header
    return f"{id_title} ({en_header})"


def parse_filename_meta(file_path):
    """Mengekstrak Tahun, Kuartal, dan Ticker dari nama file laporan keuangan IDX."""
    base_name = os.path.basename(file_path)
    match = re.match(r"FinancialStatement-(\d{4})-([A-Za-z0-9]+)-([A-Za-z0-9]+)\.xlsx", base_name, re.IGNORECASE)
    if match:
        year = match.group(1)
        q_raw = match.group(2).upper()
        ticker = match.group(3).upper()
        q_map = {"I": "Q1", "II": "Q2", "III": "Q3", "IV": "Q4", "TAHUNAN": "Tahunan"}
        quarter = q_map.get(q_raw, q_raw)
        return year, quarter, ticker
    return "", "", ""


def extract_primary_metrics(df_raw, sheet_code):
    """Mengekstrak pos-pos keuangan utama dari sheet Neraca dan Laba Rugi."""
    metrics = {}
    if df_raw.empty or len(df_raw) < 5 or df_raw.shape[1] < 3:
        return metrics

    p_curr_idx = 1

    target_patterns = {
        "total_aset": [r"^jumlah aset\b", r"^total aset\b", r"^total assets\b"],
        "total_liabilitas": [r"^jumlah liabilitas\b", r"^total liabilitas\b", r"^total liabilities\b"],
        "total_ekuitas": [r"^jumlah ekuitas\b", r"^total ekuitas\b", r"^total equity\b"],
        "kas_dan_setara_kas": [r"^kas dan setara kas\b", r"^kas\b", r"^cash and cash equivalents\b"],
        "pinjaman_kredit": [r"^kredit yang diberikan\b", r"^pinjaman yang diberikan\b", r"^pinjaman\b", r"^loans\b"],
        "simpanan_nasabah_dpk": [r"^simpanan nasabah\b", r"^dana pihak ketiga\b", r"^deposits from customers\b", r"^simpanan dari nasabah\b"],
        "pendapatan_bunga_bersih": [r"^pendapatan bunga bersih\b", r"^pendapatan bunga dan syariah bersih\b", r"^net interest income\b"],
        "laba_sebelum_pajak": [r"^jumlah laba\s*\(?rugi\)?\s*sebelum pajak\b", r"^laba sebelum pajak\b", r"^laba sebelum beban pajak\b", r"^profit before tax\b"],
        "laba_bersih_berjalan": [
            r"^jumlah laba\s*\(?rugi\)?\s*tahun berjalan\b",
            r"^jumlah laba\s*\(?rugi\)?$",
            r"^jumlah laba\s*\(?rugi\)?\s*dari operasi yang dilanjutkan\b",
            r"^laba tahun berjalan\b",
            r"^laba bersih\b",
            r"^profit for the year\b"
        ],
        "laba_atribusi_induk": [
            r"^laba\s*\(?rugi\)?\s*yang dapat diatribusikan ke(?:pada)? entitas induk\b",
            r"^laba\s*\(?rugi\)?\s*yang dapat diatribusikan kepada pemilik entitas induk\b"
        ]
    }

    for idx, row in df_raw.iloc[4:].iterrows():
        pos_id = str(row.iloc[0]).strip().lower()
        pos_en = str(row.iloc[3]).strip().lower() if df_raw.shape[1] >= 4 else ""
        raw_val = row.iloc[p_curr_idx]

        if pd.isna(raw_val) or not isinstance(raw_val, (int, float)):
            continue

        for key, patterns in target_patterns.items():
            if key not in metrics:
                for pat in patterns:
                    if re.search(pat, pos_id) or (pos_en and re.search(pat, pos_en)):
                        metrics[key] = {
                            "account_name": str(row.iloc[0]).strip(),
                            "raw_value": float(raw_val),
                            "formatted_value": format_val(raw_val),
                            "sheet_code": sheet_code
                        }
                        break
    return metrics


def extract_file_to_chunks_and_metrics(file_path):
    """Mengekstrak file Excel menjadi dokumen teks (chunks) dan metrik terstruktur."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        return [], {}

    file_year, file_quarter, file_ticker = parse_filename_meta(file_path)
    xls = pd.ExcelFile(file_path)

    info_dict = {}
    if "1000000" in xls.sheet_names:
        df_info = pd.read_excel(file_path, sheet_name="1000000").dropna(how="all")
        info_dict = dict(zip(df_info.iloc[:, 0].astype(str).str.strip(), df_info.iloc[:, 1]))

    entity_name = str(info_dict.get("Nama entitas", file_ticker or "Emiten")).strip()
    ticker = str(info_dict.get("Kode entitas", file_ticker or "UNKNOWN")).strip().upper()
    period_date = str(info_dict.get("Tanggal akhir periode berjalan", "")).strip()
    rounding = str(info_dict.get("Pembulatan yang digunakan dalam penyajian jumlah dalam laporan keuangan", "Jutaan / In Million")).strip()

    if not period_date and file_year:
        q_date_map = {"Q1": f"{file_year}-03-31", "Q2": f"{file_year}-06-30", "Q3": f"{file_year}-09-30", "Tahunan": f"{file_year}-12-31"}
        period_date = q_date_map.get(file_quarter, f"{file_year}-12-31")

    actual_year = file_year or (period_date[:4] if len(period_date) >= 4 else "UNKNOWN")

    chunks = []
    file_key_metrics = {
        "ticker": ticker,
        "entity_name": entity_name,
        "year": str(actual_year),
        "quarter": str(file_quarter),
        "period_date": str(period_date),
        "rounding": str(rounding),
        "file_name": os.path.basename(file_path),
        "metrics": {}
    }

    skip_sheets = {"Context", "InlineXBRL", "hidden", "Token", "1000000"}

    for sheet in xls.sheet_names:
        if sheet in skip_sheets:
            continue

        try:
            df_raw = pd.read_excel(file_path, sheet_name=sheet)
            if df_raw.empty or len(df_raw) < 4:
                continue

            full_title = clean_sheet_title(df_raw, sheet)
            header_context = (
                f"Emiten: {entity_name} ({ticker}) | Periode: {period_date} | Laporan: [{sheet}] {full_title}\n"
                f"Satuan Angka: {rounding}\n"
            )

            if sheet in ["4220000", "4312000", "4322000"]:
                extracted_pos = extract_primary_metrics(df_raw, sheet)
                file_key_metrics["metrics"].update(extracted_pos)

            # KASUS 1: Sheet Catatan / Pengungkapan Teks (3 Kolom)
            if df_raw.shape[1] == 3:
                df_text = df_raw.iloc[4:, :2].copy()
                df_text.columns = ["Topik", "Penjelasan"]
                df_text["Topik"] = df_text["Topik"].ffill()
                df_text = df_text.dropna(subset=["Penjelasan"])
                df_text = df_text[df_text["Penjelasan"].astype(str).str.strip() != "-"]

                if not df_text.empty:
                    grouped = df_text.groupby("Topik", sort=False)["Penjelasan"].apply(
                        lambda x: "\n\n".join(x.astype(str).str.strip())
                    ).reset_index()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ". ", " ", ""]
                    )

                    for idx, row in grouped.iterrows():
                        topik = str(row["Topik"]).strip()
                        isi = str(row["Penjelasan"]).strip()
                        if not isi:
                            continue

                        clean_topik_id = re.sub(r"[^A-Za-z0-9]", "_", topik)[:30]
                        base_meta = {
                            "ticker": ticker,
                            "entity_name": entity_name,
                            "year": str(actual_year),
                            "quarter": str(file_quarter),
                            "period": str(period_date),
                            "sheet_code": str(sheet),
                            "sheet_title_id": full_title,
                            "topic": topik,
                            "doc_type": "narrative_disclosure",
                            "rounding": str(rounding)
                        }

                        lc_docs = text_splitter.create_documents([isi], metadatas=[base_meta])
                        for sub_idx, lc_doc in enumerate(lc_docs):
                            chunk_text = f"{header_context}Topik Pengungkapan: {topik}\n\n{lc_doc.page_content}"
                            suffix = f"_{sub_idx}" if len(lc_docs) > 1 else ""
                            chunks.append({
                                "id": f"{ticker}_{actual_year}_{file_quarter}_{sheet}_{clean_topik_id}_{idx}{suffix}",
                                "page_content": chunk_text,
                                "metadata": lc_doc.metadata
                            })

            # KASUS 2: Tabel Standar 4 Kolom
            elif df_raw.shape[1] == 4:
                p_curr = str(df_raw.iloc[2, 1]) if not pd.isna(df_raw.iloc[2, 1]) else "Current_Period"
                p_prior = str(df_raw.iloc[2, 2]) if not pd.isna(df_raw.iloc[2, 2]) else "Prior_Period"

                df_data = df_raw.iloc[4:, :3].copy()
                df_data.columns = ["Pos_Akun", p_curr, p_prior]
                df_data = df_data.dropna(subset=[p_curr, p_prior], how="all").dropna(subset=["Pos_Akun"])
                if not df_data.empty:
                    df_data[p_curr] = df_data[p_curr].apply(format_val)
                    df_data[p_prior] = df_data[p_prior].apply(format_val)

                    md_table = dataframe_to_markdown(df_data)
                    chunks.append({
                        "id": f"{ticker}_{actual_year}_{file_quarter}_{sheet}",
                        "page_content": header_context + "\n" + md_table,
                        "metadata": {
                            "ticker": ticker,
                            "entity_name": entity_name,
                            "year": str(actual_year),
                            "quarter": str(file_quarter),
                            "period": str(period_date),
                            "sheet_code": str(sheet),
                            "sheet_title_id": full_title,
                            "doc_type": "financial_table",
                            "rounding": str(rounding)
                        }
                    })

            # KASUS 3: Tabel Rincian Multi-Kolom
            else:
                df_multi = df_raw.iloc[1:].copy()
                df_multi = df_multi.dropna(how="all").dropna(axis=1, how="all")
                if not df_multi.empty:
                    df_multi.iloc[:, 0] = df_multi.iloc[:, 0].ffill()
                    df_formatted = df_multi.map(format_val)

                    headers = [str(col).strip() if col and not pd.isna(col) else f"Kolom_{i}"
                               for i, col in enumerate(df_formatted.iloc[0].tolist())]
                    df_formatted = df_formatted.iloc[1:]
                    df_formatted.columns = headers

                    md_table = dataframe_to_markdown(df_formatted)
                    chunks.append({
                        "id": f"{ticker}_{actual_year}_{file_quarter}_{sheet}",
                        "page_content": header_context + "\n" + md_table,
                        "metadata": {
                            "ticker": ticker,
                            "entity_name": entity_name,
                            "year": str(actual_year),
                            "quarter": str(file_quarter),
                            "period": str(period_date),
                            "sheet_code": str(sheet),
                            "sheet_title_id": full_title,
                            "doc_type": "breakdown_table",
                            "rounding": str(rounding)
                        }
                    })
        except Exception as e:
            print(f"  [WARN] Skip sheet {sheet}: {e}")

    return chunks, file_key_metrics


def run_ingestion(data_dirs=None, output_chunks_path=None, output_metrics_path=None):
    """Menjalankan proses ekstraksi seluruh file laporan keuangan di folder data."""
    if data_dirs is None:
        data_dirs = [os.path.join(BASE_DIR, "data"), os.path.join(BASE_DIR, "data", "raw")]
    if output_chunks_path is None:
        output_chunks_path = os.path.join(BASE_DIR, "data", "processed", "financial_rag_data.json")
    if output_metrics_path is None:
        output_metrics_path = os.path.join(BASE_DIR, "data", "processed", "key_financial_metrics.json")

    all_files = []
    for d in data_dirs:
        if os.path.exists(d):
            all_files.extend(glob.glob(os.path.join(d, "*.xlsx")))

    unique_files = list({os.path.abspath(f): f for f in all_files}.values())
    if not unique_files:
        print("[INGEST] Tidak ditemukan file .xlsx di folder data.")
        return [], []

    print(f"\n[INGEST] Memulai ekstraksi {len(unique_files)} file laporan keuangan...")
    all_chunks = []
    all_metrics = []

    for f in sorted(unique_files):
        print(f" -> Memproses: {os.path.basename(f)}...")
        try:
            chunks, metrics = extract_file_to_chunks_and_metrics(f)
            all_chunks.extend(chunks)
            if metrics.get("metrics"):
                all_metrics.append(metrics)
        except Exception as e:
            print(f"  [ERROR] Gagal memproses {f}: {e}")

    os.makedirs(os.path.dirname(output_chunks_path), exist_ok=True)
    with open(output_chunks_path, "w", encoding="utf-8") as out:
        json.dump(all_chunks, out, ensure_ascii=False, indent=2)

    with open(output_metrics_path, "w", encoding="utf-8") as out:
        json.dump(all_metrics, out, ensure_ascii=False, indent=2)

    print(f"\n[SUKSES] Selesai!")
    print(f" - Total Document Chunks : {len(all_chunks)} disimpan di '{output_chunks_path}'")
    print(f" - Structured Metrics    : {len(all_metrics)} laporan disimpan di '{output_metrics_path}'")
    return all_chunks, all_metrics


if __name__ == "__main__":
    run_ingestion()
