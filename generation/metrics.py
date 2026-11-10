import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "data", "processed", "key_financial_metrics.json")


def load_metrics_data(filepath=METRICS_PATH):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_to_full_rupiah(raw_val, rounding="Jutaan / In Million"):
    """Mengonversi nilai ke Rupiah penuh dengan deteksi pintar apakah angka sudah dalam Rupiah penuh."""
    if raw_val is None or pd.isna(raw_val):
        return None
    val = float(raw_val)
    abs_val = abs(val)

    if abs_val > 100_000_000_000:
        return val

    multiplier = 1.0
    r_lower = str(rounding).lower()
    if "juta" in r_lower or "million" in r_lower:
        multiplier = 1_000_000.0
    elif "ribu" in r_lower or "thousand" in r_lower:
        multiplier = 1_000.0

    return val * multiplier


def format_rupiah_readable(raw_val, rounding="Jutaan / In Million"):
    """Mengubah angka mentah dengan satuan laporan menjadi format Triliun/Miliar yang mudah dibaca."""
    full_val = normalize_to_full_rupiah(raw_val, rounding)
    if full_val is None:
        return "N/A"

    abs_val = abs(full_val)
    prefix = "-" if full_val < 0 else ""
    if abs_val >= 1_000_000_000_000:
        return f"{prefix}Rp {abs_val / 1_000_000_000_000:.2f} Triliun"
    elif abs_val >= 1_000_000_000:
        return f"{prefix}Rp {abs_val / 1_000_000_000:.2f} Miliar"
    else:
        return f"{prefix}Rp {abs_val:,.0f}".replace(",", ".")


def get_emiten_summary(ticker: str):
    """Mengambil riwayat seluruh metrik laporan keuangan untuk suatu emiten."""
    data = load_metrics_data()
    ticker = ticker.upper()
    reports = [r for r in data if r.get("ticker") == ticker]

    reports.sort(key=lambda x: x.get("period_date", ""))
    return reports


def get_multi_year_trend_table(ticker: str):
    """Menyusun tabel perbandingan multi-tahun untuk emiten tertentu."""
    reports = get_emiten_summary(ticker)
    if not reports:
        return pd.DataFrame()

    rows = []
    for r in reports:
        m = r.get("metrics", {})
        rounding = r.get("rounding", "")
        row = {
            "Ticker": r.get("ticker"),
            "Periode": f"{r.get('year')} {r.get('quarter')}",
            "Tanggal": r.get("period_date"),
            "Total Aset": format_rupiah_readable(m.get("total_aset", {}).get("raw_value"), rounding),
            "Total Liabilitas": format_rupiah_readable(m.get("total_liabilitas", {}).get("raw_value"), rounding),
            "Total Ekuitas": format_rupiah_readable(m.get("total_ekuitas", {}).get("raw_value"), rounding),
            "Laba Bersih": format_rupiah_readable(m.get("laba_bersih_berjalan", {}).get("raw_value"), rounding),
            "Kredit (Loans)": format_rupiah_readable(m.get("pinjaman_kredit", {}).get("raw_value"), rounding),
            "Simpanan (DPK)": format_rupiah_readable(m.get("simpanan_nasabah_dpk", {}).get("raw_value"), rounding),
            "_raw_total_aset": normalize_to_full_rupiah(m.get("total_aset", {}).get("raw_value"), rounding),
            "_raw_laba_bersih": normalize_to_full_rupiah(m.get("laba_bersih_berjalan", {}).get("raw_value"), rounding),
            "_raw_total_ekuitas": normalize_to_full_rupiah(m.get("total_ekuitas", {}).get("raw_value"), rounding),
            "_raw_kredit": normalize_to_full_rupiah(m.get("pinjaman_kredit", {}).get("raw_value"), rounding),
            "_raw_dpk": normalize_to_full_rupiah(m.get("simpanan_nasabah_dpk", {}).get("raw_value"), rounding),
            "Satuan": rounding
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def calculate_key_ratios(ticker: str, year: str = None, quarter: str = None):
    """Menghitung rasio keuangan utama (ROA, ROE, LDR) secara matematis tanpa halusinasi LLM."""
    reports = get_emiten_summary(ticker)
    if not reports:
        return {}

    target = None
    for r in reversed(reports):
        match_year = (year is None or str(r.get("year")) == str(year))
        match_quarter = (quarter is None or str(r.get("quarter")).upper() == str(quarter).upper())
        if match_year and match_quarter:
            target = r
            break

    if not target:
        target = reports[-1]

    m = target.get("metrics", {})
    total_aset = m.get("total_aset", {}).get("raw_value")
    total_ekuitas = m.get("total_ekuitas", {}).get("raw_value")
    laba_bersih = m.get("laba_bersih_berjalan", {}).get("raw_value")
    kredit = m.get("pinjaman_kredit", {}).get("raw_value")
    dpk = m.get("simpanan_nasabah_dpk", {}).get("raw_value")

    ratios = {
        "ticker": target.get("ticker"),
        "period": f"{target.get('year')} {target.get('quarter')}",
        "date": target.get("period_date")
    }

    if total_aset and laba_bersih:
        ratios["ROA"] = f"{(laba_bersih / total_aset) * 100:.2f}%"
    if total_ekuitas and laba_bersih:
        ratios["ROE"] = f"{(laba_bersih / total_ekuitas) * 100:.2f}%"
    if kredit and dpk:
        ratios["LDR"] = f"{(kredit / dpk) * 100:.2f}%"

    return ratios


if __name__ == "__main__":
    df_bbca = get_multi_year_trend_table("BBCA")
    print("Ringkasan Tren Multi-Tahun BBCA:")
    print(df_bbca[["Periode", "Total Aset", "Laba Bersih", "Total Ekuitas"]])
