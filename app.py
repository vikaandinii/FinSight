import os
import sys
import importlib

# Konfigurasi backend ML: paksa PyTorch
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import streamlit as st
from dotenv import load_dotenv

# Path setup
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import retrieval.embedder as _emb_mod
import retrieval.vector_store as _vs_mod
import retrieval.query_understanding as _qu_mod
import retrieval.retriever as _ret_mod
import generation.generator as _gen_mod
importlib.reload(_emb_mod)
importlib.reload(_vs_mod)
importlib.reload(_qu_mod)
importlib.reload(_ret_mod)
importlib.reload(_gen_mod)

from ingestion import run_ingestion
from retrieval.embedder import get_raw_documents, reload_embeddings_cache
from retrieval.vector_store import get_chroma_collection, sync_chroma_collection
from generation.generator import answer_financial_query

load_dotenv()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

st.set_page_config(
    page_title="FinSight",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False
if "show_login_modal" not in st.session_state:
    st.session_state["show_login_modal"] = False
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = [
        {"title": "Analisis Laba Bersih BBCA", "messages": []},
        {"title": "Evaluasi Rasio ROA & ROE", "messages": []},
        {"title": "Perbandingan Aset Bank", "messages": []},
    ]
if "current_session_idx" not in st.session_state:
    st.session_state["current_session_idx"] = 0

current_session = st.session_state["chat_sessions"][st.session_state["current_session_idx"]]
messages = current_session["messages"]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #171717;
    }

    #MainMenu, footer, .stDeployButton, div[data-testid="stToolbarActions"], div[data-testid="stAppDeployButton"] {
        display: none !important;
        visibility: hidden !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 720px;
        margin: 0 auto;
    }

    /* Sidebar putih bersih dengan garis pemisah tipis */
    div[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #EDEDED !important;
        width: 255px !important;
    }

    /* Rapatkan jarak vertikal antar elemen di dalam sidebar */
    div[data-testid="stSidebarUserContent"] {
        padding-top: 1rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.15rem !important;
    }

    div[data-testid="stSidebar"] .element-container {
        margin-bottom: 0 !important;
    }

    /* Hilangkan semua border kotak pada semua tombol dan paksa rata kiri lurus */
    .stButton > button {
        background: transparent !important;
        border: none !important;
        color: #6B7280 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.25rem 0.4rem !important;
        min-height: 30px !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
        font-weight: 400 !important;
        box-shadow: none !important;
    }

    .stButton > button > div,
    .stButton > button p {
        text-align: left !important;
        justify-content: flex-start !important;
        margin: 0 !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        color: #111827 !important;
        background-color: #F5F5F5 !important;
        border: none !important;
    }

    .stButton > button:focus, .stButton > button:active {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Samakan ukuran tulisan semua elemen di dalam sidebar (0.86rem) */
    div[data-testid="stSidebar"] .stButton > button,
    div[data-testid="stSidebar"] .stButton > button p {
        color: #262626 !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        padding: 0.3rem 0.4rem !important;
        min-height: 28px !important;
        width: 100% !important;
    }

    div[data-testid="stSidebar"] div[data-baseweb="select"],
    div[data-testid="stSidebar"] div[data-baseweb="select"] * {
        font-size: 0.86rem !important;
        font-weight: 400 !important;
    }

    div[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: transparent !important;
        border: none !important;
        border-radius: 6px !important;
        min-height: 28px !important;
        padding: 0.1rem 0.4rem !important;
        color: #262626 !important;
        box-shadow: none !important;
    }

    div[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        background-color: #F5F5F5 !important;
    }

    .sidebar-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #171717;
        padding: 0 0.4rem 0.6rem 0.4rem;
    }

    .sidebar-label {
        font-size: 0.86rem;
        font-weight: 500;
        color: #6B7280;
        padding: 0.6rem 0.4rem 0.2rem 0.4rem;
    }

    /* Judul utama di tengah layar */
    .center-greeting {
        margin-top: 16vh;
        margin-bottom: 1.6rem;
        text-align: center;
        font-size: 1.65rem;
        font-weight: 500;
        color: #171717;
        letter-spacing: -0.02em;
    }

    /* Buat st.chat_input berbentuk OVAL */
    div[data-testid="stBottom"] {
        background: transparent !important;
    }

    div[data-testid="stChatInput"] {
        max-width: 660px !important;
        margin: 0 auto !important;
    }

    div[data-testid="stChatInput"] > div {
        border-radius: 9999px !important;
        border: 1px solid #E5E5E5 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
        padding: 4px 12px !important;
    }

    div[data-testid="stChatInputSubmitButton"] button {
        border-radius: 9999px !important;
    }

    div[data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.75rem 0 !important;
    }

    div[data-testid="stExpander"],
    div[data-testid="stExpander"] details {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# BAR ATAS KANAN (TOMBOL ADMIN)
# =============================================================
top_spacer, top_right = st.columns([8.5, 1.5])
with top_right:
    if not st.session_state["admin_logged_in"]:
        if st.button("Admin", key="top_admin_btn"):
            st.session_state["show_login_modal"] = not st.session_state["show_login_modal"]
            st.rerun()
    else:
        if st.button("Logout Admin", key="top_logout_btn"):
            st.session_state["admin_logged_in"] = False
            st.session_state["show_login_modal"] = False
            st.rerun()

# =============================================================
# SIDEBAR (Filter paling atas, New chat, & Recents)
# =============================================================
with st.sidebar:
    st.markdown("<div class='sidebar-header'>FinSight</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>Filter</div>", unsafe_allow_html=True)

    sel_bank = st.selectbox(
        "Bank",
        options=["BBCA", "BBRI", "BBNI"],
        index=None,
        placeholder="Pilih Bank",
        label_visibility="collapsed"
    )
    sel_year = st.selectbox(
        "Tahun",
        options=["2025", "2024", "2023", "2022", "2021"],
        index=None,
        placeholder="Pilih Tahun",
        label_visibility="collapsed"
    )
    sel_quarter = st.selectbox(
        "Kuartal",
        options=["FY", "Q3", "Q2", "Q1"],
        index=None,
        placeholder="Pilih Kuartal",
        label_visibility="collapsed"
    )

    filter_ticker = sel_bank
    filter_year = sel_year
    filter_quarter = sel_quarter

    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)

    if st.button("New chat", width="stretch"):
        st.session_state["chat_sessions"].insert(0, {"title": "New chat", "messages": []})
        st.session_state["current_session_idx"] = 0
        st.session_state["show_login_modal"] = False
        st.rerun()

    st.markdown("<div class='sidebar-label'>Recents</div>", unsafe_allow_html=True)
    for idx, sess in enumerate(st.session_state["chat_sessions"]):
        label = sess["title"][:28] + ("..." if len(sess["title"]) > 28 else "")
        if st.button(label, key=f"rec_{idx}", width="stretch"):
            st.session_state["current_session_idx"] = idx
            st.session_state["show_login_modal"] = False
            st.rerun()


def queue_query(q_text: str):
    if not current_session["messages"]:
        current_session["title"] = q_text[:30]
    current_session["messages"].append({"role": "user", "content": q_text})
    st.session_state["pending_query"] = q_text


# =============================================================
# LOGIN ADMIN
# =============================================================
if st.session_state["show_login_modal"] and not st.session_state["admin_logged_in"]:
    st.markdown("<div class='center-greeting'>Admin Login</div>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        u_in = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
        p_in = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign in", width="stretch"):
                if u_in == ADMIN_USERNAME and p_in == ADMIN_PASSWORD:
                    st.session_state["admin_logged_in"] = True
                    st.session_state["show_login_modal"] = False
                    st.rerun()
                else:
                    st.error("Username atau password salah.")
        with c2:
            if st.button("Cancel", width="stretch"):
                st.session_state["show_login_modal"] = False
                st.rerun()

# =============================================================
# HALAMAN ADMIN (DATA INGESTION)
# =============================================================
elif st.session_state["admin_logged_in"]:
    st.markdown("### Data Ingestion")
    raw_docs = get_raw_documents()
    chroma_col = get_chroma_collection()
    chroma_count = chroma_col.count() if chroma_col else 0
    st.caption(f"Document Chunks: {len(raw_docs)}  •  ChromaDB Vectors: {chroma_count}")

    uploaded_files = st.file_uploader(
        "Upload Excel (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        for uf in uploaded_files:
            with open(os.path.join(data_dir, uf.name), "wb") as f_out:
                f_out.write(uf.getbuffer())
        st.success(f"{len(uploaded_files)} file berhasil diunggah.")

    if st.button("Run Ingestion & Update ChromaDB"):
        with st.spinner("Memproses dokumen ke ChromaDB..."):
            run_ingestion()
            reload_embeddings_cache()
            sync_chroma_collection()
        st.success("Database vektor berhasil diperbarui.")
        st.rerun()

# =============================================================
# HALAMAN UTAMA USER
# =============================================================
else:
    if len(messages) == 0:
        st.markdown("<div class='center-greeting'>Analisis Laporan Keuangan Perbankan</div>", unsafe_allow_html=True)

        with st.container():
            inline_input = st.chat_input("Ask FinSight", key="inline_chat")
            if inline_input and inline_input.strip():
                queue_query(inline_input.strip())
                st.rerun()

        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

        if st.button("Berapa laba bersih dan total aset BBCA tahun 2024?"):
            queue_query("Berapa laba bersih dan total aset BBCA tahun 2024?")
            st.rerun()
        if st.button("Bandingkan kinerja laba bersih BBCA dan BBRI tahun 2024"):
            queue_query("Bandingkan laba bersih BBCA dan BBRI pada tahun 2024")
            st.rerun()
        if st.button("Analisis rasio profitabilitas ROA dan ROE BBCA tahun 2024"):
            queue_query("Berapa rasio ROA dan ROE BBCA pada tahun 2024?")
            st.rerun()

    else:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("Sumber dokumen"):
                        for s in msg["sources"]:
                            st.caption(s)

        # Tampilkan animasi loading tepat di bawah pesan user saat sedang memproses jawaban
        if st.session_state.get("pending_query"):
            q_to_run = st.session_state["pending_query"]
            st.session_state["pending_query"] = None
            with st.chat_message("assistant"):
                with st.spinner("Sedang menganalisis laporan keuangan..."):
                    res = answer_financial_query(
                        query=q_to_run,
                        ticker=filter_ticker,
                        year=filter_year,
                        quarter=filter_quarter
                    )
                st.markdown(res["answer"])
                if res.get("sources"):
                    with st.expander("Sumber dokumen"):
                        for s in res["sources"]:
                            st.caption(s)
            current_session["messages"].append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res.get("sources", [])
            })
            st.rerun()

        bottom_input = st.chat_input("Ask FinSight", key="bottom_chat")
        if bottom_input and bottom_input.strip():
            queue_query(bottom_input.strip())
            st.rerun()
