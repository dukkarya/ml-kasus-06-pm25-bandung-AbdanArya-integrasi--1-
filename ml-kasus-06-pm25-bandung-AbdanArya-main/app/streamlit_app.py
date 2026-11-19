import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.predictor import load_resources, predict_pm25
from src.features import STATIONS


# -----------------------------------------------------------------------------
# Page setup and visual theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Udara Bandung | Prediksi PM2.5",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
      }
      /* Latar halaman dan ornamen dekoratif, tanpa mengunci scroll browser. */
      html, body, #root {
        min-height: 100% !important;
        height: auto !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        background: #e4f0ee !important;
      }
      .stApp {
        position: relative;
        isolation: isolate;
        min-height: 100vh !important;
        height: auto !important;
        overflow: visible !important;
        background:
          radial-gradient(ellipse 42rem 34rem at 0% 16%, rgba(35, 155, 132, .40) 0%, rgba(35, 155, 132, 0) 72%),
          radial-gradient(ellipse 40rem 32rem at 100% 14%, rgba(74, 145, 190, .34) 0%, rgba(74, 145, 190, 0) 72%),
          radial-gradient(ellipse 44rem 34rem at 88% 92%, rgba(50, 153, 126, .30) 0%, rgba(50, 153, 126, 0) 72%),
          radial-gradient(ellipse 34rem 26rem at 8% 88%, rgba(102, 177, 163, .22) 0%, rgba(102, 177, 163, 0) 70%),
          linear-gradient(132deg, #dcefeb 0%, #f1f7f8 48%, #e0edf4 100%) !important;
        background-attachment: scroll !important;
      }
      /* Ornamen ikut memanjang bersama halaman; tidak menutupi area input. */
      .stApp::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image:
          repeating-radial-gradient(circle at 2% 18%, transparent 0 76px, rgba(26, 111, 105, .10) 77px 79px, transparent 80px 118px),
          repeating-radial-gradient(circle at 98% 82%, transparent 0 92px, rgba(46, 115, 151, .09) 93px 95px, transparent 96px 140px);
      }
      .stApp::after {
        content: "";
        position: absolute;
        width: 310px;
        height: 310px;
        right: -150px;
        top: 145px;
        z-index: 0;
        pointer-events: none;
        border: 1px solid rgba(40, 125, 133, .15);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(40, 125, 133, .045), 0 0 0 58px rgba(40, 125, 133, .035);
      }
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="stMainBlockContainer"],
      .main {
        background: transparent !important;
      }
      /* Kembalikan tinggi alami konten agar bagian bawah bisa di-scroll. */
      [data-testid="stAppViewContainer"] {
        position: relative;
        z-index: 1;
        min-height: 100vh !important;
        height: auto !important;
        overflow: visible !important;
      }
      [data-testid="stMain"] {
        min-height: 0 !important;
        height: auto !important;
        overflow: visible !important;
        flex: 1 0 auto !important;
      }
      [data-testid="stMainBlockContainer"] {
        overflow: visible !important;
      }
      [data-testid="stHeader"] { background: rgba(0,0,0,0); }
      .block-container {
        max-width: 1240px;
        padding-top: 2.1rem;
        padding-bottom: 3.2rem;
      }
      .hero {
        padding: 2.1rem 2.2rem;
        border-radius: 24px;
        color: #f5fffd;
        background: linear-gradient(120deg, #102f3a 0%, #155b5d 58%, #23877b 100%);
        box-shadow: 0 18px 45px rgba(16, 47, 58, .16);
        margin-bottom: 1.15rem;
        position: relative;
        overflow: hidden;
      }
      .hero:after {
        content: "";
        position: absolute;
        width: 270px;
        height: 270px;
        right: -75px;
        top: -125px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.14);
        box-shadow: 0 0 0 28px rgba(255,255,255,.035), 0 0 0 58px rgba(255,255,255,.025);
      }
      .hero-kicker {
        color: #b7e8dd;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .8rem;
      }
      .hero h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.15rem);
        line-height: 1.08;
        letter-spacing: -.045em;
        margin: 0 0 .8rem 0;
        font-weight: 800;
      }
      .hero-copy {
        color: #d8eeea;
        max-width: 720px;
        font-size: 1rem;
        line-height: 1.65;
        margin: 0;
      }
      .hero-meta { margin-top: 1.25rem; display: flex; gap: .55rem; flex-wrap: wrap; }
      .hero-pill {
        display: inline-block;
        border: 1px solid rgba(222,255,247,.25);
        background: rgba(255,255,255,.10);
        color: #effffb;
        border-radius: 999px;
        padding: .42rem .78rem;
        font-size: .78rem;
        font-weight: 600;
      }
      .section-eyebrow {
        color: #23877b;
        text-transform: uppercase;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .14em;
        margin: .15rem 0 .3rem 0;
      }
      .section-title {
        color: #173b40;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -.025em;
        margin: 0 0 .25rem 0;
      }
      .section-copy { color: #687d80; font-size: .9rem; margin-bottom: .85rem; }
      .workspace-header {
        margin: 1.65rem 0 .9rem 0;
        padding: 0 .1rem;
      }
      .workspace-header .section-eyebrow {
        margin: 0 0 .38rem 0;
      }
      .workspace-header .section-title {
        font-size: clamp(1.45rem, 2.2vw, 1.8rem);
        line-height: 1.2;
        margin: 0 0 .38rem 0;
      }
      .workspace-header .section-copy {
        color: #607779;
        font-size: .94rem;
        line-height: 1.55;
        margin: 0;
        max-width: 760px;
      }
      .metric-card {
        background: rgba(255,255,255,.92);
        border: 1px solid #e1ece9;
        border-radius: 17px;
        padding: 1rem 1.15rem;
        box-shadow: 0 7px 22px rgba(21, 67, 68, .045);
        min-height: 112px;
      }
      .metric-label {
        color: #6c8182;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
      }
      .metric-value {
        color: #174b4b;
        font-size: 1.8rem;
        line-height: 1.2;
        font-weight: 800;
        margin-top: .38rem;
      }
      .metric-note { color: #829392; font-size: .75rem; margin-top: .2rem; }
      .panel-heading {
        color: #173b40;
        font-size: 1.08rem;
        font-weight: 800;
        line-height: 1.4;
        margin: .15rem 0 .38rem 0;
      }
      .panel-subheading {
        color: #637b7d;
        font-size: .88rem;
        line-height: 1.6;
        margin: 0 0 .8rem 0;
      }
      .result-card {
        background: linear-gradient(135deg, #103c43 0%, #176b68 100%);
        color: white;
        border-radius: 22px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 16px 36px rgba(16, 60, 67, .16);
        margin: .7rem 0 1rem 0;
      }
      .result-label {
        color: #bce7dd;
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-weight: 800;
      }
      .result-number {
        color: #ffffff;
        font-size: clamp(2.5rem, 5vw, 3.8rem);
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -.05em;
        margin: .45rem 0 .15rem 0;
      }
      .result-unit { color: #d5f0e9; font-size: .95rem; }
      .detail-chip {
        background: #f0f7f5;
        border: 1px solid #deebe7;
        color: #315c5b;
        border-radius: 12px;
        padding: .75rem .85rem;
        height: 100%;
      }
      .detail-label { color: #718785; font-size: .7rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
      .detail-value { color: #1c4647; font-size: .94rem; font-weight: 700; margin-top: .25rem; }
      div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 12px;
        font-weight: 800;
        min-height: 3rem;
      }
      div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(90deg, #176b68, #23877b);
        border: 0;
      }
      [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #dce9e6;
        border-radius: 22px;
        padding: 1.65rem 1.75rem 1.55rem 1.75rem;
        box-shadow: 0 12px 32px rgba(21, 67, 68, .075);
      }
      /* Ruang antarkomponen dibuat lega agar form lebih mudah dipindai. */
      [data-testid="stForm"] [data-testid="stVerticalBlock"] {
        gap: .62rem !important;
      }
      [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
        gap: 1.35rem !important;
        align-items: flex-start !important;
      }
      [data-testid="stForm"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: .28rem !important;
        line-height: 1.55 !important;
      }
      [data-testid="stForm"] [data-testid="stWidgetLabel"] {
        margin-bottom: .28rem !important;
      }
      [data-testid="stForm"] [data-testid="stElementContainer"] {
        margin-bottom: .08rem !important;
      }
      .form-section-divider {
        height: 1px;
        width: 100%;
        background: linear-gradient(90deg, rgba(35,135,123,.30), rgba(35,135,123,.08));
        margin: .55rem 0 .72rem 0;
      }
      /* Make widget labels and helper text readable on light backgrounds. */
      [data-testid="stWidgetLabel"] p,
      [data-testid="stWidgetLabel"] label,
      [data-testid="stForm"] label {
        color: #173b40 !important;
        font-weight: 700 !important;
        opacity: 1 !important;
      }
      [data-testid="stCaptionContainer"] p {
        color: #5c7375 !important;
        font-size: .82rem !important;
        line-height: 1.45 !important;
      }
      [data-testid="stNumberInput"] input,
      [data-testid="stDateInput"] input,
      [data-testid="stTimeInput"] input {
        color: #173b40 !important;
        background: #f8fbfa !important;
        border-color: #bdcfcc !important;
        font-weight: 600 !important;
      }
      [data-baseweb="select"] > div {
        color: #173b40 !important;
        background: #f8fbfa !important;
        border-color: #bdcfcc !important;
      }
      [data-testid="stForm"] [data-testid="stAlert"] {
        border-radius: 12px;
      }
      .input-intro {
        color: #415f61;
        background: #eef7f4;
        border: 1px solid #d8eae4;
        border-left: 4px solid #23877b;
        border-radius: 13px;
        padding: .95rem 1.15rem;
        margin: .85rem 0 1.25rem 0;
        font-size: .93rem;
        line-height: 1.7;
      }
      .input-group-title {
        color: #173b40;
        font-size: .96rem;
        font-weight: 800;
        line-height: 1.45;
        margin: .12rem 0 .48rem 0;
      }
      .input-help {
        min-height: 2.9rem;
        color: #718689;
        font-size: .84rem;
        line-height: 1.55;
        margin-top: .28rem;
      }
      [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0ebe8;
        border-radius: 15px;
        padding: .85rem 1rem;
      }
      [data-testid="stAlert"] { border-radius: 13px; }
      .footer-note {
        color: #839394;
        font-size: .78rem;
        text-align: center;
        padding: 1.2rem .5rem .2rem .5rem;
        border-top: 1px solid #e6eeec;
        margin-top: 2rem;
      }
      @media (max-width: 900px) {
        [data-testid="stForm"] { padding: 1.25rem 1.1rem; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] { gap: .8rem !important; }
      }
      @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem 1rem; }
        .hero { padding: 1.55rem 1.25rem; border-radius: 18px; }
        .input-help { min-height: 0; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">🌿 Bandung air quality lab</div>
      <h1>Memahami udara.<br>Memperkirakan esok beberapa jam ke depan.</h1>
      <p class="hero-copy">
        Dashboard simulasi prediksi konsentrasi PM2.5 untuk lima stasiun pemantauan
        di Kota Bandung, menggunakan data pengamatan historis dan model machine learning.
      </p>
      <div class="hero-meta">
        <span class="hero-pill">◉ Prediksi +3 jam</span>
        <span class="hero-pill">⌖ 5 stasiun pemantauan</span>
        <span class="hero-pill">▧ Simulasi data historis</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load model and historical data
# -----------------------------------------------------------------------------
@st.cache_resource
def get_resources():
    return load_resources()


try:
    model, metadata, history = get_resources()
except Exception as exc:
    st.error(
        "Model atau data historis belum dapat dimuat. Pastikan versi library sesuai "
        "requirements.txt dan seluruh file artifact tersedia."
    )
    st.exception(exc)
    st.stop()

history_times = history.index.get_level_values("timestamp")
min_time = pd.Timestamp(history_times.min()).to_pydatetime()
max_time = pd.Timestamp(history_times.max()).to_pydatetime()

st.markdown('<div class="section-eyebrow">Ringkasan model</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Performa model pada data uji</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Metrik evaluasi model final Tahap F dari pengujian proyek.</div>',
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Mean absolute error</div>'
        '<div class="metric-value">4.64</div><div class="metric-note">µg/m³ · semakin kecil semakin baik</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Root mean squared error</div>'
        '<div class="metric-value">6.33</div><div class="metric-note">µg/m³ · data uji</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">R² score</div>'
        '<div class="metric-value">0.713</div><div class="metric-note">Koefisien determinasi · data uji</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
with st.expander("Tentang data dan batasan aplikasi", expanded=False):
    st.write(
        "Aplikasi berjalan dalam mode demonstrasi berbasis arsip historis. "
        "Nilai sensor pada waktu pengamatan dimasukkan oleh pengguna, sedangkan "
        "fitur lag diambil dari pengamatan historis stasiun yang sama pada timestamp tepat. "
        "Aplikasi ini belum terhubung ke sensor real-time."
    )
    st.caption(
        f"Rentang arsip: {min_time:%d %b %Y %H:%M} – "
        f"{max_time:%d %b %Y %H:%M} WIB"
    )


# -----------------------------------------------------------------------------
# Prediction form
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="workspace-header">
      <div class="section-eyebrow">PREDIKSI PM2.5</div>
      <div class="section-title">Masukkan data pengamatan</div>
      <div class="section-copy">
        Pilih stasiun dan waktu pengamatan, lalu sesuaikan enam nilai sensor dan cuaca
        untuk memperkirakan konsentrasi PM2.5 tiga jam berikutnya.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("observation_form"):
    st.markdown('<div class="panel-heading">01 · Lokasi dan waktu</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subheading">Tentukan stasiun dan waktu ketika kondisi udara diamati. Pilihan tanggal harus berada dalam rentang arsip.</div>',
        unsafe_allow_html=True,
    )
    # Three aligned controls avoid an empty space below the time field.
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        station = st.selectbox(
            "Stasiun pemantauan",
            STATIONS,
            help="Pilih lokasi stasiun tempat data kondisi udara diamati.",
        )
    with c2:
        observation_date = st.date_input(
            "Tanggal pengamatan",
            value=max_time.date(),
            min_value=min_time.date(),
            max_value=max_time.date(),
            help="Pilih tanggal pengamatan yang tersedia dalam arsip historis.",
        )
    with c3:
        observation_time = st.time_input(
            "Waktu pengamatan (WIB)",
            value=max_time.time().replace(second=0, microsecond=0),
            step=timedelta(hours=1),
            help="Pilih jam ketika kondisi sensor dan cuaca diamati.",
        )

    st.markdown('<div class="form-section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">02 · Parameter sensor dan cuaca</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-subheading">Enam kolom di bawah berisi kondisi pada waktu pengamatan yang dipilih, bukan hasil prediksi.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="input-intro"><strong>Petunjuk pengisian:</strong> angka yang sudah terisi adalah contoh awal, '
        'bukan pembacaan sensor otomatis. Ganti setiap angka sesuai data pengamatan pada stasiun, tanggal, dan jam di atas. '
        'Jangan menganggap angka contoh sebagai kondisi aktual.</div>',
        unsafe_allow_html=True,
    )

    # Baris pertama: PM2.5, kelembapan, dan arah angin.
    # Baris kedua: suhu, kecepatan angin, dan curah hujan.
    # Pemisahan baris membuat ketiga kelompok pada baris kedua sejajar,
    # meskipun tinggi keterangan pada baris pertama berbeda.
    row1 = st.columns(3, gap="large")
    with row1[0]:
        st.markdown('<div class="input-group-title">Partikel udara</div>', unsafe_allow_html=True)
        pm25 = st.number_input(
            "PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=25.0, step=0.1,
            help="Konsentrasi partikel halus PM2.5 yang terukur pada stasiun dan waktu pengamatan pilihan.",
        )
        st.markdown('<div class="input-help">Jumlah partikel isi dengan hasil pengukuran PM2.5.</div>', unsafe_allow_html=True)

    with row1[1]:
        st.markdown('<div class="input-group-title">Kelembapan udara</div>', unsafe_allow_html=True)
        humidity = st.number_input(
            "Kelembapan (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1,
            help="Persentase kelembapan udara pada waktu pengamatan pilihan.",
        )
        st.markdown('<div class="input-help">Tingkat kelembapan relatif udara dalam persen.</div>', unsafe_allow_html=True)

    with row1[2]:
        st.markdown('<div class="input-group-title">Arah angin</div>', unsafe_allow_html=True)
        wind_direction = st.number_input(
            "Arah angin (derajat)", min_value=0.0, max_value=360.0, value=180.0, step=1.0,
            help="Nilai arah angin dalam derajat sesuai data pengamatan. Rentang input tetap 0–360 derajat.",
        )
        st.markdown('<div class="input-help">Arah angin yang dicatat sensor, dalam derajat.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)
    row2 = st.columns(3, gap="large")
    with row2[0]:
        st.markdown('<div class="input-group-title">Suhu udara</div>', unsafe_allow_html=True)
        temperature = st.number_input(
            "Suhu udara (°C)", min_value=-10.0, max_value=60.0, value=27.0, step=0.1,
            help="Suhu udara pada stasiun dan waktu pengamatan pilihan, dalam derajat Celsius.",
        )
        st.markdown('<div class="input-help">Temperatur udara saat pengamatan berlangsung.</div>', unsafe_allow_html=True)

    with row2[1]:
        st.markdown('<div class="input-group-title">Kecepatan angin</div>', unsafe_allow_html=True)
        wind_speed = st.number_input(
            "Kecepatan angin (m/s)", min_value=0.0, max_value=100.0, value=2.0, step=0.1,
            help="Kecepatan angin yang tercatat pada waktu pengamatan, dalam meter per detik.",
        )
        st.markdown('<div class="input-help">Seberapa cepat angin bertiup, dalam meter per detik.</div>', unsafe_allow_html=True)

    with row2[2]:
        st.markdown('<div class="input-group-title">Curah hujan</div>', unsafe_allow_html=True)
        rainfall = st.number_input(
            "Curah hujan (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1,
            help="Curah hujan pada waktu pengamatan pilihan, dalam milimeter.",
        )
        st.markdown('<div class="input-help">Jumlah curah hujan yang tercatat, dalam milimeter.</div>', unsafe_allow_html=True)

    st.write("")
    submitted = st.form_submit_button(
        "🌿  Hitung prediksi PM2.5 · +3 jam",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Prediction result
# -----------------------------------------------------------------------------
if submitted:
    observation_time_dt = datetime.combine(observation_date, observation_time)
    observation = {
        "waktu": observation_time_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "stasiun": station,
        "pm25_ugm3": pm25,
        "suhu_C": temperature,
        "kelembaban_pct": humidity,
        "kecepatan_angin_ms": wind_speed,
        "arah_angin_deg": wind_direction,
        "curah_hujan_mm": rainfall,
    }

    errors = []
    if observation_time_dt < min_time or observation_time_dt > max_time:
        errors.append("Waktu pengamatan berada di luar rentang arsip historis.")
    if humidity < 0 or humidity > 100:
        errors.append("Kelembapan harus berada pada rentang 0–100%.")
    if wind_speed < 0:
        errors.append("Kecepatan angin tidak boleh negatif.")
    if not 0 <= wind_direction <= 360:
        errors.append("Arah angin harus berada pada rentang 0–360 derajat.")
    if rainfall < 0 or pm25 < 0:
        errors.append("PM2.5 dan curah hujan tidak boleh negatif.")

    st.markdown('<div class="section-eyebrow">Prediction output</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Hasil prakiraan</div>', unsafe_allow_html=True)

    if errors:
        for err in errors:
            st.error(err)
    else:
        try:
            prediction, target_time, lag_status = predict_pm25(
                observation, model, metadata, history
            )
            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-label">Estimasi konsentrasi PM2.5 · {station}</div>'
                f'<div class="result-number">{prediction:.2f}</div>'
                f'<div class="result-unit">µg/m³ &nbsp;|&nbsp; Prediksi untuk {target_time:%d %b %Y, %H:%M} WIB</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            d1, d2 = st.columns(2)
            with d1:
                st.markdown(
                    f'<div class="detail-chip"><div class="detail-label">Waktu pengamatan</div>'
                    f'<div class="detail-value">{observation_time_dt:%d %b %Y · %H:%M} WIB</div></div>',
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    f'<div class="detail-chip"><div class="detail-label">Waktu target (+3 jam)</div>'
                    f'<div class="detail-value">{target_time:%d %b %Y · %H:%M} WIB</div></div>',
                    unsafe_allow_html=True,
                )

            missing_lags = [name for name, available in lag_status.items() if not available]
            if missing_lags:
                st.warning(
                    "Sebagian data lag tidak ditemukan pada timestamp historis yang tepat: "
                    + ", ".join(missing_lags)
                    + ". Pipeline model akan menangani nilai kosong menggunakan imputasi "
                    "yang sudah dipelajari saat training."
                )
            else:
                st.success("Ketiga fitur lag ditemukan pada timestamp historis yang tepat.")

            with st.expander("Lihat ringkasan input", expanded=False):
                input_summary = pd.DataFrame(
                    {
                        "Parameter": [
                            "Stasiun", "PM2.5 (µg/m³)", "Suhu (°C)",
                            "Kelembapan (%)", "Kecepatan angin (m/s)",
                            "Arah angin (derajat)", "Curah hujan (mm)",
                        ],
                        "Nilai": [
                            station, f"{pm25:.1f}", f"{temperature:.1f}",
                            f"{humidity:.1f}", f"{wind_speed:.1f}",
                            f"{wind_direction:.0f}", f"{rainfall:.1f}",
                        ],
                    }
                )
                st.dataframe(input_summary, hide_index=True, use_container_width=True)

            st.caption(
                "Hasil ini merupakan simulasi berbasis arsip historis, bukan pengukuran "
                "atau prakiraan operasional real-time. Kategori kualitas udara tidak "
                "ditampilkan karena standar kategori untuk tugas ini belum ditetapkan."
            )
        except Exception as exc:
            st.error("Prediksi gagal dijalankan.")
            st.exception(exc)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    f'<div class="footer-note">'
    f'Model: {metadata.get("selected_model", "sesuai metadata")} '
    f'&nbsp;·&nbsp; Horizon: {metadata.get("horizon_hours", 3)} jam '
    f'&nbsp;·&nbsp; Machine Learning Project — Kasus 06<br>'
    f'Arsip historis: {min_time:%d %b %Y} – {max_time:%d %b %Y}'
    f'</div>',
    unsafe_allow_html=True,
)
