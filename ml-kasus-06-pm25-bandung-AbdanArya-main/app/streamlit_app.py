import sys
from pathlib import Path
from datetime import datetime, date, time, timedelta

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.predictor import load_resources, predict_pm25
from src.features import STATIONS

st.set_page_config(
    page_title="Prakiraan PM2.5 Bandung",
    page_icon="🌫️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Prakiraan PM2.5 Bandung")
st.caption(
    "Prakiraan konsentrasi PM2.5 tiga jam ke depan pada lima stasiun pemantauan."
)
st.info(
    "Mode demonstrasi berbasis arsip historis: pilih waktu yang tercakup dalam dataset. "
    "Nilai sensor saat ini diisi pengguna, sedangkan fitur lag diambil dari pengamatan "
    "historis stasiun yang sama pada timestamp tepat. Aplikasi ini belum terhubung ke sensor real-time."
)

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

# Dataset menentukan rentang waktu simulasi yang dapat dipilih.
history_times = history.index.get_level_values("timestamp")
min_time = pd.Timestamp(history_times.min()).to_pydatetime()
max_time = pd.Timestamp(history_times.max()).to_pydatetime()

st.caption(
    f"Arsip historis tersedia: {min_time:%d %b %Y %H:%M} – "
    f"{max_time:%d %b %Y %H:%M} WIB"
)

with st.form("observation_form"):
    st.subheader("Data pengamatan")
    c1, c2 = st.columns(2)
    with c1:
        station = st.selectbox("Stasiun", STATIONS)
        observation_date = st.date_input(
            "Tanggal pengamatan",
            value=max_time.date(),
            min_value=min_time.date(),
            max_value=max_time.date(),
        )
    with c2:
        observation_time = st.time_input(
            "Waktu pengamatan (WIB)",
            value=max_time.time().replace(second=0, microsecond=0),
            step=timedelta(hours=1),
        )

    st.markdown("**Parameter sensor pada waktu pengamatan**")
    c1, c2 = st.columns(2)
    with c1:
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=25.0, step=0.1)
        temperature = st.number_input("Suhu udara (°C)", min_value=-10.0, max_value=60.0, value=27.0, step=0.1)
        humidity = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    with c2:
        wind_speed = st.number_input("Kecepatan angin (m/s)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
        wind_direction = st.number_input("Arah angin (derajat)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
        rainfall = st.number_input("Curah hujan (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)

    submitted = st.form_submit_button("Prediksi PM2.5 +3 jam", type="primary", use_container_width=True)

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

    if errors:
        for err in errors:
            st.error(err)
    else:
        try:
            prediction, target_time, lag_status = predict_pm25(
                observation, model, metadata, history
            )
            st.success("Prediksi berhasil dihitung menggunakan model final Tahap F.")
            st.subheader("Hasil prakiraan")
            st.metric("Prediksi PM2.5", f"{prediction:.2f} µg/m³")
            st.write(f"**Stasiun:** {station}")
            st.write(f"**Waktu pengamatan:** {observation_time_dt:%d-%m-%Y %H:%M} WIB")
            st.write(f"**Waktu target:** {target_time:%d-%m-%Y %H:%M} WIB")

            missing_lags = [name for name, available in lag_status.items() if not available]
            if missing_lags:
                st.warning(
                    "Sebagian data lag tidak ditemukan pada timestamp historis yang tepat: "
                    + ", ".join(missing_lags)
                    + ". Pipeline model akan menangani nilai kosong menggunakan imputasi "
                    "yang sudah dipelajari saat training."
                )
            else:
                st.caption("Ketiga fitur lag ditemukan pada timestamp historis yang tepat.")

            st.caption(
                "Hasil ini adalah simulasi berbasis arsip, bukan pengukuran atau prakiraan "
                "operasional real-time. Kategori kualitas udara belum ditampilkan karena "
                "rentang kategori perlu dikonfirmasi sesuai standar yang digunakan dalam tugas."
            )
        except Exception as exc:
            st.error("Prediksi gagal dijalankan.")
            st.exception(exc)

st.divider()
st.caption(
    f"Model: {metadata.get('selected_model', 'sesuai metadata')} · "
    f"Horizon: {metadata.get('horizon_hours', 3)} jam · "
    "Machine Learning Project — Kasus 06"
)
