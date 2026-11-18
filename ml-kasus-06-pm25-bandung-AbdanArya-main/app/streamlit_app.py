            "Kelembapan (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1
        )
        wind_speed = st.number_input(
            "Kecepatan angin (m/s)", min_value=0.0, max_value=100.0, value=2.0, step=0.1
        )
    with c3:
        wind_direction = st.number_input(
            "Arah angin (derajat)", min_value=0.0, max_value=360.0, value=180.0, step=1.0
        )
        rainfall = st.number_input(
            "Curah hujan (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1
        )

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
