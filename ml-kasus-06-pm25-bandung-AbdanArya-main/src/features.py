"""Feature engineering untuk aplikasi prediksi PM2.5.

Rumus fitur waktu dan arah angin diselaraskan dengan daftar fitur model Tahap F.
Fitur lag dicari dengan timestamp historis yang tepat pada stasiun yang sama.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATIONS = [
    "Dago",
    "Soekarno-Hatta",
    "Cibeureum",
    "Antapani",
    "Ujungberung",
]

FEATURE_COLUMNS = [
    "pm25_ugm3",
    "suhu_C",
    "kelembaban_pct",
    "kecepatan_angin_ms",
    "curah_hujan_mm",
    "arah_angin_sin",
    "arah_angin_cos",
    "jam_sin",
    "jam_cos",
    "hari_minggu_sin",
    "hari_minggu_cos",
    "bulan_sin",
    "bulan_cos",
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_24h",
    "stasiun_std",
]


def build_feature_row(observation: dict, history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bangun satu baris fitur. History harus berisi data bersih dengan timestamp."""
    station = observation["stasiun"]
    timestamp = pd.Timestamp(observation["waktu"])

    if station not in STATIONS:
        raise ValueError(f"Stasiun tidak dikenal: {station}")

    row = {
        "pm25_ugm3": float(observation["pm25_ugm3"]),
        "suhu_C": float(observation["suhu_C"]),
        "kelembaban_pct": float(observation["kelembaban_pct"]),
        "kecepatan_angin_ms": float(observation["kecepatan_angin_ms"]),
        "curah_hujan_mm": float(observation["curah_hujan_mm"]),
        "stasiun_std": station,
    }

    # Siklis arah angin dan kalender, dihitung dari waktu pengamatan t.
    wind_rad = np.deg2rad(float(observation["arah_angin_deg"]))
    row["arah_angin_sin"] = float(np.sin(wind_rad))
    row["arah_angin_cos"] = float(np.cos(wind_rad))

    hour = timestamp.hour
    weekday = timestamp.dayofweek
    month = timestamp.month
    row["jam_sin"] = float(np.sin(2 * np.pi * hour / 24))
    row["jam_cos"] = float(np.cos(2 * np.pi * hour / 24))
    row["hari_minggu_sin"] = float(np.sin(2 * np.pi * weekday / 7))
    row["hari_minggu_cos"] = float(np.cos(2 * np.pi * weekday / 7))
    row["bulan_sin"] = float(np.sin(2 * np.pi * (month - 1) / 12))
    row["bulan_cos"] = float(np.cos(2 * np.pi * (month - 1) / 12))

    # Lookup timestamp eksak, bukan baris terdekat / shift berdasarkan urutan.
    hist = history
    if not isinstance(hist.index, pd.MultiIndex):
        hist = hist.set_index(["stasiun_std", "timestamp"])
    lag_status = {}
    for lag_hours in (1, 3, 24):
        lag_time = timestamp - pd.Timedelta(hours=lag_hours)
        key = (station, lag_time)
        try:
            value = hist.loc[key, "pm25_ugm3"]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            value = pd.to_numeric(value, errors="coerce")
        except KeyError:
            value = np.nan
        row[f"pm25_lag_{lag_hours}h"] = float(value) if pd.notna(value) else np.nan
        lag_status[f"pm25_lag_{lag_hours}h"] = bool(pd.notna(value))

    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return X, lag_status
