"""Memuat artifact dan menjalankan prediksi PM2.5 tiga jam ke depan."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_feature_row

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"
HISTORY_PATH = ROOT / "data" / "processed" / "pm25_bandung_clean.csv"


def load_resources():
    """Load model, metadata, dan dataset historis bersih."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Artifact model tidak ditemukan: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {METADATA_PATH}")
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Dataset historis bersih tidak ditemukan: {HISTORY_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(MODEL_PATH)

    history = pd.read_csv(HISTORY_PATH)
    required = {"waktu", "stasiun_std", "pm25_ugm3"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Kolom dataset historis tidak lengkap: {sorted(missing)}")

    history["timestamp"] = pd.to_datetime(history["waktu"], errors="coerce")
    history["pm25_ugm3"] = pd.to_numeric(history["pm25_ugm3"], errors="coerce")
    history = history.dropna(subset=["timestamp", "stasiun_std"])
    history = history.set_index(["stasiun_std", "timestamp"]).sort_index()

    return model, metadata, history


def predict_pm25(observation: dict, model, metadata: dict, history: pd.DataFrame):
    """Prediksi target PM2.5 pada t+3 jam tanpa fitting ulang."""
    X, lag_status = build_feature_row(observation, history)

    expected = metadata.get("feature_columns")
    if expected and list(X.columns) != list(expected):
        raise ValueError("Urutan/nama fitur input tidak cocok dengan metadata model.")

    prediction = float(model.predict(X)[0])
    target_time = pd.Timestamp(observation["waktu"]) + pd.Timedelta(
        hours=int(metadata.get("horizon_hours", 3))
    )
    return prediction, target_time, lag_status
