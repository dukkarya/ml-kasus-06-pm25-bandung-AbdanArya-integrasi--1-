import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_feature_row, FEATURE_COLUMNS


def test_exact_timestamp_lags():
    t = pd.Timestamp("2026-06-30 20:00:00")
    rows = [
        ("Dago", t - pd.Timedelta(hours=1), 11.0),
        ("Dago", t - pd.Timedelta(hours=3), 13.0),
        ("Dago", t - pd.Timedelta(hours=24), 24.0),
    ]
    hist = pd.DataFrame(rows, columns=["stasiun_std", "timestamp", "pm25_ugm3"])
    hist = hist.set_index(["stasiun_std", "timestamp"])
    observation = {
        "waktu": str(t), "stasiun": "Dago", "pm25_ugm3": 20,
        "suhu_C": 25, "kelembaban_pct": 70, "kecepatan_angin_ms": 1.2,
        "arah_angin_deg": 90, "curah_hujan_mm": 0,
    }
    X, status = build_feature_row(observation, hist)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.loc[0, "pm25_lag_1h"] == 11
    assert X.loc[0, "pm25_lag_3h"] == 13
    assert X.loc[0, "pm25_lag_24h"] == 24
    assert all(status.values())
