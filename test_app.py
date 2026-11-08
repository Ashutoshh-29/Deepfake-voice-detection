"""Unit tests for the Deepfake Voice Detection System.

Run with:
  pip install pytest httpx
  pytest -v
"""
import io
import os

# Use a throwaway database for tests (must be set BEFORE importing the app)
os.environ["DATABASE_URL"] = "sqlite:///./test_predictions.db"

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import main
from database import Prediction, SessionLocal
from model_def import N_MELS, N_SAMPLES, fix_length, to_logmel, to_mfcc_stats

client = TestClient(main.app)


def make_wav_bytes(seconds=2, sr=16000):
    """Generate a simple sine-wave WAV file in memory."""
    t = np.linspace(0, seconds, sr * seconds, endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


# ---------- Feature extraction ----------
def test_fix_length_pads_short_audio():
    y = np.random.randn(1000).astype("float32")
    assert len(fix_length(y)) == N_SAMPLES


def test_fix_length_crops_long_audio():
    y = np.random.randn(N_SAMPLES * 2).astype("float32")
    assert len(fix_length(y)) == N_SAMPLES


def test_logmel_shape():
    feat = to_logmel(np.random.randn(N_SAMPLES).astype("float32"))
    assert feat.shape[0] == N_MELS
    assert feat.shape[1] == 126  # 64000 samples / hop 512 + 1


def test_mfcc_stats_length():
    feat = to_mfcc_stats(np.random.randn(N_SAMPLES).astype("float32"))
    assert feat.shape == (80,)  # 40 means + 40 stds


# ---------- Database ----------
def test_database_insert_and_read():
    db = SessionLocal()
    try:
        db.add(Prediction(filename="unit_test.wav", label="real", confidence=0.91))
        db.commit()
        row = db.query(Prediction).filter_by(filename="unit_test.wav").first()
        assert row is not None
        assert row.label == "real"
        assert row.created_at is not None
    finally:
        db.close()


# ---------- API ----------
def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert "model_loaded" in res.json()


def test_home_page_loads():
    assert client.get("/").status_code == 200


def test_history_page_loads():
    assert client.get("/history").status_code == 200


@pytest.mark.skipif(not main.model_loaded, reason="model.pt not present")
def test_rejects_unsupported_extension():
    res = client.post("/predict", files={"audio": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "Unsupported" in res.json()["error"]


@pytest.mark.skipif(not main.model_loaded, reason="model.pt not present")
def test_rejects_empty_file():
    res = client.post("/predict", files={"audio": ("empty.wav", b"", "audio/wav")})
    assert res.status_code == 400


@pytest.mark.skipif(not main.model_loaded, reason="model.pt not present")
def test_rejects_corrupt_audio():
    res = client.post("/predict", files={"audio": ("bad.wav", b"not really audio", "audio/wav")})
    assert res.status_code == 400


@pytest.mark.skipif(not main.model_loaded, reason="model.pt not present")
def test_valid_audio_returns_prediction():
    res = client.post("/predict", files={"audio": ("tone.wav", make_wav_bytes(), "audio/wav")})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] in {"real", "fake"}
    assert 0.5 <= body["confidence"] <= 1.0


def test_prediction_unavailable_without_model(monkeypatch):
    monkeypatch.setattr(main, "model_loaded", False)
    res = client.post("/predict", files={"audio": ("tone.wav", make_wav_bytes(), "audio/wav")})
    assert res.status_code == 503