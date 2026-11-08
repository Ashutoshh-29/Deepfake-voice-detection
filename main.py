"""Deepfake Voice Detection System - FastAPI app.

Serves the web pages (Jinja2 + Bootstrap), runs the trained CNN on uploaded
audio, and stores each prediction in SQLite.
"""
import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import Prediction, SessionLocal, init_db
from model_def import N_SAMPLES, SpoofCNN, fix_length, load_audio, to_logmel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pt"

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ADMIN_KEY = os.environ.get("ADMIN_KEY")  # optional: protects /history if set

app = FastAPI(title="Deepfake Voice Detection System")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
init_db()

# ---- Load the trained model once at startup ----
model = SpoofCNN()
model_loaded = False
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model_loaded = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def predict_probs(y):
    """Split long audio into 4s windows (max 5) and average the predictions.

    Returns a tensor [p_real, p_fake].
    """
    chunks = [y[i:i + N_SAMPLES] for i in range(0, len(y), N_SAMPLES)][:5]
    if len(chunks) > 1 and len(chunks[-1]) < N_SAMPLES // 2:
        chunks = chunks[:-1]  # drop a tiny trailing piece
    feats = np.stack([to_logmel(fix_length(c)) for c in chunks])
    x = torch.tensor(feats).unsqueeze(1)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).mean(0)


# ---- Pages ----
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request, "index.html",
        {"model_loaded": model_loaded, "max_mb": MAX_UPLOAD_BYTES // (1024 * 1024)},
    )


@app.get("/history", response_class=HTMLResponse)
def history(
    request: Request,
    key: str | None = Query(None),
    db: Session = Depends(get_db),
):
    if ADMIN_KEY and key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin key required: open /history?key=YOUR_KEY")

    rows = db.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(100)).all()
    total = db.scalar(select(func.count()).select_from(Prediction)) or 0
    fake_total = db.scalar(
        select(func.count()).select_from(Prediction).where(Prediction.label == "fake")
    ) or 0
    return templates.TemplateResponse(
        request, "history.html",
        {"rows": rows, "total": total, "fake_total": fake_total, "real_total": total - fake_total},
    )


# ---- API ----
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(audio: UploadFile = File(...), db: Session = Depends(get_db)):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"error": "Model file (model.pt) not found on the server."})

    # 1. Validate the upload (format + size)
    filename = os.path.basename(audio.filename or "upload")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"},
        )

    data = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        return JSONResponse(status_code=400, content={"error": "The uploaded file is empty."})
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
        )

    # 2. Decode audio
    try:
        y = load_audio(io.BytesIO(data))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Could not decode this audio file. Is it a valid recording?"})

    # 3. Classify
    probs = predict_probs(y)
    p_real, p_fake = probs[0].item(), probs[1].item()
    label = "fake" if p_fake > p_real else "real"
    confidence = round(max(p_real, p_fake), 4)

    # 4. Store in history
    db.add(Prediction(filename=filename[:255], label=label, confidence=confidence))
    db.commit()

    return {"label": label, "confidence": confidence, "filename": filename}