import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_def import N_SAMPLES, SpoofCNN, fix_length, load_audio, to_logmel

app = FastAPI(title="Voicecheck API")

# TODO: after deploying, replace "*" with your frontend URL,
# e.g. ["https://your-project.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
model = SpoofCNN()
model_loaded = False
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    model_loaded = True


def predict_probs(y):
    """Split long audio into 4s windows (max 5), average the predictions."""
    chunks = [y[i:i + N_SAMPLES] for i in range(0, len(y), N_SAMPLES)][:5]
    if len(chunks) > 1 and len(chunks[-1]) < N_SAMPLES // 2:
        chunks = chunks[:-1]  # drop a tiny trailing piece
    feats = np.stack([to_logmel(fix_length(c)) for c in chunks])
    x = torch.tensor(feats).unsqueeze(1)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).mean(0)  # [p_real, p_fake]


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"error": "Model file model.pt not found on server"})

    audio_bytes = await audio.read()
    try:
        y = load_audio(io.BytesIO(audio_bytes))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read audio: {e}"})

    probs = predict_probs(y)
    p_real, p_fake = probs[0].item(), probs[1].item()
    label = "fake" if p_fake > p_real else "real"
    return {"label": label, "confidence": round(max(p_real, p_fake), 3)}
