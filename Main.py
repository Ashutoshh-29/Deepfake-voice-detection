from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import io

app = FastAPI(title="Voicecheck API")

# TODO: once deployed, replace "*" with your actual frontend URL
# e.g. ["https://your-frontend.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_features(audio_bytes: bytes):
    """Load audio and extract MFCC features for the model."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Voicecheck API is running"}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    try:
        features = extract_features(audio_bytes)
    except Exception as e:
        return {"error": f"Could not process audio: {str(e)}"}

    # --- Placeholder logic ---
    # Replace this with: load your trained model once at startup,
    # then run model.predict(features) here instead of the random guess below.
    #
    # Example once you have a trained model saved as model.pkl or model.pt:
    #   prediction = model.predict([features])[0]
    #   confidence = model.predict_proba([features])[0].max()
    fake_score = float(np.clip(np.mean(features) % 1, 0.05, 0.95))
    label = "real" if fake_score < 0.5 else "fake"
    confidence = fake_score if label == "fake" else 1 - fake_score

    return {"label": label, "confidence": round(confidence, 2)}
