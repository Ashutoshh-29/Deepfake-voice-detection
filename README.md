# Deepfake Voice Detection System

Web app that classifies an uploaded voice recording as **real (bonafide)** or **AI-generated (spoofed)**.
Log-mel spectrogram features (Librosa) feed a CNN (PyTorch) trained on ASVspoof 2019 LA.
Built with FastAPI, Jinja2 + Bootstrap, and SQLite (SQLAlchemy) for prediction history.

## Project layout
| File | Purpose |
|---|---|
| `main.py` | FastAPI app: pages, upload validation, prediction, history |
| `model_def.py` | Audio preprocessing + CNN (shared by training and the app) |
| `database.py` | SQLAlchemy model for stored predictions |
| `train.py` | Trains the CNN on ASVspoof 2019 LA (run on Colab GPU) |
| `evaluate.py` | Accuracy/precision/recall/F1/confusion matrix/EER + Random Forest baseline |
| `test_app.py` | Unit tests (`pytest -v`) |
| `templates/` | Bootstrap pages (detect, history) |
| `model.pt` | Trained weights (produced by `train.py`) |

## Run locally
```
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 . History page: `/history`.
Optional: set env var `ADMIN_KEY=secret` to protect history (`/history?key=secret`).

## Train and evaluate (Google Colab, GPU runtime)
```
pip install librosa soundfile
python train.py --root /content/LA --per-class 2500 --epochs 15
python evaluate.py --root /content/LA --train-per-class 1000 --eval-per-class 800
```
Download `model.pt` and put it next to `main.py`.

## Tests
```
pip install pytest httpx
pytest -v
```

## Deploy on Render
- Root Directory: *(leave empty)*
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Free tier note: the disk is temporary, so the history database resets on each redeploy/restart.