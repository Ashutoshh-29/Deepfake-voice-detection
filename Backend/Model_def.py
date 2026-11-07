"""Shared preprocessing + model definition.
 
Both train.py and main.py import from here, so the audio is always
processed the exact same way at training time and at prediction time.
 
Labels: 0 = real (bonafide), 1 = fake (spoof)
"""
import numpy as np
import librosa
import torch.nn as nn
 
SR = 16000                      # sample rate
DURATION = 4                    # seconds per clip fed to the model
N_SAMPLES = SR * DURATION
N_MELS = 64
HOP = 512
N_FFT = 1024
 
 
def load_audio(source):
    """source can be a file path or a file-like object (BytesIO)."""
    y, _ = librosa.load(source, sr=SR, mono=True)
    if len(y) == 0:
        raise ValueError("Audio file is empty")
    return y
 
 
def fix_length(y):
    """Pad (by repeating) or crop to exactly DURATION seconds."""
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)), mode="wrap")
    return y[:N_SAMPLES]
 
 
def to_logmel(y):
    """Waveform -> normalized log-mel spectrogram, shape (N_MELS, frames)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)
 
 
class SpoofCNN(nn.Module):
    """Small CNN that treats the spectrogram like a 1-channel image."""
 
    def __init__(self):
        super().__init__()
 
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
 
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 2))
 
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
 
