"""Evaluate the trained CNN and compare it with a Random Forest baseline.

Computes accuracy, precision, recall, F1, confusion matrix and Equal Error
Rate (EER) on the ASVspoof 2019 LA eval set, and saves:
  - results.json              (all numbers, for your report)
  - confusion_matrix_cnn.png
  - confusion_matrix_rf.png

Usage (run in Colab after training, with model.pt in the same folder):
  python evaluate.py --root /content/LA --train-per-class 1000 --eval-per-class 800

Positive class = "fake" (label 1), so precision/recall describe fake detection.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)

from model_def import SpoofCNN, fix_length, load_audio, to_logmel, to_mfcc_stats
from train import read_protocol, sample_balanced


def get_items(root, protocol_file, per_class):
    path = os.path.join(root, "ASVspoof2019_LA_cm_protocols", protocol_file)
    return sample_balanced(read_protocol(path), per_class)


def extract(root, split, items, want_logmel):
    """Load each clip once and compute MFCC stats (and optionally log-mel)."""
    mfccs, mels, labels = [], [], []
    for n, (file_id, label) in enumerate(items, start=1):
        path = os.path.join(root, f"ASVspoof2019_LA_{split}", "flac", file_id + ".flac")
        try:
            y = fix_length(load_audio(path))
        except Exception as e:
            print(f"  skipped {path}: {e}")
            continue
        mfccs.append(to_mfcc_stats(y))
        if want_logmel:
            mels.append(to_logmel(y))
        labels.append(label)
        if n % 500 == 0:
            print(f"  [{split}] processed {n}/{len(items)} files")
    mels = np.stack(mels) if want_logmel else None
    return np.stack(mfccs), mels, np.array(labels)


@torch.no_grad()
def cnn_fake_probs(model, mels, device):
    model.eval()
    out = []
    for i in range(0, len(mels), 64):
        x = torch.tensor(mels[i:i + 64]).unsqueeze(1).to(device)
        out.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def compute_eer(y_true, scores):
    """Equal Error Rate: the point where false accept rate == false reject rate."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def report(name, y_true, p_fake):
    y_pred = (p_fake > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "eer": compute_eer(y_true, p_fake),
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy : {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}  (of clips flagged fake, how many really were)")
    print(f"Recall   : {result['recall']:.4f}  (of all fake clips, how many were caught)")
    print(f"F1-score : {result['f1']:.4f}")
    print(f"EER      : {result['eer']:.4f}  (lower is better)")
    print("Confusion matrix (rows = true, cols = predicted; order: real, fake):")
    print(cm)
    return result, cm


def save_confusion_png(cm, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed, skipping confusion matrix image)")
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Real", "Fake"])
    ax.set_yticks([0, 1], labels=["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the ASVspoof2019 LA folder")
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--train-per-class", type=int, default=1000,
                    help="clips per class used to fit the Random Forest baseline")
    ap.add_argument("--eval-per-class", type=int, default=800)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Extracting baseline training features (train split)...")
    train_items = get_items(args.root, "ASVspoof2019.LA.cm.train.trn.txt", args.train_per_class)
    Xtr, _, ytr = extract(args.root, "train", train_items, want_logmel=False)

    print("Extracting eval features (eval split, unseen attack types)...")
    eval_items = get_items(args.root, "ASVspoof2019.LA.cm.eval.trl.txt", args.eval_per_class)
    Xev_mfcc, Xev_mel, yev = extract(args.root, "eval", eval_items, want_logmel=True)

    # --- CNN ---
    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    cnn_scores = cnn_fake_probs(model, Xev_mel, device)
    cnn_res, cnn_cm = report("CNN (log-mel spectrogram)", yev, cnn_scores)

    # --- Random Forest baseline ---
    print("\nTraining Random Forest baseline on MFCC statistics...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    rf_scores = rf.predict_proba(Xev_mfcc)[:, 1]
    rf_res, rf_cm = report("Random Forest (MFCC baseline)", yev, rf_scores)

    # --- Save for the report ---
    with open("results.json", "w") as f:
        json.dump({"cnn": cnn_res, "random_forest": rf_res,
                   "eval_clips": int(len(yev)), "baseline_train_clips": int(len(ytr))}, f, indent=2)
    print("\nSaved results.json")
    save_confusion_png(cnn_cm, "CNN", "confusion_matrix_cnn.png")
    save_confusion_png(rf_cm, "Random Forest", "confusion_matrix_rf.png")

    print("\n--- Comparison ---")
    print(f"{'Model':<28}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'EER':>8}")
    for name, r in [("CNN (log-mel)", cnn_res), ("Random Forest (MFCC)", rf_res)]:
        print(f"{name:<28}{r['accuracy']:>8.3f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['eer']:>8.3f}")


if __name__ == "__main__":
    main()