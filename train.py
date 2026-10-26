"""Train the real-vs-fake voice CNN on ASVspoof 2019 (Logical Access).

Expected folder layout (this is how the dataset downloads):

  LA/
    ASVspoof2019_LA_train/flac/*.flac
    ASVspoof2019_LA_dev/flac/*.flac
    ASVspoof2019_LA_eval/flac/*.flac
    ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt

Usage:
  python train.py --root /path/to/LA --per-class 2500 --epochs 15

Run it on Google Colab with a GPU runtime for speed.
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_def import SpoofCNN, fix_length, load_audio, to_logmel


def read_protocol(path):
    """Each line: SPEAKER FILE_ID - ATTACK_ID LABEL  (label = bonafide/spoof)."""
    items = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            label = 0 if parts[-1] == "bonafide" else 1
            items.append((parts[1], label))
    return items


def sample_balanced(items, per_class, seed=42):
    """ASVspoof is heavily skewed toward spoof; take equal numbers of each."""
    rng = random.Random(seed)
    real = [i for i in items if i[1] == 0]
    fake = [i for i in items if i[1] == 1]
    rng.shuffle(real)
    rng.shuffle(fake)
    return real[:per_class] + fake[:per_class]


def build_split(root, split, protocol_file, per_class):
    protocol_path = os.path.join(root, "ASVspoof2019_LA_cm_protocols", protocol_file)
    items = sample_balanced(read_protocol(protocol_path), per_class)
    random.Random(0).shuffle(items)

    feats, labels = [], []
    for n, (file_id, label) in enumerate(items, start=1):
        path = os.path.join(root, f"ASVspoof2019_LA_{split}", "flac", file_id + ".flac")
        try:
            feats.append(to_logmel(fix_length(load_audio(path))))
            labels.append(label)
        except Exception as e:
            print(f"  skipped {path}: {e}")
        if n % 500 == 0:
            print(f"  [{split}] processed {n}/{len(items)} files")

    x = torch.tensor(np.stack(feats)).unsqueeze(1)  # (N, 1, mels, frames)
    y = torch.tensor(labels)
    return TensorDataset(x, y)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    tp = np.zeros(2)
    count = np.zeros(2)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
        for c in (0, 1):
            mask = y == c
            count[c] += mask.sum().item()
            tp[c] += (pred[mask] == c).sum().item()
    acc = correct / max(total, 1)
    real_recall = tp[0] / max(count[0], 1)
    fake_recall = tp[1] / max(count[1], 1)
    return acc, real_recall, fake_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the ASVspoof2019 LA folder")
    ap.add_argument("--per-class", type=int, default=2500,
                    help="clips per class for training (dev/eval use a smaller share)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("Extracting features (this is the slow part, only happens once)...")
    train_ds = build_split(args.root, "train", "ASVspoof2019.LA.cm.train.trn.txt", args.per_class)
    dev_ds = build_split(args.root, "dev", "ASVspoof2019.LA.cm.dev.trl.txt", args.per_class // 3)
    eval_ds = build_split(args.root, "eval", "ASVspoof2019.LA.cm.eval.trl.txt", args.per_class // 3)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=64)
    eval_loader = DataLoader(eval_ds, batch_size=64)

    model = SpoofCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)

        train_loss = running / len(train_ds)
        acc, real_r, fake_r = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | dev acc {acc:.3f} "
              f"(real recall {real_r:.3f}, fake recall {fake_r:.3f})")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            print(f"  saved best model -> {args.out}")

    # Final check on the eval set, which contains attack types NOT seen in training.
    model.load_state_dict(torch.load(args.out, map_location=device))
    acc, real_r, fake_r = evaluate(model, eval_loader, device)
    print(f"\nFINAL eval acc {acc:.3f} (real recall {real_r:.3f}, fake recall {fake_r:.3f})")
    print("Eval uses unseen attack types, so expect this to be lower than dev. "
          "That gap is a great thing to discuss in your report.")


if __name__ == "__main__":
    main()
