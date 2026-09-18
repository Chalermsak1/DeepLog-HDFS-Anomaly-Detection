#!/usr/bin/env python3
"""
DeepLog Training Pipeline
=========================
Trains a 2-layer LSTM model for next-event log anomaly detection on HDFS log sequences.

Key Engineering Highlights:
- Canonical Event-ID Mapping: Derived strictly from the training dataset to avoid label leakage
  and ensure vocabulary consistency across training, validation, and testing.
- Hardware Acceleration: Automatic device detection (Apple Silicon MPS, NVIDIA CUDA, or CPU).
- Deterministic Validation Split: 90/10 train/validation split using fixed random seeds.
- Checkpointing: Evaluates validation loss and saves the best model state dictionary.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLog LSTM model on HDFS log sequences.")
    parser.add_argument("--data_dir", type=str, default="examples/data", help="Directory containing HDFS data splits.")
    parser.add_argument("--output_model", type=str, default="deeplog_model_improved_v1.pt", help="Output path for best checkpoint.")
    parser.add_argument("--context_length", type=int, default=10, help="Context sequence length (default: 10).")
    parser.add_argument("--input_size", type=int, default=30, help="Event vocabulary size (default: 30).")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden layer dimension (default: 128).")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked LSTM layers (default: 2).")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40).")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size (default: 256).")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate (default: 0.001).")
    parser.add_argument("--val_split", type=float, default=0.10, help="Validation split fraction (default: 0.10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = select_device()

    torch.manual_seed(args.seed)
    train_file = os.path.join(args.data_dir, "hdfs_train")

    print("=" * 75)
    print("  🚀 DEEPLOG LSTM MODEL TRAINING PIPELINE")
    print("=" * 75)
    print(f"⚡ Compute Device      : {device}")
    print(f"📦 Training Dataset     : {train_file}")
    print(f"🧠 Architecture         : {args.num_layers}-Layer LSTM (Hidden Size: {args.hidden_size})")
    print(f"📏 Context Window (h)   : {args.context_length}")
    print(f"🔄 Epochs / Batch Size  : {args.epochs} / {args.batch_size}")
    print(f"🎯 Learning Rate (Adam) : {args.lr}")
    print(f"💾 Output Checkpoint    : {args.output_model}")
    print("=" * 75)

    # 1. Preprocess Training Data & Extract Canonical Vocabulary Mapping
    print("\n[1/4] Preprocessing training sequences & extracting canonical mapping...")
    preprocessor = Preprocessor(length=args.context_length, timeout=float("inf"))
    X_train_raw, y_train_raw, _, train_mapping = preprocessor.text(train_file, verbose=False)

    total_samples = X_train_raw.shape[0]
    val_size = int(total_samples * args.val_split)
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total_samples, generator=generator)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    X_train, y_train = X_train_raw[train_idx], y_train_raw[train_idx]
    X_val, y_val = X_train_raw[val_idx], y_train_raw[val_idx]

    print(f"   - Total Training Sequences   : {train_size:,}")
    print(f"   - Total Validation Sequences : {val_size:,}")
    print(f"   - Canonical Event Vocabulary : {len(train_mapping)} distinct events mapped")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)

    # 2. Instantiate Model
    print("\n[2/4] Initializing DeepLog model...")
    deeplog = DeepLog(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.input_size,
        num_layers=args.num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in deeplog.parameters() if p.requires_grad)
    print(f"   - Trainable Parameters: {total_params:,}")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(deeplog.parameters(), lr=args.lr)

    # 3. Training Loop
    print("\n[3/4] Training LSTM across epochs...")
    print("-" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Top-1 Acc':<15} | {'Status'}")
    print("-" * 75)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        deeplog.train()
        running_train_loss = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            log_probs = deeplog(batch_x)
            loss = criterion(log_probs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = running_train_loss / train_batches

        # Validation Pass
        deeplog.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                log_probs = deeplog(batch_x)
                loss = criterion(log_probs, batch_y)

                running_val_loss += loss.item()
                val_batches += 1

                preds = log_probs.argmax(dim=-1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        avg_val_loss = running_val_loss / val_batches
        val_acc = (val_correct / val_total) * 100

        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            deeplog.save(args.output_model)
            status = "💾 BEST CHECKPOINT"

        if epoch % 5 == 0 or epoch == 1 or status != "":
            print(f"{epoch:<8} | {avg_train_loss:<12.5f} | {avg_val_loss:<12.5f} | {val_acc:<14.2f}% | {status}")

    total_time = time.time() - start_train_time
    print("=" * 75)
    print(f"[4/4] Training finished in {total_time:.2f}s")
    print(f"🏆 Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.5f} | Best Val Top-1 Acc: {best_val_acc:.2f}%")
    print(f"💾 Checkpoint saved to: '{args.output_model}'")


if __name__ == "__main__":
    main()
