"""
DeepLog Improved V1 Training Pipeline
- Consistent Event ID mapping derived strictly from training set (no test data leakage)
- Clean Train/Val split (90:10) with fixed random seed
- Best model selection on Validation Loss
- Saves checkpoint to 'deeplog_model_improved_v1.pt'
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor

def main():
    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"⚡ Compute Device: {device}")

    # 2. Hyperparameters & Configuration
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    
    CONTEXT_LENGTH = 10
    INPUT_SIZE = 30
    OUTPUT_SIZE = 30
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    EPOCHS = 40
    BATCH_SIZE = 256
    LR = 0.001
    VAL_SPLIT = 0.10
    MODEL_OUTPUT = "deeplog_model_improved_v1.pt"

    print("=" * 70)
    print("  🚀 TRAINING DEEPLOG IMPROVED V1")
    print("=" * 70)
    print(f"Architecture    : {NUM_LAYERS}-Layer LSTM (Hidden: {HIDDEN_SIZE})")
    print(f"Context Length  : {CONTEXT_LENGTH}")
    print(f"Total Epochs    : {EPOCHS}")
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Learning Rate   : {LR}")
    print(f"Random Seed     : {RANDOM_SEED}")
    print(f"Output Checkpoint: {MODEL_OUTPUT}")

    # 3. Preprocess Training Data & Extract Canonical Vocabulary Mapping
    preprocessor = Preprocessor(length=CONTEXT_LENGTH, timeout=float('inf'))
    X_train_raw, y_train_raw, _, train_mapping = preprocessor.text("examples/data/hdfs_train", verbose=False)
    
    total_train = X_train_raw.shape[0]
    val_size = int(total_train * VAL_SPLIT)
    train_size = total_train - val_size

    # Deterministic train/val split
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(total_train, generator=generator)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    X_train, y_train = X_train_raw[train_idx], y_train_raw[train_idx]
    X_val, y_val = X_train_raw[val_idx], y_train_raw[val_idx]

    print(f"\n📦 Dataset Split:")
    print(f"   - Training Sequences   : {train_size:,}")
    print(f"   - Validation Sequences : {val_size:,}")
    print(f"   - Canonical Train Event Map: {train_mapping}")

    # PyTorch DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # 4. Instantiate Model
    deeplog = DeepLog(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)

    total_params = sum(p.numel() for p in deeplog.parameters() if p.requires_grad)
    print(f"\n🧠 Model Parameters: {total_params:,} trainable parameters")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(deeplog.parameters(), lr=LR)

    # 5. Training Loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "=" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Top-1 Acc':<15} | {'Status'}")
    print("-" * 75)

    start_train_time = time.time()
    for epoch in range(1, EPOCHS + 1):
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
            deeplog.save(MODEL_OUTPUT)
            status = "💾 BEST CHECKPOINT"

        if epoch % 5 == 0 or epoch == 1 or status != "":
            print(f"{epoch:<8} | {avg_train_loss:<12.5f} | {avg_val_loss:<12.5f} | {val_acc:<14.2f}% | {status}")

    total_train_sec = time.time() - start_train_time
    print("=" * 75)
    print(f"✅ Training Finished in {total_train_sec:.2f}s")
    print(f"🏆 Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.5f} | Best Val Top-1 Acc: {best_val_acc:.2f}%")
    print(f"💾 Checkpoint Saved: '{MODEL_OUTPUT}'")

if __name__ == "__main__":
    main()
