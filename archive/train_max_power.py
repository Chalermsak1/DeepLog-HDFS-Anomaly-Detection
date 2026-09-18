"""
DeepLog MAX POWER Benchmark & Full Training Pipeline
"""

import time
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor

def main():
    print("=" * 75)
    print("  🔥 DEEPLOG MAX POWER BENCHMARK & ADVANCED MODEL TRAINING")
    print("=" * 75)

    # 1. Hardware Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"⚡ Compute Acceleration Engine: {device}")

    # 2. Maximum Power Hyperparameters
    CONTEXT_LENGTH = 10     # Context sequence length
    INPUT_SIZE = 30         # Number of unique event types
    HIDDEN_SIZE = 128       # Extended LSTM Hidden Capacity (Double standard size)
    NUM_LAYERS = 3          # Deep 3-Layer Stacked LSTM Architecture
    EPOCHS = 50             # 50 Full Training Epochs
    BATCH_SIZE = 256        # Larger Batch Size for High GPU Parallelism
    MODEL_SAVE_PATH = "deeplog_model_max_power.pt"

    print("\n⚙️  Max Power Model Architecture Configuration:")
    print(f"   - Context Length (Window) : {CONTEXT_LENGTH}")
    print(f"   - Hidden Capacity (Units)  : {HIDDEN_SIZE}")
    print(f"   - Deep Stacked LSTM Layers: {NUM_LAYERS}")
    print(f"   - Total Training Epochs   : {EPOCHS}")
    print(f"   - Parallel Batch Size     : {BATCH_SIZE}")
    print(f"   - Target Model Checkpoint : {MODEL_SAVE_PATH}")

    # 3. Load Datasets
    print("\n📦 Loading Full HDFS Datasets...")
    preprocessor = Preprocessor(length=CONTEXT_LENGTH, timeout=float('inf'))
    
    start_time = time.time()
    X_train, y_train, _, _ = preprocessor.text("examples/data/hdfs_train", verbose=True)
    X_test_norm, y_test_norm, _, _ = preprocessor.text("examples/data/hdfs_test_normal", verbose=True)
    X_test_anom, y_test_anom, _, _ = preprocessor.text("examples/data/hdfs_test_abnormal", verbose=True)
    
    print(f"✅ Data Loading Complete ({time.time() - start_time:.2f}s)")
    print(f"   - Train Sequences         : {X_train.shape[0]:,}")
    print(f"   - Test Normal Sequences   : {X_test_norm.shape[0]:,}")
    print(f"   - Test Abnormal Sequences : {X_test_anom.shape[0]:,}")
    print(f"   - Total Evaluation Points : {X_test_norm.shape[0] + X_test_anom.shape[0]:,}")

    # Move training tensors to device
    X_train, y_train = X_train.to(device), y_train.to(device)

    # 4. Instantiate Deep 3-Layer DeepLog Neural Network
    deeplog = DeepLog(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=INPUT_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)

    # 5. Model Training Loop (50 Epochs)
    print(f"\n🏋️ Training Deep 3-Layer DeepLog Model for {EPOCHS} Epochs...")
    train_start = time.time()
    
    deeplog.fit(
        X=X_train,
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss()
    )
    
    train_duration = time.time() - train_start
    print(f"✅ Max Power Training Complete in {train_duration:.2f}s ({train_duration/60:.2f} mins)")

    # 6. Save Model Checkpoint
    deeplog.save(MODEL_SAVE_PATH)
    print(f"💾 Max Power Model Checkpoint saved to '{MODEL_SAVE_PATH}'")

    # 7. Batched Vectorised Inference Engine (High-speed VRAM efficient)
    @torch.no_grad()
    def fast_predict_in_batches(model, X, max_k, batch_size=32768):
        model.eval()
        preds = []
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size].to(device)
            log_probs = model(X_batch)
            _, topk_indices = log_probs.topk(max_k, dim=-1)
            preds.append(topk_indices.cpu())
            if torch.backends.mps.is_available() and (i % (batch_size * 10) == 0):
                torch.mps.empty_cache()
        return torch.cat(preds, dim=0)

    # 8. Single-Pass Multi Top-K Candidate Evaluation Grid (g = 1, 2, 3, 4, 5, 9, 15)
    print("\n" + "=" * 75)
    print("  📊 MULTI TOP-K (PARAMETER g) PERFORMANCE BENCHMARK GRID")
    print("=" * 75)

    top_k_candidates = [1, 2, 3, 4, 5, 9, 15]
    max_k = max(top_k_candidates)

    print(f"⚡ Running Single-Pass Forward Inference (Max Top-K = {max_k})...")
    infer_start = time.time()
    y_pred_norm_full = fast_predict_in_batches(deeplog, X_test_norm, max_k=max_k)
    y_pred_anom_full = fast_predict_in_batches(deeplog, X_test_anom, max_k=max_k)
    print(f"✅ Full Inference Complete in {time.time() - infer_start:.2f}s ({((X_test_norm.shape[0]+X_test_anom.shape[0])/(time.time()-infer_start)):,.0f} samples/s)")

    print(f"\n{'Top-K (g)':<10} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<10} | {'FPR (%)':<9} | {'False Alarms (FP)':<18}")
    print("-" * 75)

    for k_val in top_k_candidates:
        y_pred_norm = y_pred_norm_full[:, :k_val]
        y_pred_anom = y_pred_anom_full[:, :k_val]

        anomalies_norm = ~torch.any(y_test_norm == y_pred_norm.T, dim=0)
        anomalies_anom = ~torch.any(y_test_anom == y_pred_anom.T, dim=0)

        y_pred_total = torch.cat((anomalies_norm, anomalies_anom)).numpy()
        y_true_total = torch.cat((
            torch.zeros(anomalies_norm.shape[0], dtype=bool),
            torch.ones(anomalies_anom.shape[0], dtype=bool)
        )).numpy()

        tn, fp, fn, tp = confusion_matrix(y_true_total, y_pred_total).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"g = {k_val:<6} | {precision*100:8.2f}%   | {recall*100:8.2f}%   | {f1:8.4f}   | {fpr*100:6.3f}%   | {fp:8,d} / {tn+fp:,}")

    print("\n" + "=" * 75)
    print("  🏆 MAX POWER BENCHMARK EVALUATION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()

