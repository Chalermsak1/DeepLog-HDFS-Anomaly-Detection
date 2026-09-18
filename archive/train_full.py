"""
DeepLog Full Training & Evaluation Script
"""

import time
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor

def main():
    print("=" * 70)
    print("  🚀 DEEPLOG FULL MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Compute Device: {device}")

    # 2. Hyperparameters
    CONTEXT_LENGTH = 10     # Context sequence length
    INPUT_SIZE = 30         # Unique event types
    HIDDEN_SIZE = 64        # LSTM hidden state dimensions
    NUM_LAYERS = 2          # Stacked LSTM layers
    EPOCHS = 30             # Full training epochs
    BATCH_SIZE = 128        # Batch size
    TOP_K = 9               # Parameter g (Top-k predictions considered normal)
    MODEL_SAVE_PATH = "deeplog_model_full.pt"

    print("\n📋 Training Configuration:")
    print(f"   - Context Length : {CONTEXT_LENGTH}")
    print(f"   - Hidden Size    : {HIDDEN_SIZE}")
    print(f"   - LSTM Layers    : {NUM_LAYERS}")
    print(f"   - Total Epochs   : {EPOCHS}")
    print(f"   - Batch Size     : {BATCH_SIZE}")
    print(f"   - Top-K (g)      : {TOP_K}")
    print(f"   - Output Model   : {MODEL_SAVE_PATH}")

    # 3. Load Datasets
    print("\n📦 Loading Full HDFS Datasets...")
    preprocessor = Preprocessor(length=CONTEXT_LENGTH, timeout=float('inf'))
    
    start_time = time.time()
    X_train, y_train, _, _ = preprocessor.text("examples/data/hdfs_train", verbose=True)
    X_test_norm, y_test_norm, _, _ = preprocessor.text("examples/data/hdfs_test_normal", verbose=True)
    X_test_anom, y_test_anom, _, _ = preprocessor.text("examples/data/hdfs_test_abnormal", verbose=True)
    
    print(f"✅ Data Loading Complete ({time.time() - start_time:.2f}s)")
    print(f"   - Train Samples          : {X_train.shape[0]:,}")
    print(f"   - Test Normal Samples    : {X_test_norm.shape[0]:,}")
    print(f"   - Test Abnormal Samples  : {X_test_anom.shape[0]:,}")

    # Transfer to compute device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test_norm, y_test_norm = X_test_norm.to(device), y_test_norm.to(device)
    X_test_anom, y_test_anom = X_test_anom.to(device), y_test_anom.to(device)

    # 4. Instantiate DeepLog Model
    deeplog = DeepLog(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=INPUT_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)

    # 5. Model Training Pass
    print("\n🏋️ Starting DeepLog Model Training...")
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
    print(f"✅ Model Training Finished in {train_duration:.2f}s ({train_duration/60:.2f} mins)")

    # 6. Save Checkpoint
    deeplog.save(MODEL_SAVE_PATH)
    print(f"💾 Full Model Checkpoint saved to '{MODEL_SAVE_PATH}'")

    # Helper function to predict in batches to prevent GPU memory accumulation
    def predict_in_batches(model, X, k, batch_size=10000):
        preds = []
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            pred, _ = model.predict(X_batch, k=k, verbose=False)
            preds.append(pred.cpu())
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        return torch.cat(preds, dim=0)

    # 7. Model Evaluation
    print("\n🔍 Running Evaluation on Normal Test Dataset...")
    y_pred_norm = predict_in_batches(deeplog, X_test_norm, k=TOP_K)

    print("🔍 Running Evaluation on Abnormal Test Dataset...")
    y_pred_anom = predict_in_batches(deeplog, X_test_anom, k=TOP_K)

    # Calculate Anomaly Detection Metrics
    anomalies_norm = ~torch.any(y_test_norm.cpu() == y_pred_norm.T, dim=0)
    anomalies_anom = ~torch.any(y_test_anom.cpu() == y_pred_anom.T, dim=0)


    y_pred_total = torch.cat((anomalies_norm, anomalies_anom)).cpu().numpy()
    y_true_total = torch.cat((
        torch.zeros(anomalies_norm.shape[0], dtype=bool),
        torch.ones(anomalies_anom.shape[0], dtype=bool)
    )).cpu().numpy()

    # 8. Print Classification Metrics Report
    print("\n" + "=" * 70)
    print("  📊 ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 70)
    report = classification_report(
        y_true=y_true_total,
        y_pred=y_pred_total,
        labels=[False, True],
        target_names=["Normal Event", "Anomaly Event"],
        digits=4
    )
    print(report)
    print("=" * 70)

if __name__ == "__main__":
    main()

     