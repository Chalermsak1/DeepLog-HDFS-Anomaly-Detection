import time
# pyrefly: ignore [missing-import]
import torch
from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor

def main():
    print("=" * 65)
    print("  🚀 DEEPLOG REAL-TIME LOG ANOMALY DETECTION DEMO")
    print("=" * 65)

    # 1. Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🔹 Using hardware acceleration: {device}")

    # 2. Check or load model
    model_path = "deeplog_hdfs_model.pt"
    try:
        print(f"🔹 Loading trained DeepLog model from '{model_path}'...")
        deeplog = DeepLog.load(model_path, device=device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Model not found or error loading ({e}), training a new model...")
        preprocessor = Preprocessor(length=10, timeout=float('inf'))
        X_train, y_train, _, _ = preprocessor.text("examples/data/hdfs_train", nrows=1000)
        X_train, y_train = X_train.to(device), y_train.to(device)
        deeplog = DeepLog(input_size=30, hidden_size=64, output_size=30).to(device)
        deeplog.fit(X=X_train, y=y_train, epochs=5, batch_size=128)
        deeplog.save(model_path)

    # 3. Load sample log sequences
    print("\n📦 Loading sample test logs for real-time simulation...")
    preprocessor = Preprocessor(length=10, timeout=float('inf'))
    
    # Load 15 normal sequences and 15 abnormal sequences
    X_norm, y_norm, _, _ = preprocessor.text("examples/data/hdfs_test_normal", nrows=20)
    X_anom, y_anom, _, _ = preprocessor.text("examples/data/hdfs_test_abnormal", nrows=20)

    # Combine normal and abnormal sequences for demo stream
    test_stream = []
    for i in range(10):
        test_stream.append((X_norm[i], y_norm[i], "Normal Log Source"))
    for i in range(10):
        test_stream.append((X_anom[i], y_anom[i], "Abnormal Log Source"))

    print("\n" + "=" * 65)
    print("  📡 STARTING REAL-TIME LOG STREAMING MONITOR")
    print("=" * 65 + "\n")

    normal_count = 0
    anomaly_count = 0

    for idx, (seq, next_event, source) in enumerate(test_stream, 1):
        # Prepare tensor
        seq_tensor = seq.unsqueeze(0).to(device)
        actual_event = next_event.item()

        # Predict top-k expected next events
        top_k_pred, confidence = deeplog.predict(seq_tensor, k=9, verbose=False)
        expected_events = top_k_pred[0].cpu().numpy().tolist()

        # Check if actual event is in expected top-k
        is_anomaly = actual_event not in expected_events

        # Format output
        seq_str = " -> ".join(map(str, seq.numpy()[:5])) + " ..."
        exp_str = ", ".join(map(str, expected_events[:5]))

        print(f"[{idx:02d}] Source: {source}")
        print(f"     Context History (Sequence) : [{seq_str}]")
        print(f"     Expected Events (AI Top-5) : [{exp_str}]")
        print(f"     Incoming Event (Actual)    : {actual_event}")

        if is_anomaly:
            anomaly_count += 1
            print(f"     STATUS: 🚨 [ANOMALY ALERT!] Event {actual_event} deviates from expected workflow!")
        else:
            normal_count += 1
            print(f"     STATUS: ✅ [NORMAL] Event matches AI expected pattern.")

        print("-" * 65)
        time.sleep(0.5)  # Simulate real-time stream pause

    print("\n" + "=" * 65)
    print("  📊 SUMMARY MONITORING REPORT")
    print("=" * 65)
    print(f"  Total Log Sequences Processed : {len(test_stream)}")
    print(f"  ✅ Normal Events Detected      : {normal_count}")
    print(f"  🚨 Anomalies Detected (Alerts) : {anomaly_count}")
    print("=" * 65)

if __name__ == "__main__":
    main()
