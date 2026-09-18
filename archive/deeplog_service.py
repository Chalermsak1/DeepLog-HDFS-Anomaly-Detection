"""
DeepLog Real-Time Anomaly Detection Service (Production-Ready)
"""

import os
import sys
import time
import logging
from typing import List, Tuple

# Suppress TQDM progress bars in production logs
os.environ["TQDM_DISABLE"] = "1"

import torch
from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor


# Configure production logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("deeplog_service.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("DeepLogService")


class LogAnomalyDetector:
    def __init__(self, model_path: str, top_k: int = 9):
        self.model_path = model_path
        self.top_k = top_k
        self.device = self._select_device()
        self.model = self._load_model()

    def _select_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Initialized detector on device: {device}")
        return device

    def _load_model(self) -> DeepLog:
        logger.info(f"Loading DeepLog model checkpoint from: {self.model_path}")
        try:
            model = DeepLog.load(self.model_path, device=self.device)
            logger.info("Model checkpoint loaded successfully")
            return model
        except Exception as err:
            logger.error(f"Failed to load model from {self.model_path}: {err}")
            raise err

    def evaluate_sequence(self, sequence: List[int], actual_event: int) -> Tuple[bool, List[int]]:
        """
        Evaluate a sequence of historical log events against an incoming event.
        Returns (is_anomaly, expected_top_k_events).
        """
        seq_tensor = torch.tensor([sequence], dtype=torch.long, device=self.device)
        
        # Predict top-k expected next events without progress bar
        top_k_pred, _ = self.model.predict(seq_tensor, k=self.top_k, verbose=False)
        expected_events = top_k_pred[0].cpu().numpy().tolist()

        is_anomaly = actual_event not in expected_events
        return is_anomaly, expected_events


def main():
    model_file = "deeplog_hdfs_model.pt"
    detector = LogAnomalyDetector(model_path=model_file, top_k=9)

    logger.info("Loading test log data stream...")
    preprocessor = Preprocessor(length=10, timeout=float('inf'))
    X_norm, y_norm, _, _ = preprocessor.text("examples/data/hdfs_test_normal", nrows=10)
    X_anom, y_anom, _, _ = preprocessor.text("examples/data/hdfs_test_abnormal", nrows=10)

    # Prepare stream items
    stream_items = []
    for i in range(len(X_norm)):
        stream_items.append((X_norm[i].numpy().tolist(), y_norm[i].item(), "NORMAL_STREAM"))
    for i in range(len(X_anom)):
        stream_items.append((X_anom[i].numpy().tolist(), y_anom[i].item(), "ABNORMAL_STREAM"))

    logger.info(f"Starting stream evaluation ({len(stream_items)} log sequences)...")

    processed_count = 0
    anomaly_count = 0

    try:
        for idx, (sequence, actual_event, stream_source) in enumerate(stream_items, start=1):
            is_anomaly, expected_events = detector.evaluate_sequence(sequence, actual_event)
            processed_count += 1

            if is_anomaly:
                anomaly_count += 1
                logger.warning(
                    f"ANOMALY DETECTED | Source: {stream_source} | "
                    f"Actual Event: {actual_event} | Expected (Top-5): {expected_events[:5]} | "
                    f"Sequence Context: {sequence[-5:]}"
                )
            else:
                logger.info(
                    f"EVENT OK | Source: {stream_source} | "
                    f"Actual Event: {actual_event} | Expected (Top-5): {expected_events[:5]}"
                )

            time.sleep(0.2)

    except KeyboardInterrupt:
        logger.info("Stream processing interrupted by user")

    logger.info(f"Processing Complete | Total: {processed_count} | Anomalies: {anomaly_count}")


if __name__ == "__main__":
    main()


