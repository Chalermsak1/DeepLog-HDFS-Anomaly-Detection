"""
DeepLog HDFS Log Anomaly Detection - Streamlit Portfolio Dashboard
==================================================================
Interactive demonstration of DeepLog sequential log anomaly detection.
Features:
- Dataset & System Statistics
- Official Baseline Evaluation Metrics
- Confusion Matrix & Performance Visualizations
- Interactive Sequence Anomaly Diagnoser
"""

import os
import sys
import torch
import streamlit as st

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from deeplog import DeepLog
from deeplog.preprocessor import Preprocessor

# Set page configuration
st.set_page_config(
    page_title="DeepLog HDFS Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_mapping(model_path="deeplog_model_improved_v1.pt", train_path="examples/data/hdfs_train"):
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    preprocessor = Preprocessor(length=10, timeout=float("inf"))
    _, _, _, mapping = preprocessor.text(train_path, verbose=False)

    if os.path.exists(model_path):
        model = DeepLog.load(model_path, device=device).to(device)
        model.eval()
    else:
        model = None

    return model, mapping, preprocessor, device


model, mapping, preprocessor, device = load_model_and_mapping()

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("🛡️ DeepLog HDFS")
st.sidebar.markdown("**Log Anomaly Detection**")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Configuration")
st.sidebar.markdown("""
- **Model**: Stacked LSTM
- **Layers**: 2 Stacked LSTM
- **Hidden Units**: 128
- **Context Length**: 10
- **Vocab Size**: 30 classes
- **Training Epochs**: 40
- **Optimizer**: Adam (lr=0.001)
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Dataset Summary")
st.sidebar.markdown("""
- **Total Blocks**: 575,061
- **Normal**: 558,223 (97.07%)
- **Anomaly**: 16,838 (2.93%)
- **Events**: 29 templates (E1-E29)
""")

# ==============================================================================
# MAIN PAGE CONTENT
# ==============================================================================
st.title("DeepLog: HDFS Log Anomaly Detection System")
st.markdown("""
DeepLog models execution workflows in unstructured HDFS system logs as sequential language tokens using a 
deep Long Short-Term Memory (LSTM) network. Anomalies are flagged whenever unexpected log transitions deviate from 
statistically normal operational patterns.
""")

# High-Level Metrics
st.markdown("### 🏆 Official Baseline Performance (Top-5 Threshold)")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""<div class="metric-card"><div class="metric-val" style="color:#2563eb;">98.53%</div><div class="metric-label">Accuracy</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="metric-card"><div class="metric-val" style="color:#059669;">85.73%</div><div class="metric-label">Precision</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="metric-card"><div class="metric-val" style="color:#d97706;">60.07%</div><div class="metric-label">Recall</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="metric-card"><div class="metric-val" style="color:#7c3aed;">70.65%</div><div class="metric-label">F1-Score</div></div>""", unsafe_allow_html=True)
with col5:
    st.markdown("""<div class="metric-card"><div class="metric-val" style="color:#dc2626;">0.304%</div><div class="metric-label">False Positive Rate</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# Visualizations & Architecture Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Confusion Matrix", 
    "📈 Next-Event Top-K", 
    "🔄 Transitions & Frequency", 
    "🧱 Class Imbalance", 
    "📐 Architecture Flow"
])

with tab1:
    col_cm1, col_cm2 = st.columns([1.2, 1])
    with col_cm1:
        cm_file = "results/figures/confusion_matrix.png"
        if os.path.exists(cm_file):
            st.image(cm_file, caption="Block-Level Confusion Matrix (g = 5)", width="stretch")
    with col_cm2:
        st.markdown("#### Confusion Matrix Breakdown")
        st.markdown("""
        | Metric | Count | Description |
        |---|---:|---|
        | **True Positives (TP)** | **10,115** | Anomaly blocks correctly identified |
        | **False Positives (FP)** | **1,683** | Normal blocks incorrectly flagged |
        | **True Negatives (TN)** | **551,683** | Normal blocks correctly accepted |
        | **False Negatives (FN)** | **6,723** | Anomaly blocks evading Top-5 detection |
        | **Total Evaluation** | **570,204** | Total test set blocks evaluated |
        """)
        st.info("💡 **Low False Alarm Rate**: At g=5, FPR is constrained to **0.304%**, preventing operator alert fatigue.")

with tab2:
    col_top1, col_top2 = st.columns([1.2, 1])
    with col_top1:
        topk_file = "results/figures/topk_accuracy.png"
        if os.path.exists(topk_file):
            st.image(topk_file, caption="Next-Event Prediction Accuracy Across Top-K Candidates", width="stretch")
    with col_top2:
        st.markdown("#### Next-Event Accuracy Table")
        st.markdown("""
        | Threshold (g) | Next-Event Accuracy |
        |---|---:|
        | **Top-1** | **90.05%** |
        | **Top-3** | **99.15%** |
        | **Top-5** | **99.79%** |
        | **Top-9** | **99.88%** |
        | **Top-15** | **99.91%** |
        """)
        st.success("✅ **Consistent Mapping Fix**: Reusing the canonical training mapping improved Top-1 next-event accuracy to **90.05%**.")

with tab3:
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        freq_file = "results/figures/event_frequency.png"
        if os.path.exists(freq_file):
            st.image(freq_file, caption="Total Event Frequency Distribution", width="stretch")
    with col_tr2:
        trans_file = "results/figures/event_transitions.png"
        if os.path.exists(trans_file):
            st.image(trans_file, caption="Sequential Event Bigram Transitions", width="stretch")

with tab4:
    col_imb1, col_imb2 = st.columns([1.2, 1])
    with col_imb1:
        dist_file = "results/figures/block_distribution.png"
        if os.path.exists(dist_file):
            st.image(dist_file, caption="HDFS Class Distribution", width="stretch")
    with col_imb2:
        st.markdown("#### Class Imbalance Context")
        st.markdown("""
        - Normal execution blocks dominate the dataset (**97.07%**, 558,223 blocks).
        - Anomalies represent a rare class (**2.93%**, 16,838 blocks).
        - Standard accuracy is insufficient due to base-rate fallacy; Precision (85.73%) and Recall (60.07%) provide realistic operational insight.
        """)

with tab5:
    wf_file = "docs/workflow.png"
    if os.path.exists(wf_file):
        st.image(wf_file, caption="DeepLog End-to-End Operational Flowchart", width="stretch")

st.markdown("---")

# ==============================================================================
# INTERACTIVE SEQUENCE DIAGNOSIS
# ==============================================================================
st.markdown("### 🧪 Live Log Sequence Anomaly Diagnoser")
st.markdown("Enter or select a sequence of HDFS log event IDs to evaluate real-time sequence prediction.")

sample_options = {
    "Normal Block Example (Standard File Write)": "5 5 5 22 11 9 11 9 11 9 26 26 26 23 23 23 21 21 21",
    "Normal Block Example (Short Replicated Block)": "22 5 5 5 11 9 11 9 11 9 26 26 26",
    "Anomaly Block Example (Write Exception E7)": "5 5 5 22 11 9 7 9 26 26 23 23 21",
    "Custom Sequence": "",
}

selected_sample = st.selectbox("Choose a sample sequence or enter custom:", list(sample_options.keys()))

default_seq = sample_options[selected_sample] if selected_sample != "Custom Sequence" else "5 5 5 22 11 9 11 9 11 9 26 26"
input_sequence = st.text_area("HDFS Event Sequence (space-separated event numbers or IDs):", value=default_seq, height=80)
top_k_select = st.slider("Top-K Candidates Threshold (g):", min_value=1, max_value=15, value=5)

if st.button("🔍 Run Anomaly Diagnosis", type="primary"):
    if not model:
        st.error("Model checkpoint not found. Ensure `deeplog_model_improved_v1.pt` is present.")
    else:
        tokens = input_sequence.strip().split()
        events = []
        for t in tokens:
            clean = t.strip().upper().replace(",", "").replace("[", "").replace("]", "")
            if clean.startswith("E"):
                events.append(int(clean[1:]))
            elif clean.isdigit():
                events.append(int(clean))

        if len(events) == 0:
            st.warning("Please enter a valid sequence of event numbers.")
        else:
            id_to_canonical = {v: k for k, v in mapping.items()}
            no_event_canonical = id_to_canonical.get(-1337, len(mapping) - 1)

            mapped_events = [id_to_canonical.get(e, -1) for e in events]
            block_anomaly = False
            results_table = []

            for i in range(len(mapped_events)):
                actual_mapped = mapped_events[i]
                actual_raw = events[i]

                if i < 10:
                    raw_ctx = [no_event_canonical] * (10 - i) + mapped_events[:i]
                else:
                    raw_ctx = mapped_events[i - 10:i]

                ctx = [c if c >= 0 else no_event_canonical for c in raw_ctx]
                ctx_tensor = torch.tensor([ctx], dtype=torch.long, device=device)

                with torch.no_grad():
                    log_probs = model(ctx_tensor)
                    _, topk_indices = log_probs.topk(top_k_select, dim=-1)

                topk_list = topk_indices[0].cpu().tolist()
                expected_events = [f"E{mapping.get(idx, idx)}" for idx in topk_list]

                is_anom = (actual_mapped not in topk_list) or (actual_mapped == -1)
                if is_anom:
                    block_anomaly = True

                results_table.append({
                    "Step": i + 1,
                    "Actual Event": f"E{actual_raw}",
                    "Top-K Candidates": ", ".join(expected_events),
                    "Status": "🚨 Anomaly" if is_anom else "✅ Normal"
                })

            if block_anomaly:
                st.error("### 🚨 VERDICT: ANOMALOUS BLOCK DETECTED")
                st.markdown("At least one event in this block deviated from the model's Top-K next-event expectations.")
            else:
                st.success("### ✅ VERDICT: NORMAL EXECUTION BLOCK")
                st.markdown("All sequential events matched the learned normal operational workflows.")

            st.dataframe(results_table, width="stretch")
