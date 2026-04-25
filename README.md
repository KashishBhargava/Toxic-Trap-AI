# 🛡️ Toxic Trap AI — Neural Multilingual Moderation

A dual-engine NLP suite designed to detect and categorize toxic content across English, Hindi, and code-switched text using **XLM-RoBERTa**.

### 📌 What is this project?

**Toxic Trap AI** is a production-ready moderation pipeline that:
* **Binary Check:** High-speed classification (Safe/Toxic) for instant filtering.
* **Deep Scan:** Multi-label breakdown across 6 categories (Insult, Threat, etc.).
* **Multilingual:** Handles Hindi and English natively without translation lag.
* **SaaS UI:** Enterprise-style dashboard for real-time community monitoring.

---

### 🏗️ Architecture & Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING  (Google Colab T4 GPU)         │
│                                                             │
│ Data (150K+) ──►  XLM-RoBERTa  ──►  Fine-Tuning  ──► Weights│
│  (Mixed Sets)     (Multilingual)     (3 Epochs)      (800MB)│
│                                                             │
│                          ▼                                  │
│                 Dual-Engine Inference Head                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   INFERENCE  (Gradio App)                   │
│                                                             │
│User Input  ──►  Neural Scan  ──►  Softmax/Sigmoid  ──► Result│
│(Any Text)       (Dual Model)      (Conf. Scores)      (UI)  │
└─────────────────────────────────────────────────────────────┘
```

---

### 🗂️ Training & Datasets

The system is trained on two distinct datasets to ensure both broad coverage and specific category accuracy:
1. **Jigsaw Multilingual Dataset:** 150,000+ samples for robust multilingual toxicity detection.
2. **Toxic Comment Classification Set:** Focused on multi-label traits (Severe Toxic, Obscene, Threat, Insult, Identity Hate).

| Property | Details |
| :--- | :--- |
| **Model Architecture** | **XLM-RoBERTa-base** (Transformer) |
| **Training Hardware** | **Google Colab (NVIDIA T4 GPU)** |
| **Max Token Length** | 128 (Optimized for speed) |
| **Optimizer** | AdamW with Linear Learning Rate Scheduler |

---

### 🤖 Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Inference Engine** | Python, PyTorch |
| **Models** | Hugging Face Transformers (XLM-RoBERTa) |
| **Processing** | NumPy, Pandas |
| **Hardware** | Google Colab NVIDIA T4 GPU (Training) |
| **Interface** | Gradio (Web Dashboard with Custom CSS) |
| **Version Control** | Git / GitHub |

---

### 📊 Model Performance

| Metric | Score | Status |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.98+** | ✅ High Precision |
| **Latency** | **Sub-150ms** | ⚡ Real-time |
| **Accuracy** | **96.5%** | 🎯 Verified |

---

### 📁 Project Structure

```text
Toxic-Trap-AI/
├── app.py                        # Gradio Dashboard & Dual-Engine Logic
├── requirements.txt              # Dependencies (torch, transformers, gradio)
├── .gitignore                    # Excludes large weight folders
├── Toxic_AI_Final_Weights/       # (Local) Binary Classification Weights
└── toxic_model_multilingual/     # (Local) Multi-label Breakdown Weights
```

---

### ⚡ Quick Start

1. **Clone & Setup:**
```bash
git clone https://github.com/Kashish/Toxic-Trap-AI.git
cd Toxic-Trap-AI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Weights:**
Ensure the weights folders are placed in the root directory (downloaded from external storage).

3. **Run:**
```bash
python app.py
```

---

### 🚀 Strategic Roadmap
* **API Integration:** Discord and Telegram moderation hooks.
* **Contextual Scanning:** Thread-level analysis for sarcasm detection.
* **Visual Guard:** OCR-based moderation for memes and images.

---
