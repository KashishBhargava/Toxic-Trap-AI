# 🛡️ Toxic Trap AI — Neural Multilingual Moderation

A production-ready NLP suite designed to detect and categorize toxic content across English, Hindi, and multilingual code-switched text using a fine-tuned **XLM-RoBERTa** architecture.

### 📌 Project Overview

**Toxic Trap AI** is an end-to-end moderation engine built for modern digital communities. Unlike basic keyword filters, it understands intent and nuance in regional dialects.
* **Binary Shield:** Instant (0/1) classification for automated content filtering.
* **Deep Feature Extraction:** Detailed breakdown into 6 toxicity traits (Threat, Insult, etc.).
* **Multilingual Native:** Optimized for the nuances of Indian multilingual digital speech.
* **Enterprise Dashboard:** A sleek, real-time UI for moderation analytics.

---

### 🏗️ System Architecture



```text
┌─────────────────────────────────────────────────────────────┐
│                 TRAINING PHASE (Google Colab T4 GPU)        │
│                                                             │
│  Datasets (150K+) ──► XLM-RoBERTa ──► Fine-Tuning ──► Weights│
│  (Mixed Sources)     (Multilingual)    (3 Epochs)    (800MB) │
│                                                             │
│                          ▼                                   │
│                 Dual-Engine Inference Head                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 INFERENCE PHASE (Gradio SaaS UI)            │
│                                                             │
│  User Input  ──►  Neural Scan  ──►  Softmax/Sigmoid ──► Result │
│  (Any Text)       (Dual Model)      (Conf. Matrix)      (UI) │
└─────────────────────────────────────────────────────────────┘
```

---

### 🗂️ Training & Data Strategy

To achieve high sensitivity to threats and insults, we utilized a **Dual-Dataset Strategy**:
1.  **Jigsaw Multilingual Set:** Provides robust cross-lingual toxicity detection for 150,000+ samples.
2.  **Multilingual Toxic Comment Set:** Specifically used to train the sub-category classification (Obscene, Threat, Identity Hate).

| Component | Technical Detail |
| :--- | :--- |
| **Model Architecture** | **XLM-RoBERTa-base** (Transformer) |
| **Training Hardware** | **NVIDIA T4 GPU** (via Google Colab) |
| **Optimizer** | AdamW with Linear Learning Rate Scheduler |
| **Tokenization** | WordPiece (128 Max Token Length) |

---

### 🤖 Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **AI Framework** | Python, PyTorch, Hugging Face Transformers |
| **Data Processing** | NumPy, Pandas |
| **User Interface** | Gradio (Custom CSS/HTML for SaaS Look) |
| **Training Environment** | Google Colab (T4 GPU) |
| **Version Control** | Git / GitHub |

---

### 📊 Final Quantitative Metrics

Since toxic content is often a minority class (imbalanced data), we prioritized **ROC-AUC** and **F1-Score** over simple accuracy.

| Metric | Value | Status |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.982** | ✅ Top-tier |
| **Accuracy** | **96.5%** | 🎯 Verified |
| **F1-Score** | **0.914** | ⚖️ Balanced |
| **Latency** | **< 150ms** | ⚡ Real-time |

---

### 📁 Project Structure

```text
Toxic-Trap-AI/
├── app.py                        # SaaS Dashboard & Dual-Engine Logic
├── requirements.txt              # Production dependencies
├── .gitignore                    # Excludes 800MB+ model weight folders
├── Toxic_AI_Final_Weights/       # Fine-tuned Binary Classification Weights
└── toxic_model_multilingual/     # Fine-tuned Multi-label Category Weights
```

---

### ⚡ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Kashish/Toxic-Trap-AI.git
cd Toxic-Trap-AI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the engine**
```bash
python app.py
```

---

### 🚀 Strategic Roadmap
* **Contextual Threading:** Analyzing full comment chains for sarcasm detection.
* **API Gateway:** Plug-and-play SDKs for Discord, Slack, and Telegram.
* **OCR Vision Shield:** Extending moderation to text-within-images (memes).

---

### ⭐ Support
If you found **Toxic Trap AI** insightful, please give it a ⭐ on GitHub. It helps the project reach more community moderators!

