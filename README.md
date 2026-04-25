# 🛡️ Toxic Trap AI — Neural Multilingual Moderation

A production-ready NLP application that detects toxic content in English, Hindi, and Hinglish using a dual-engine transformer pipeline — powered by **XLM-RoBERTa**.

### 📌 What is this project?

**Toxic Trap AI** is an end-to-end moderation suite that:
* **Identifies** harmful content with a strict Binary Check (0/1) for compliance.
* **Analyzes** toxic intent across 6 distinct categories (Threats, Insults, etc.).
* **Supports** Multilingual inputs, specialized for Indian code-switched (Hinglish) text.
* **Visualizes** confidence scores through a professional Enterprise SaaS dashboard.

Whether you're moderating a community forum or building a safe social space, Toxic Trap provides the neural intelligence to "trap" toxicity before it spreads.

---

### 🏗️ Architecture & Pipeline

```text
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING  (Colab T4 GPU)                │
│                                                             │
│  Dataset  ──►  XLM-RoBERTa  ──►  Fine-Tuning  ──►  Weights  │
│  (150K+)      (Multilingual)     (3 Epochs)      (800MB+)   │
│                                                             │
│                          ▼                                   │
│                 Binary & Multi-label                         │
│                 Classification Heads                         │
└──────────────────────────┬──────────────────────────────────┘
                           │  Inference logic
┌──────────────────────────▼──────────────────────────────────┐
│                   DEPLOYMENT (Gradio UI)                    │
│                                                             │
│  User Input  ──►  Neural Scan  ──►  Softmax  ──►  Status    │
│  (Hinglish)       (Dual Engine)     (Probs)      SAFE/TOXIC │
└─────────────────────────────────────────────────────────────┘
```

---

### 🗂️ Dataset & Model Info

**Source:** Jigsaw Multilingual & Custom Synthetic Indian Slang Data.

| Property | Details |
| :--- | :--- |
| **Model** | **XLM-RoBERTa-base** (Fine-tuned) |
| **Parameters** | 270M+ |
| **Languages** | English, Hindi, Hinglish (Transliterated) |
| **Task** | Binary + Multi-Label Classification |
| **Class Balance** | Weighted Loss used for minority toxic classes |

---

### 🤖 Technology Stack

| Layer | Technology | Why we chose it |
| :--- | :--- | :--- |
| **Model** | XLM-RoBERTa | Superior cross-lingual performance over DistilBERT. |
| **Framework** | PyTorch / HuggingFace | Research-standard for transformer fine-tuning. |
| **Optimization** | AdamW + LR Scheduler | Ensures stable convergence on imbalanced data. |
| **UI** | Gradio | Modern, interactive, and easy to deploy for demos. |

---

### 📊 Model Performance

Trained for **3 epochs** on NVIDIA T4 GPU:

| Metric | Score | Status |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.982+** | ✅ Excellent |
| **Inference Time** | **< 120ms** | ⚡ Real-time |
| **Language Support** | **100+ Languages** | 🌍 Global |

---

### 📁 Project Structure

```text
Toxic-Trap-AI/
│
├── app.py                        # Professional Gradio UI & Logic
├── requirements.txt              # Production dependencies
├── .gitignore                    # Excludes 800MB+ model weights
│
├── Toxic_AI_Final_Weights/       # ⚠️ BINARY MODEL (Download separately)
│   ├── config.json               # Model config
│   └── pytorch_model.bin         # Weights
│
└── toxic_model_multilingual/     # ⚠️ ADVANCED MODEL (Download separately)
    └── pytorch_model.bin         # Weights
```

---

### ⚡ Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/Kashish/Toxic-Trap-AI.git
cd Toxic-Trap-AI
```

**2. Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Dashboard**
```bash
python app.py
```

---

### 🛠️ Strategic Roadmap
* [ ] **Contextual Thread Analysis:** Scanning entire conversations for sarcasm.
* [ ] **API Gateway:** Native SDKs for Discord and Telegram bots.
* [ ] **OCR Shield:** Computer Vision to detect toxic text in memes/images.

### ⭐ Support
If you found this project helpful for your moderation needs, consider giving it a ⭐ on GitHub — it helps the project grow!

