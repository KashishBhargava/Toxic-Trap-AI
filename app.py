import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- CONFIGURATION ---
BINARY_MODEL_PATH = "Toxic_AI_Final_Weights" 
ADV_MODEL_PATH = "toxic_model_multilingual" 
LABELS = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

# --- Load Engines ---
try:
    bin_tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_PATH)
    bin_model = AutoModelForSequenceClassification.from_pretrained(BINARY_MODEL_PATH)
    adv_tokenizer = AutoTokenizer.from_pretrained(ADV_MODEL_PATH)
    adv_model = AutoModelForSequenceClassification.from_pretrained(ADV_MODEL_PATH)
except Exception as e:
    print(f"Loading Error: {e}")

def analyze_comment(text):
    if not text or not text.strip():
        return "<div style='text-align:center; padding:20px; color:#94a3b8;'>Enter text to begin neural scan.</div>", {}

    # Core Binary Detection
    bin_inputs = bin_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        bin_outputs = bin_model(**bin_inputs)
    
    bin_probs = torch.softmax(bin_outputs.logits, dim=-1).numpy()[0]
    toxic_score = float(bin_probs[1])
    is_toxic = toxic_score > 0.5
    
    if is_toxic:
        # Detailed Categorization
        adv_inputs = adv_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            adv_outputs = adv_model(**adv_inputs)
        adv_probs = torch.sigmoid(adv_outputs.logits).numpy()[0]
        detailed_probs = {LABELS[i]: float(adv_probs[i]) for i in range(len(LABELS))}
        
        status_html = f"""
        <div class='status-card toxic'>
            <span class='status-icon'>🛑</span>
            <div>
                <h3 style='margin:0; font-size:1.4rem;'>Status: Toxic Content</h3>
                <p style='margin:5px 0 0 0;'>Probability: <b>{toxic_score:.1%}</b>. Automatic moderation recommended.</p>
            </div>
        </div>
        """
    else:
        detailed_probs = {"Neutral / Safe": 1.0}
        status_html = f"""
        <div class='status-card safe'>
            <span class='status-icon'>✅</span>
            <div>
                <h3 style='margin:0; font-size:1.4rem;'>Status: Safe Content</h3>
                <p style='margin:5px 0 0 0;'>Compliance: <b>{1-toxic_score:.1%}</b>. Content approved for community feed.</p>
            </div>
        </div>
        """
    
    return status_html, detailed_probs

# --- Professional Light Mode CSS ---
custom_css = """
body { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1200px !important; margin: auto; }

/* Hero Branding */
.hero-text { text-align: center; padding: 60px 20px; }
.hero-text h1 { font-size: 3.5rem; font-weight: 900; color: #0f172a; letter-spacing: -2px; margin-bottom: 5px; }
.hero-text p { font-size: 1.2rem; color: #64748b; max-width: 600px; margin: auto; }

/* Status Notifications */
.status-card { display: flex; align-items: center; padding: 25px; border-radius: 20px; margin-bottom: 20px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); }
.status-icon { font-size: 2.5rem; margin-right: 20px; }
.safe { background-color: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
.toxic { background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }

/* Roadmap Elements */
.roadmap-container { margin-top: 50px; padding: 40px; background: white; border-radius: 24px; border: 1px solid #e2e8f0; }
.roadmap-grid { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 20px; }
.roadmap-item { flex: 1; min-width: 250px; padding: 20px; border-radius: 16px; background: #f8fafc; border: 1px dashed #cbd5e1; }
.roadmap-item h4 { margin: 0; color: #1e293b; font-size: 1.1rem; }
.roadmap-item p { margin: 8px 0 0 0; font-size: 0.9rem; color: #64748b; line-height: 1.4; }
.badge { background: #6366f1; color: white; padding: 3px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 800; float: right; }

.analyze-btn { background: #0f172a !important; color: white !important; border: none !important; padding: 15px !important; font-size: 1.1rem !important; border-radius: 14px !important; cursor: pointer; font-weight: 700 !important; }
"""

with gr.Blocks(title="Toxic Trap AI | Neural Guard", css=custom_css) as demo:
    # 1. Branding Header
    with gr.Column(elem_classes="hero-text"):
        gr.Markdown("# Toxic Trap AI")
        gr.Markdown("Real-time Multilingual Intelligence for Content Integrity and Community Safety.")

    # 2. Workspace Row
    with gr.Row():
        with gr.Column(scale=3):
            input_msg = gr.Textbox(
                label="Neural Input Scan", 
                placeholder="Enter Hinglish, Hindi, or English text for analysis...",
                lines=7
            )
            analyze_btn = gr.Button("🚀 Execute Deep Scan", elem_classes="analyze-btn")
            gr.Examples(
                examples=["You are doing a great job!", "तुम बहुत बेकार इंसान हो", "Stop this nonsense right now."],
                inputs=input_msg
            )
            
        with gr.Column(scale=2):
            status_box = gr.HTML("<div style='text-align:center; padding:70px; color:#94a3b8; background:white; border:2px dashed #e2e8f0; border-radius:24px;'>Awaiting system input...</div>")
            label_chart = gr.Label(label="Safety Breakdown Matrix")

    # 3. Product Roadmap (Coming Soon)
    with gr.Column(elem_classes="roadmap-container"):
        gr.Markdown("### 🛠️ Product Roadmap & Scalability")
        with gr.Row(elem_classes="roadmap-grid"):
            with gr.Column(elem_classes="roadmap-item"):
                gr.HTML("<h4>API Integration <span class='badge'>COMING SOON</span></h4><p>Native SDKs for Discord, Telegram, and Slack moderation bots.</p>")
            with gr.Column(elem_classes="roadmap-item"):
                gr.HTML("<h4>Contextual Analysis <span class='badge'>COMING SOON</span></h4><p>Multi-turn conversation scanning for sarcasm and subtle gaslighting.</p>")
            with gr.Column(elem_classes="roadmap-item"):
                gr.HTML("<h4>Visual Shield <span class='badge'>COMING SOON</span></h4><p>AI-driven OCR to detect toxic text within images and memes.</p>")

    analyze_btn.click(fn=analyze_comment, inputs=input_msg, outputs=[status_box, label_chart])

if __name__ == "__main__":
    demo.launch(share=True)