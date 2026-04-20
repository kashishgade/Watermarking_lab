# app.py

import streamlit as st
import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from watermark import WatermarkProcessor
from detector import WatermarkDetector
import attacks

st.title("Welcome to the Watermarking Lab")

device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

vocab_size = tokenizer.vocab_size
wm = WatermarkProcessor(vocab_size)
detector = WatermarkDetector(vocab_size)

# 🎛️ UI
prompt = st.text_input("Enter Prompt", "AI will change the world")
attack_type = st.selectbox("Select Attack", ["None", "Delete", "Swap", "Insert"])

if st.button("Generate & Analyze"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_length=120,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        logits_processor=[wm],
    )

    original_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # FIX: apply attacks at word/text level using the shared attacks module
    attacked_text = original_text
    if attack_type == "Delete":
        attacked_text = attacks.deletion(original_text)
    elif attack_type == "Swap":
        attacked_text = attacks.swap(original_text)
    elif attack_type == "Insert":
        attacked_text = attacks.insertion(original_text)

    # FIX: pass plain token lists to detector (it handles normalization internally)
    original_tokens = tokenizer(original_text)["input_ids"]
    attacked_tokens = tokenizer(attacked_text)["input_ids"]

    result_orig = detector.detect(original_tokens)
    result_attacked = detector.detect(attacked_tokens)

    # 🆚 Side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original (Watermarked)")
        st.write(original_text)
        st.json(result_orig)

    with col2:
        st.subheader(f"After Attack: {attack_type}")
        st.write(attacked_text)
        st.json(result_attacked)

    # 📊 Graph comparison
    st.subheader("Watermark Confidence Comparison")

    fig, ax = plt.subplots()
    ax.plot(result_orig["trajectory"], label="Original")
    ax.plot(result_attacked["trajectory"], label="Attacked")
    ax.axhline(y=detector.fraction, color="gray", linestyle="--", label="Baseline (fraction)")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Match Ratio")
    ax.set_title("Watermark Robustness")
    ax.legend()

    st.pyplot(fig)