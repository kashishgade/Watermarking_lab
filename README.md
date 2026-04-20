## 🔏 Watermarking Lab

A hands on watermarking lab for LLM text watermarking — embed invisible statistical signatures into GPT-2 generated text, test robustness under attacks, and visualize detection confidence in real time.

What Is This?
This project implements the green list watermarking scheme for large language models. During text generation, tokens are probabilistically biased toward a secret "allowed" set derived from a hash of the preceding context. A detector can then verify whether a piece of text was generated with the watermark without access to the original prompt or model weights.

## 🧠 Architecture

Input
  ↓
Embedder (watermark.py)
  ↓
LLM (GPT-2)
  ↓
Watermarked Output
  ↓
Detector (detector.py)
  ↓
Watermark Decision

## 🧩 Components

1. Input

Any text prompt provided to the model.

2. Embedder (watermark.py)

Uses a shared secret key and the current token context.

Hashes the previous tokens to deterministically select a “green list” (≈50% of vocabulary).

Boosts logits of green-list tokens (e.g., +1.5 bias).

Result: model subtly prefers certain tokens without degrading output quality.

3. LLM

Generates text as usual, but with biased token probabilities.

Produces watermarked text implicitly.

4. Detector (detector.py)

Reconstructs the same green list using the shared secret.

Compares actual tokens against expected green-list tokens.

Computes a z-score against the null hypothesis (random token selection).

5. Decision

If z-score > threshold → Likely Watermarked

Otherwise → Not Watermarked

## 🔐 Key Idea

The watermark is statistical, not visible.
No need for access to: 
Original prompt
Model weights

Only the shared secret key is required for detection.

## 📊 Why It Works

Random text → 50% green tokens

Watermarked text → significantly higher proportion

This deviation enables reliable detection via statistical testing.

📁 Project Structure
Watermarking-Lab/
├── watermark.py      # Embedder: biases token probabilities (green list selection)
├── detector.py       # Detector: reconstructs green list + computes z-score
├── attacks.py        # Adversarial text attacks (deletion, swap, insertion)
├── logger.py         # Logs results (CSV) + plots robustness metrics
├── main.py           # CLI pipeline: generate → attack → detect → log
├── app.py            # Streamlit UI (interactive demo + visualization)
├── gradio_app.py     # Lightweight Gradio demo interface
├── requirements.txt
└── README.md

## How It Works

Embedding (watermark.py)
At each generation step, the previous token sequence is hashed (SHA-256 + secret key) to deterministically select a "green list"— a random subset of 50% of the vocabulary. The logits for green-list tokens are boosted by +1.5, making the model prefer them without changing the visible quality of the output.

Detection (detector.py)

Given any token sequence, the detector reconstructs the green list for each position and checks whether the actual token falls in it. A z-score is computed against the null hypothesis (random token selection, 50% baseline). A z-score > 2.326 corresponds to p < 0.01 the text is flagged as watermarked.

Attacks (attacks.py)

Three word-level perturbations test robustness:

Deletion — randomly drops ~20% of words
Swap — randomly swaps 3 word pairs
Insertion — inserts 3 random duplicate words

# Local Python

Install dependencies:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers streamlit gradio matplotlib scipy numpy

- Run the Streamlit UI:

- streamlit run app.py

- Run the CLI pipeline:
python main.py

- Run the Gradio demo:
python gradio_app.py

## Model

This project uses GPT-2 by OpenAI, released under the MIT License and freely available via Hugging Face. No API key or account is required — weights download automatically on first run (or are pre-baked in the Docker image).

## Requirements

Python 3.8+
PyTorch (CPU build is sufficient)
Hugging Face Transformers
Streamlit (for app.py)
Gradio (for gradio_app.py)
scipy, numpy, matplotlib

## References

Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023 — arXiv:2301.10226