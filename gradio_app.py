# gradio_app.py

import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from watermark import WatermarkProcessor
from detector import WatermarkDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

vocab_size = tokenizer.vocab_size
wm = WatermarkProcessor(vocab_size)
detector = WatermarkDetector(vocab_size)


def run(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids,
        max_length=120,
        do_sample=True,
        logits_processor=[wm],
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # FIX: tokenizer returns a plain list — detector now handles this correctly
    tokens = tokenizer(text)["input_ids"]
    result = detector.detect(tokens)

    # Remove trajectory from JSON display (too long for Gradio output)
    display_result = {k: v for k, v in result.items() if k != "trajectory"}

    return text, display_result


gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Prompt"),
    outputs=[
        gr.Textbox(label="Generated Text"),
        gr.JSON(label="Detection Result"),
    ],
    title="Watermark Detection Demo",
).launch()