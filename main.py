# main.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from watermark import WatermarkProcessor
from detector import WatermarkDetector
import attacks
from logger import log_results, plot_results


def generate(model, tokenizer, prompt, processor=None, device="cpu"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    processors = []
    if processor:
        processors.append(processor)

    output = model.generate(
        input_ids,
        max_length=120,
        do_sample=True,
        top_k=50,
        logits_processor=processors,
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate(detector, tokenizer, text):
    # FIX: tokenizer returns a plain list — detector handles normalization internally
    tokens = tokenizer(text)["input_ids"]
    return detector.detect(tokens)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    vocab_size = tokenizer.vocab_size
    prompt = "Artificial intelligence will reshape"

    wm = WatermarkProcessor(vocab_size)
    detector = WatermarkDetector(vocab_size)

    watermarked = generate(model, tokenizer, prompt, wm, device)
    print(f"\nGenerated text:\n{watermarked}\n")

    attack_map = {
        "original":  lambda x: x,
        "deletion":  attacks.deletion,
        "swap":      attacks.swap,
        "insertion": attacks.insertion,
    }

    rows = []

    for name, fn in attack_map.items():
        attacked = fn(watermarked)
        result = evaluate(detector, tokenizer, attacked)

        # FIX: use updated result keys (z_score replaces old threshold key)
        rows.append([name, result["match_ratio"], result["is_watermarked"]])

        print(
            f"{name:12s} | match_ratio={result['match_ratio']:.3f}"
            f" | z_score={result['z_score']:.3f}"
            f" | watermarked={result['is_watermarked']}"
        )

    log_results(rows)
    plot_results()


if __name__ == "__main__":
    main()