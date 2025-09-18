# src/meme_gen_local.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

def generate_memes_local(description, n=5):
    prompt = f"Write {n} short witty meme captions (1-2 lines) for this image description:\n{description}\n\n1."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # crude parsing: return remainder after the prompt
    generated = text[len(prompt):].strip()
    # split on newline or numbering
    lines = [l.strip() for l in generated.splitlines() if l.strip()]
    return lines[:n] if lines else [generated]
