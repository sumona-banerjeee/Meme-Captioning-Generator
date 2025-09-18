# src/blip_caption.py
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use the small/medium BLIP model for fast inference on a laptop
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_description(image_path, max_length=40):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    # beam search for stable captions
    out = model.generate(**inputs, max_new_tokens=max_length, num_beams=5)
    # processor.decode works for BLIP outputs
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
