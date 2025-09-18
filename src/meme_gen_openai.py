# src/meme_gen_openai.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

def generate_memes_openai(description: str, n: int = 5):
    prompt = (
        f"Write {n} short, witty, family-friendly meme captions (1–2 lines) "
        f"based on this image description:\n{description}\n\nCaptions:\n1."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # lightweight + cheap, you can swap to gpt-4o
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.8,
        top_p=0.9
    )

    content = resp.choices[0].message.content.strip()

    # Parse numbered captions
    caps = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            parts = line.split(".", 1)
            if len(parts) > 1:
                caps.append(parts[1].strip())
        else:
            caps.append(line)

    return caps[:n] if caps else [content]
