import streamlit as st
import tempfile
from src.blip_caption import generate_description
from src.meme_gen_openai import generate_memes_openai

st.set_page_config(page_title="Meme Captioning", layout="wide")
st.title("😂 Meme Captioning Generator")
st.write("Upload an image → Get a factual description → Funny meme captions!")

# Upload section
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    # Generate description with BLIP
    with st.spinner("🔎 Generating description with BLIP..."):
        desc = generate_description(tmp_path)

    # Create 2-column layout (left: image, right: text outputs)
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.image(uploaded, width="stretch", caption="Your uploaded image")

    with col2:
        st.subheader("📝 Image Description")
        st.success(desc)  # show BLIP description

        st.subheader("🚀 Meme Captions (OpenAI GPT-4o-mini)")
        if st.button("Generate Meme Captions"):
            with st.spinner("Calling OpenAI API..."):
                try:
                    api_caps = generate_memes_openai(desc, n=5)
                    for c in api_caps:
                        st.write("- ", c)
                except Exception as e:
                    st.error(f"Error calling OpenAI API: {e}")
