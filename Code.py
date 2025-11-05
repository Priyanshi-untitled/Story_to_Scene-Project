import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Scene to Story", layout="centered")
st.title("📸 Scene to Story Generator")
st.write("Upload an image to see its caption and a generated story based on it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        caption_output = captioner(image)
        caption_text = caption_output[0]["generated_text"]
    st.subheader("🖋️ Caption:")
    st.write(caption_text)

    with st.spinner("Turning caption into a story..."):
        story_gen = pipeline("text-generation", model="gpt2")
        story_output = story_gen(
            caption_text,
            max_length=120,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8
        )
        story_text = story_output[0]["generated_text"]

    st.subheader("📖 Generated Story:")
    st.write(story_text)


     
