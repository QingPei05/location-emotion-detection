import streamlit as st
from PIL import Image

st.title("Location and Emotion Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success("Analysis complete!")
