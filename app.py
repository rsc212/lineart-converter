
# Line Art Conversion Web App
# Requirements: streamlit, opencv-python, numpy, pillow

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing")
st.markdown("Upload your image and instantly convert it into printable line art.")

# --- Upload ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Original Image", use_column_width=True)

    # --- Preprocess ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # --- Edge Detection ---
    edges = cv2.Canny(smooth, threshold1=30, threshold2=100)

    # --- Post-process for clean print ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    inverted = cv2.bitwise_not(cleaned)

    # --- Convert for Display ---
    lineart_pil = Image.fromarray(inverted)
    st.image(lineart_pil, caption="Line Drawing Output", use_column_width=True)

    # --- Download Button ---
    buf = io.BytesIO()
    lineart_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Line Art PNG",
        data=byte_im,
        file_name="lineart.png",
        mime="image/png"
    )
