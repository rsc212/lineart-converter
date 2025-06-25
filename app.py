# Line Art Conversion Web App
# Updated for better detail and compatibility with Streamlit Cloud

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
    st.image(image, caption="Original Image", use_container_width=True)

    # --- Preprocess ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- Strong Line Art with Adaptive Threshold ---
    line_art = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, blockSize=9, C=2
    )

    # --- Display Output ---
    lineart_pil = Image.fromarray(line_art)
    st.image(lineart_pil, caption="Line Drawing Output", use_container_width=True)

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
