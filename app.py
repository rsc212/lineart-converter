# Line Art Conversion Web App – Final version with Laplacian edge detection

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

    # --- Convert to Grayscale ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Enhance Contrast ---
    norm = cv2.equalizeHist(gray)

    # --- Apply Gaussian Blur ---
    blurred = cv2.GaussianBlur(norm, (3, 3), 0)

    # --- Laplacian Edge Detection ---
    laplacian = cv2.Laplacian(blurred, cv2.CV_8U, ksize=5)
    inverted = cv2.bitwise_not(laplacian)

    # --- Threshold to Binary for Clean Line Look ---
    _, binary = cv2.threshold(inverted, 150, 255, cv2.THRESH_BINARY)

    # --- Display Output ---
    lineart_pil = Image.fromarray(binary)
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
