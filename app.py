# Streamlit App: Local Photo to Line Art Using OpenCV

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing")
st.markdown("Upload your photo and get a crisp black-and-white line drawing instantly — no API needed!")

# --- Upload ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # Invert and blend to get line effect
    image_invert = cv2.bitwise_not(image_blur)
    image_sketch = cv2.divide(image_gray, 255 - image_blur, scale=256)

    st.subheader("✏️ Line Drawing Output")
    st.image(image_sketch, caption="Line Drawing", use_container_width=True, clamp=True)

    # Save and provide download
    result = Image.fromarray(image_sketch)
    st.download_button(
        label="📥 Download Line Art",
        data=cv2.imencode('.png', image_sketch)[1].tobytes(),
        file_name="line_drawing.png",
        mime="image/png"
    )
