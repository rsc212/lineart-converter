```python
# Streamlit App: Photo to Coloring Page & Line Drawing Converter

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# --- App Setup ---
st.set_page_config(page_title="Photo to Coloring / Line Drawing Converter", layout="centered")
st.title("📷➡️✏️🖍️ Photo to Line Drawing & Coloring Page")
st.markdown("Upload a photo to generate either a vector-style line drawing or a coloring page.")

# --- Pipeline Selection ---
mode = st.radio("Choose output style:", ("Line Drawing", "Coloring Page"))

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Please upload an image to begin.")
    st.stop()

# --- Load and Display Original ---
image = Image.open(uploaded_file).convert("RGB")
st.subheader("Original Photo")
st.image(image, use_container_width=True)

# --- Processing ---
with st.spinner(f"Converting to {mode.lower()}... "):
    # Convert PIL image to OpenCV BGR array
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Smooth while preserving edges
    smoothed = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    if mode == "Line Drawing":
        # Canny edge detection
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        # Close small gaps
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        output = clean

    else:  # Coloring Page
        # Adaptive thresholding for crisp lines
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )
        # Invert so lines are black on white
        inverted = cv2.bitwise_not(thresh)
        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        output = clean

# --- Display & Download ---
result_img = Image.fromarray(output)
st.subheader(f"Output: {mode}")
st.image(result_img, use_container_width=True)

# Prepare download
buf = io.BytesIO()
result_img.save(buf, format="PNG")
buf.seek(0)
st.download_button(
    label=f"📥 Download {mode}",
    data=buf,
    file_name=("line_drawing.png" if mode == "Line Drawing" else "coloring_page.png"),
    mime="image/png"
)
```
