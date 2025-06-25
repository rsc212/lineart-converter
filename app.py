import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# --- App Configuration ---
st.set_page_config(page_title="Photo to Vector-Style Line Art", layout="centered")
st.title("📷➡️🖍️ Photo to Vector-Style Line Art & Coloring Page")
st.markdown("Upload a photo and instantly convert it into a crisp line drawing or coloring-page vector style.")

# --- Select Mode ---
mode = st.radio("Choose output style:", ["Line Drawing", "Coloring Page"])

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg","jpeg","png"])
if not uploaded_file:
    st.info("Please upload an image to get started.")
    st.stop()

# --- Load Image ---
image = Image.open(uploaded_file).convert("RGB")
st.subheader("Original Photo")
st.image(image, use_container_width=True)

# --- Convert to OpenCV Format ---
arr = np.array(image)
bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# --- Processing Spinner ---
with st.spinner(f"Converting to {mode.lower()}... this may take a few seconds"):
    # Smooth and grayscale
    smoothed = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    # Kernel for morphology
    kernel = np.ones((2,2), np.uint8)

    if mode == "Line Drawing":
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        # Close small gaps
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # Invert: lines black on white
        processed = cv2.bitwise_not(closed)

    else:
        # Adaptive threshold (binary inverse)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=9, C=2
        )
        # Remove small artifacts
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Invert: lines black on white
        processed = cv2.bitwise_not(clean)

# --- Display Result ---
result = Image.fromarray(processed)
st.subheader(f"Output: {mode}")
st.image(result, use_container_width=True)

# --- Download Button ---
buffer = io.BytesIO()
result.save(buffer, format="PNG")
buffer.seek(0)
st.download_button(
    label=f"Download {mode}",
    data=buffer,
    file_name=("line_drawing.png" if mode=="Line Drawing" else "coloring_page.png"),
    mime="image/png"
)
