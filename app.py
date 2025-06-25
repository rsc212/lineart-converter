import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# --- App Setup ---
st.set_page_config(page_title="Photo to Coloring Page", layout="centered")
st.title("📷➡️🖍️ Photo to Coloring Page")
st.markdown("Upload a photo to generate a high-quality printable coloring page.")

# --- File Uploader ---
uploaded = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload an image to begin.")
    st.stop()

# --- Load and Display Original ---
image = Image.open(uploaded).convert("RGB")
st.subheader("Original Photo")
st.image(image, use_container_width=True)

# --- Convert to Coloring Page ---
with st.spinner("Converting to coloring page..."):
    # Convert PIL to OpenCV BGR
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # 1) Smooth while preserving edges
    blurred = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # 2) Grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 3) Adaptive threshold for crisp lines
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    # 4) Invert (so lines are black on white)
    coloring = cv2.bitwise_not(thresh)

    # 5) Clean small noise
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(coloring, cv2.MORPH_OPEN, kernel)

# Convert back to PIL and show
result = Image.fromarray(clean)
st.subheader("🖍️ Coloring Page Output")
st.image(result, use_container_width=True)

# --- Download Button ---
buf = io.BytesIO()
result.save(buf, format="PNG")
buf.seek(0)
st.download_button(
    label="📥 Download Coloring Page (PNG)",
    data=buf,
    file_name="coloring_page.png",
    mime="image/png"
)
