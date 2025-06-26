import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
import io

st.set_page_config(page_title="Photo to Vector-Style Coloring Page", layout="centered")
st.title("📷➡️🖍️ Photo to Line Art & Vector Coloring Page")
st.markdown("Upload a photo, convert it to clean line art (OpenCV pipeline), and export as SVG (vector).")

# --------------------
# YOUR CUSTOM PIPELINE
# --------------------
def custom_lineart_pipeline(img):
    # 1. Gaussian Blur (intensity: 50 → kernel=51)
    blur1 = cv2.GaussianBlur(img, (51, 51), 0)

    # 2. Grayscale
    gray = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)

    # 3. Adaptive Threshold (block size: 55, c: 14, method: gaussian, blend: 100)
    adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        55, 14
    )

    # 4. Median Blur (kernelSize: 5)
    median1 = cv2.medianBlur(adapt, 5)

    # 5. "Sketch #1" (Canny edge)
    edges = cv2.Canny(median1, 50, 150)

    # 6. Gaussian Blur (intensity: 22 → kernel=23)
    blur2 = cv2.GaussianBlur(edges, (23, 23), 0)

    # 7. Median Blur (kernelSize: 5)
    median2 = cv2.medianBlur(blur2, 5)

    # Invert for coloring page look
    out = cv2.bitwise_not(median2)
    return out

# ----------------
# FILE UPLOAD STEP
# ----------------
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Please upload an image to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.subheader("Original Photo")
st.image(image, use_container_width=True)

# ----------------------
# PIPELINE APPLICATION
# ----------------------
with st.spinner("Processing image for clean line art..."):
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    output = custom_lineart_pipeline(bgr)

lineart_img = Image.fromarray(output)
st.subheader("🖍️ Line Art (Raster, OpenCV pipeline)")
st.image(lineart_img, use_container_width=True)

# ---- PNG Download ----
buf = io.BytesIO()
lineart_img.save(buf, format="PNG")
buf.seek(0)
st.download_button("Download PNG", data=buf, file_name="lineart.png", mime="image/png")

# ---- SVG Vectorization ----
st.subheader("🪄 Convert to SVG (Vector) via Hugging Face API")
if st.button("Vectorize (Export as SVG)"):
    st.info("Sending to Hugging Face openfree/image-to-vector API...")
    buf.seek(0)
    response = requests.post(
        "https://hf.space/embed/openfree/image-to-vector/api/predict/",
        files={"image": ("lineart.png", buf, "image/png")},
        timeout=120
    )
    if response.status_code == 200:
        try:
            svg_url = response.json()["data"][0]
            svg_response = requests.get(svg_url)
            if svg_response.status_code == 200:
                st.success("SVG ready! Download below.")
                st.download_button("Download SVG", data=svg_response.content,
                                   file_name="vector_coloring_page.svg", mime="image/svg+xml")
                st.markdown(f"[Open SVG in new tab]({svg_url})")
            else:
                st.error("Failed to download SVG from Hugging Face.")
        except Exception as e:
            st.error(f"Error parsing Hugging Face API response: {e}")
    else:
        st.error(f"Error with Hugging Face API request. Status: {response.status_code}")

