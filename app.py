import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
import io

st.set_page_config(page_title="Photo to Vector-Style Coloring Page", layout="centered")
st.title("📷➡️🖍️ Photo to Line Art & Vector Coloring Page")
st.markdown("Upload a photo, convert it to clean line art, and export as SVG (vector).")

uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Please upload an image to begin.")
    st.stop()

# Show original
image = Image.open(uploaded_file).convert("RGB")
st.subheader("Original Photo")
st.image(image, use_container_width=True)

# Processing: grayscale, blur, adaptive threshold, median, etc.
with st.spinner("Processing image for clean line art..."):
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adapt = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    median = cv2.medianBlur(adapt, 3)
    sketch = cv2.Canny(median, 50, 150)
    blur2 = cv2.GaussianBlur(sketch, (3, 3), 0)
    median2 = cv2.medianBlur(blur2, 3)
    output = cv2.bitwise_not(median2)

lineart_img = Image.fromarray(output)
st.subheader("🖍️ Line Art (Raster)")
st.image(lineart_img, use_container_width=True)

# Save PNG to buffer
buf = io.BytesIO()
lineart_img.save(buf, format="PNG")
buf.seek(0)

st.download_button("Download PNG", data=buf, file_name="lineart.png", mime="image/png")

# --- Vectorization step via Hugging Face API ---
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
        svg_url = response.json()["data"][0]
        svg_response = requests.get(svg_url)
        if svg_response.status_code == 200:
            st.success("SVG ready! Download below.")
            st.download_button("Download SVG", data=svg_response.content,
                               file_name="vector_coloring_page.svg", mime="image/svg+xml")
            st.markdown(f"[Open SVG in new tab]({svg_url})")
        else:
            st.error("Failed to download SVG from Hugging Face.")
    else:
        st.error("Error with Hugging Face API request.")
