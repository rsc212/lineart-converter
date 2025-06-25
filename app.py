# Streamlit App: Photo to Line Art and Vectorization Using Replicate and Local Processing

import streamlit as st
import requests
from PIL import Image
import io
import time
import numpy as np
from skimage import color, filters, morphology, util

# --- App Configuration ---
st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing & Vectorization")
st.markdown("Upload your image to generate clean line art and optional vectorized skeleton output.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Please upload an image to begin.")
    st.stop()

# --- Display Original ---
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Original Image", use_container_width=True)

# --- Prepare In-Memory PNG ---
buffered = io.BytesIO()
image.save(buffered, format="PNG")
buffered.seek(0)

# --- Upload to ImgBB ---
st.info("Uploading image to temporary host...")
imgbb_key = st.secrets.get("IMGBB_API_KEY")
if not imgbb_key:
    st.error("ImgBB API key missing. Add IMGBB_API_KEY to your Streamlit secrets.")
    st.stop()

resp = requests.post(
    "https://api.imgbb.com/1/upload",
    params={"key": imgbb_key},
    files={"image": buffered}
)
if resp.status_code != 200:
    st.error("Image hosting failed. Check your ImgBB key.")
    st.stop()

image_url = resp.json()["data"]["url"]
st.success(f"✅ Image URL: {image_url}")

# --- Call Replicate for Deep Learning Line Art ---
replicate_token = st.secrets.get("REPLICATE_API_TOKEN")
if not replicate_token:
    st.error("Replicate API token missing. Add REPLICATE_API_TOKEN to your Streamlit secrets.")
    st.stop()

headers = {
    "Authorization": f"Token {replicate_token}",
    "Content-Type": "application/json",
}
version_id = "fcb14999f4b54db9a7e7ff7fb5d5b6ec5a4b30392e0d342ec23dbff712d702b3"  # Verified working version

with st.spinner("Generating deep-learning line art (10–20s)..."):
    prediction = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json={
            "version": version_id,
            "input": {"image": image_url}
        }
    ).json()

st.write("🧠 Replicate response:", prediction)
status = prediction.get("status")

if status in ("starting", "processing"):
    get_url = prediction["urls"]["get"]
    for _ in range(30):
        time.sleep(1)
        pred = requests.get(get_url, headers=headers).json()
        if pred.get("status") == "succeeded":
            dl = pred.get("output")
            break
    else:
        st.error("Generation timed out.")
        st.stop()
elif status == "succeeded":
    dl = prediction.get("output")
else:
    st.error("Failed to start prediction. Check API key or image URL.")
    st.stop()

if dl:
    st.subheader("✏️ Deep-Learning Line Drawing Output")
    st.image(dl, use_container_width=True)
    st.markdown(f"[📥 Download Line Art]({dl})")

# --- Optional: Local Vector Skeletonization ---
st.subheader("🖼️ Vectorized Skeletonization (Local)")
process = st.checkbox("Perform local vector skeletonization (binarize & skeletonize)")

if process:
    # Convert to grayscale
    gray = color.rgb2gray(np.array(image))
    # Binarize via Otsu threshold
    thresh = filters.threshold_otsu(gray)
    bw = gray < thresh
    # Skeletonize
    skeleton = morphology.skeletonize(bw)
    # Convert to uint8 image
    vec_img = util.img_as_ubyte(skeleton)

    st.image(vec_img, caption="Skeleton Vector Output", use_container_width=True, clamp=True)

    # Download skeleton PNG
    buf2 = io.BytesIO()
    Image.fromarray(vec_img).save(buf2, format="PNG")
    st.download_button(
        "📥 Download Vector Skeleton (PNG)",
        data=buf2.getvalue(),
        file_name="skeleton.png",
        mime="image/png"
    )
