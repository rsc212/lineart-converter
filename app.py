import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from skimage.morphology import skeletonize

st.set_page_config(page_title="Photo → Line Art", layout="wide")
st.title("📷 ➡️ 🖊 Photo to Line Art")

st.markdown(
    """
    Upload a photo and instantly get **print-ready**, **vector-style** line art.
    This runs 100% in Python—no external APIs or hidden secrets.
    """
)

uploaded = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to proceed.")
    st.stop()

# Read into OpenCV
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

st.subheader("Original Photo")
st.image(orig_rgb, use_column_width=True)

# ---- Edge detection ----
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
# You can tweak these thresholds interactively if you like:
low_thresh, high_thresh = 50, 150
edges = cv2.Canny(gray, low_thresh, high_thresh)

# ---- Skeletonize for crispness ----
# skeletonize() expects boolean array (True=foreground)
# we invert edges so white edges → True
bw = edges > 0
skel = skeletonize(bw)
skel_u8 = (skel * 255).astype(np.uint8)

# Convert back to PIL for display & download
pil_line = Image.fromarray(skel_u8)

st.subheader("🖊 Line Art Output")
st.image(pil_line, use_column_width=True)

# Download button
buf = io.BytesIO()
pil_line.save(buf, format="PNG")
buf.seek(0)
st.download_button(
    "🚀 Download Line Art as PNG",
    data=buf,
    file_name="line_art.png",
    mime="image/png"
)
