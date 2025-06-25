# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing (Local)")

st.markdown(
    """
    Upload a JPEG/PNG and this app will convert it into printable line art  
    using OpenCV’s Canny edge detector—no external API calls needed.
    """
)

uploaded = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Nothing uploaded yet.")
    st.stop()

# 1) Read & show original
img = Image.open(uploaded).convert("RGB")
img_np = np.array(img)
st.image(img_np, caption="Original Image", use_column_width=True)

# 2) Convert to grayscale + blur
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3) Canny edge detection
edges = cv2.Canny(blur, threshold1=50, threshold2=150)

# 4) (Optional) Dilate to thicken lines
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# 5) Invert so lines are black on white
line_art = cv2.bitwise_not(dilated)

st.header("✏️ Line Drawing Output")
st.image(line_art, caption="Line Art", use_column_width=True)

# 6) Provide download button
buf = io.BytesIO()
Image.fromarray(line_art).save(buf, format="PNG")
buf.seek(0)
st.download_button(
    label="📥 Download Line Art (PNG)",
    data=buf,
    file_name="line_art.png",
    mime="image/png",
)
