```python
# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# For skeletonization and vector tracing
from skimage.morphology import skeletonize
import potrace

# Page setup
st.set_page_config(page_title="Photo to Vector-Ready Line Art", layout="centered")
st.title("📷➡️🖋️ Photo to Vector-Ready Line Drawing")
st.markdown(
    """
    Upload a JPEG/PNG and convert it into clean, single-pixel skeleton line art—ready for SVG tracing.
    """
)

# Upload
uploaded_file = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Upload an image to get started.")
    st.stop()

# Read and display original image
pil_img = Image.open(uploaded_file).convert("RGB")
img = np.array(pil_img)
st.image(img, caption="Original Image", use_container_width=True)

# Preprocessing: grayscale + blur
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Dilate to close small gaps
kernel = np.ones((2,2), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Skeletonize: convert to boolean then skimage skeletonize
binary = dilated > 0
skeleton = skeletonize(binary)
# Convert skeleton back to uint8 for display
skeleton_img = (skeleton.astype(np.uint8) * 255)

st.header("✏️ Skeletonized Line Art")
st.image(skeleton_img, caption="1-px Skeleton", use_container_width=True)

# Vector tracing with Potrace
bmp = potrace.Bitmap(skeleton.astype(np.uint8))
path = bmp.trace()

# Build SVG from traced paths
def paths_to_svg(path_obj, width, height):
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" ' +
        'xmlns:xlink="http://www.w3.org/1999/xlink">'
    ]
    for curve in path_obj:
        svg_parts.append('<path d="')
        # Move to start point
        start = curve.start_point
        svg_parts.append(f'M {start[0]} {start[1]} ')
        for segment in curve.segments:
            if segment.is_corner:
                c = segment.c
                svg_parts.append(f'L {c[0]} {c[1]} ')
            else:
                c1, c2 = segment.c1, segment.c2
                end = segment.end_point
                svg_parts.append(f'C {c1[0]} {c1[1]}, {c2[0]} {c2[1]}, {end[0]} {end[1]} ')
        svg_parts.append('Z" fill="none" stroke="black" stroke-width="1"/>')
    svg_parts.append('</svg>')
    return "".join(svg_parts)

h, w = skeleton_img.shape
svg_data = paths_to_svg(path, w, h)

st.header("🖋️ Vector-Ready SVG Preview")
st.download_button(
    label="📥 Download SVG",
    data=svg_data,
    file_name="line_art.svg",
    mime="image/svg+xml"
)

# Optional: show raw SVG markup for inspection
with st.expander("Show raw SVG code"):
    st.code(svg_data, language='xml')
```
