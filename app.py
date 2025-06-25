```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# For skeletonization and vector tracing
from skimage.morphology import skeletonize
import potrace

# Streamlit page config
st.set_page_config(page_title="Photo to Vector-Ready Line Art", layout="centered")
st.title("📷➡️🖋️ Photo to Vector-Ready Line Drawing")
st.markdown(
    """
    Upload a JPEG/PNG and convert it into clean, single-pixel skeleton line art—ready for SVG tracing.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("Upload an image to get started.")
    st.stop()

# Load image
pil_img = Image.open(uploaded_file).convert("RGB")
img = np.array(pil_img)
st.image(img, caption="Original Image", use_container_width=True)

# Convert to grayscale and blur
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Dilation to close gaps
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Skeletonize the binary edge map
binary = dilated > 0
skeleton = skeletonize(binary)
# Convert skeleton boolean to grayscale image
skeleton_img = (skeleton.astype(np.uint8) * 255)
st.header("✏️ Skeletonized Line Art")
st.image(skeleton_img, caption="1-px Skeleton", use_container_width=True)

# Vector tracing via Potrace
bmp = potrace.Bitmap(skeleton.astype(np.uint8))
path = bmp.trace()

# Helper to convert Potrace paths into SVG
def paths_to_svg(paths, width, height):
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'  
    ]
    for curve in paths:
        svg_parts.append('<path d="')
        start = curve.start_point
        svg_parts.append(f'M {start[0]} {start[1]} ')
        for segment in curve.segments:
            if segment.is_corner:
                x, y = segment.c
                svg_parts.append(f'L {x} {y} ')
            else:
                c1, c2 = segment.c1, segment.c2
                end = segment.end_point
                svg_parts.append(
                    f'C {c1[0]} {c1[1]},{c2[0]} {c2[1]},{end[0]} {end[1]} '
                )
        svg_parts.append('Z" fill="none" stroke="black" stroke-width="1"/>')
    svg_parts.append('</svg>')
    return ''.join(svg_parts)

# Generate SVG
h, w = skeleton_img.shape
svg_data = paths_to_svg(path, w, h)

# Present SVG preview and download button
st.header("🖋️ Vector-Ready SVG Preview")
st.download_button(
    label="📥 Download SVG",
    data=svg_data,
    file_name="line_art.svg",
    mime="image/svg+xml"
)

# Optionally show raw SVG code
with st.expander("Show raw SVG code"):
    st.code(svg_data, language='xml')
```
