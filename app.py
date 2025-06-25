import streamlit as st
import requests
from PIL import Image
import io
import time

# — Page setup —
st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing (Powered by AI)")
st.markdown("Upload your image and get professional-grade line art via a deep learning model.")

# — Upload widget —
uploaded = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()

# — Display original —
img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Original Image", use_container_width=True)

# — Send to ImgBB to get a public URL —
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

imgbb_key = st.secrets.get("IMGBB_API_KEY")
if not imgbb_key:
    st.error("⚠️ IMGBB_API_KEY is missing. Add it under Settings → Secrets.")
    st.stop()

st.info("Uploading image to temporary host…")
resp = requests.post(
    "https://api.imgbb.com/1/upload",
    params={"key": imgbb_key},
    files={"image": buf},
)
if resp.status_code != 200:
    st.error("❌ ImgBB upload failed. Check your IMGBB_API_KEY.")
    st.stop()

image_url = resp.json()["data"]["url"]
st.success("✅ Image URL: " + image_url)

# — Call Replicate —
rep_key = st.secrets.get("REPLICATE_API_TOKEN")
if not rep_key:
    st.error("⚠️ REPLICATE_API_TOKEN is missing. Add it under Settings → Secrets.")
    st.stop()

headers = {
    "Authorization": f"Token {rep_key}",
    "Content-Type": "application/json",
}

st.info("Generating line art… this may take ~10–20 seconds")
payload = {
    "version": "eff7bcd87c2bb1d4de0090634be9e6265ecf80e33e8eae0d4e8a38cd62d43e9a",
    "input": {
        "image": image_url,
        "detect_resolution": 768,
        "image_resolution": 1024
    }
}

r2 = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
pred = r2.json()
st.write("🧠 Replicate response:", pred)

status = pred.get("status")
if status in ("starting", "processing"):
    poll_url = pred["urls"]["get"]
    for _ in range(30):
        time.sleep(1)
        p = requests.get(poll_url, headers=headers).json()
        if p["status"] == "succeeded":
            out_url = p["output"]
            break
    else:
        st.error("❌ Generation timed out.")
        st.stop()
elif status == "succeeded":
    out_url = pred["output"]
else:
    st.error("❌ Failed to start prediction. Check your Replicate API key & version.")
    st.stop()

# — Show & download —
st.header("✏️ Line Drawing Output")
st.image(out_url, use_container_width=True)
st.markdown(f"[📥 Download Line Art PNG]({out_url})")
