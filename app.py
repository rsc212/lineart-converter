import streamlit as st
import requests
from PIL import Image
import io
import time

st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing (Powered by AI)")
st.markdown("Upload your image and get professional-grade line art using a deep learning model.")

# --- 1) File uploader ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

if not uploaded_file:
    st.info("Please upload an image to get started.")
    st.stop()

# Display original
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Original Image", use_container_width=True)

# Save to buffer PNG
buffered = io.BytesIO()
image.save(buffered, format="PNG")
buffered.seek(0)

# --- 2) Upload to ImgBB ---
st.info("Uploading image to temporary host…")
imgbb_api_key = st.secrets.get("IMGBB_API_KEY")
if not imgbb_api_key:
    st.error("🔑 IMGBB API key not found. Set `IMGBB_API_KEY` in Secrets.")
    st.stop()

resp = requests.post(
    "https://api.imgbb.com/1/upload",
    params={"key": imgbb_api_key},
    files={"image": buffered}
)
if resp.status_code != 200:
    st.error("❌ ImgBB upload failed. Check your IMGBB key.")
    st.stop()

image_url = resp.json()["data"]["url"]
st.success(f"✅ Image URL: {image_url}")

# --- 3) Call Replicate ---
replicate_token = st.secrets.get("REPLICATE_API_TOKEN")
if not replicate_token:
    st.error("🔑 Replicate API token not found. Set `REPLICATE_API_TOKEN` in Secrets.")
    st.stop()

headers = {
    "Authorization": f"Token {replicate_token}",
    "Content-Type": "application/json",
}

# ←––– **UPDATE THIS** to a model version you have access to!
version_id = "jagilley/controlnet-hed:cde353130c86f37d0af4060cd757ab3009cac68eb58df216768f907f0d0a0653"

with st.spinner("Generating line art… this may take ~15 seconds"):
    r = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json={
            "version": version_id,
            "input": {
                "input_image": image_url,
                "detect_resolution": 512,
                "image_resolution": 512,
                "scale": 9,
                "ddim_steps": 20,
                # for “line drawing” style you can override the prompt:
                "prompt": "line art sketch, black and white, high contrast",
                "a_prompt": "best quality, extremely detailed",
                "n_prompt": "longbody, lowres, bad anatomy"
            },
        },
    )
    pred = r.json()
    st.json(pred, expanded=False)

    status = pred.get("status")
    output_url = None

    if status in ("starting", "processing"):
        poll_url = pred["urls"]["get"]
        for _ in range(60):
            time.sleep(1)
            p = requests.get(poll_url, headers=headers).json()
            if p["status"] == "succeeded":
                output_url = p["output"][0]
                break
    elif status == "succeeded":
        output_url = pred["output"][0]

    if not output_url:
        st.error("❌ Generation failed. Check your API key, version, or image URL.")
        st.stop()

# Show result
st.image(output_url, caption="✏️ Line Art Output", use_container_width=True)
st.markdown(f"[📥 Download Line Art]({output_url})")
