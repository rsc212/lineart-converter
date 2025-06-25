# Streamlit App: Photo to Line Art Using Replicate API + ImgBB

import streamlit as st
import requests
from PIL import Image
import io
import time

st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing (Powered by AI)")
st.markdown("Upload your image and get professional-grade line art using a deep learning model.")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Save to memory buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Upload to ImgBB
    st.info("Uploading image to temporary host...")
    imgbb_api_key = st.secrets.get("IMGBB_API_KEY")
    if not imgbb_api_key:
        st.error("IMGBB API key not found. Please set it as a Streamlit secret.")
        st.stop()

    res = requests.post(
        "https://api.imgbb.com/1/upload",
        params={"key": imgbb_api_key},
        files={"image": buffer}
    )

    if res.status_code != 200:
        st.error("Image hosting failed. Check your ImgBB API key.")
        st.stop()

    image_url = res.json()["data"]["url"]
    st.write("✅ Image URL:", image_url)

    # --- Call Replicate API ---
    replicate_api_token = st.secrets.get("REPLICATE_API_TOKEN")
    if not replicate_api_token:
        st.error("Replicate API token not found. Please set it as a Streamlit secret.")
        st.stop()

    headers = {
        "Authorization": f"Token {replicate_api_token}",
        "Content-Type": "application/json"
    }

    with st.spinner("Generating line art... this may take ~10 seconds"):
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json={
                "version": "fcb14999f4b54db9a7e7ff7fb5d5b6ec5a4b30392e0d342ec23dbff712d702b3",  # v1.1 of controlnet-scribble
                "input": {
                    "image": image_url,
                    "detect_resolution": 512,
                    "image_resolution": 512
                }
            }
        )

        prediction = response.json()
        st.write("🧠 Replicate response:", prediction)

        if prediction.get("status") not in ["starting", "processing", "succeeded"]:
            st.error("Failed to start prediction. Check your API key and input image.")
            st.stop()

        prediction_url = prediction.get("urls", {}).get("get")
        if not prediction_url:
            st.error("Missing prediction polling URL from Replicate.")
            st.stop()

        # Wait for result
        output_url = None
        for _ in range(30):
            time.sleep(1)
            status_resp = requests.get(prediction_url, headers=headers).json()
            if status_resp["status"] == "succeeded":
                output_url = status_resp.get("output")
                break
            elif status_resp["status"] == "failed":
                st.error("Prediction failed. Try a different photo.")
                st.stop()

    if output_url:
        st.image(output_url, caption="Line Drawing Output", use_container_width=True)
        st.markdown(f"[Download Line Art PNG]({output_url})")
    else:
        st.error("Model returned no output. Please try again with a different photo.")
