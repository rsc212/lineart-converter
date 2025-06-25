# Streamlit App: Photo to Line Art Using Replicate API (with image upload to ImgBB)

import streamlit as st
import requests
from PIL import Image
import io
import time

st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing (Powered by AI)")
st.markdown("Upload your image and get professional-grade line art using a deep learning model.")

# --- Upload ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Save to buffer
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    # Upload image to ImgBB (or any image host that gives a public URL)
    st.info("Uploading image to temporary host...")
    imgbb_api_key = "YOUR_IMGBB_API_KEY"
    res = requests.post(
        "https://api.imgbb.com/1/upload",
        params={"key": imgbb_api_key},
        files={"image": buffered}
    )

    if res.status_code != 200:
        st.error("Image hosting failed. Check your ImgBB API key.")
        st.stop()

    image_url = res.json()["data"]["url"]

    # --- Replicate API call ---
    REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    with st.spinner("Generating line art... this may take ~10 seconds"):
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json={
                "version": "eff7bcd87c2bb1d4de0090634be9e6265ecf80e33e8eae0d4e8a38cd62d43e9a",
                "input": {
                    "image": image_url,
                    "detect_resolution": 768,
                    "image_resolution": 1024
                }
            }
        )

        prediction = response.json()
        status = prediction.get("status")

        if status in ["starting", "processing"]:
            prediction_url = prediction["urls"]["get"]
            for _ in range(30):
                time.sleep(1)
                status_resp = requests.get(prediction_url, headers=headers).json()
                if status_resp["status"] == "succeeded":
                    output_url = status_resp["output"]
                    break
            else:
                st.error("Generation took too long or failed.")
                st.stop()
        elif status == "succeeded":
            output_url = prediction.get("output")
        else:
            st.error("Failed to start prediction. Check Replicate API key or image URL.")
            st.stop()

    if output_url:
        st.image(output_url, caption="Line Drawing Output", use_container_width=True)
        st.markdown(f"[Download Line Art PNG]({output_url})")
    else:
        st.error("The model did not return an image. Please try again with a different photo.")
