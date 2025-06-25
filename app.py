# Streamlit App: Photo to Line Art Using Replicate API

import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(page_title="Photo to Line Art Converter", layout="centered")
st.title("📷➡️✏️ Photo to Line Drawing (Powered by AI)")
st.markdown("Upload your image and get professional-grade line art using a deep learning model.")

# --- Upload ---
uploaded_file = st.file_uploader("Upload a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # Convert image to base64 for Replicate API
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # --- Replicate API call ---
    REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN") or "your-replicate-api-key"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    with st.spinner("Generating line art... this may take ~10 seconds"):
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json={
                "version": "cb4f8b62b431cf2b6d064e7d23647f39b6a63c012a4b9e65f8948a5e9f8a4261",
                "input": {
                    "image": f"data:image/png;base64,{img_base64}",
                    "detect_resolution": 512,
                    "image_resolution": 768,
                    "scribble": False,
                    "return_mask": False,
                    "invert": True
                }
            }
        )

        prediction = response.json()
        status = prediction.get("status")

        if status == "starting" or status == "processing":
            prediction_url = prediction["urls"]["get"]
            import time
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
            st.error("Failed to start prediction. Check API key or input.")
            st.stop()

    # Show and enable download
    st.image(output_url, caption="Line Drawing Output", use_container_width=True)
    st.markdown(f"[Download Line Art PNG]({output_url})")
