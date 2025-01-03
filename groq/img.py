import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Define API details
API_KEY = "SG_6f72e4b5ed6f976f"
API_URL = "https://api.segmind.com/v1/sdxl1.0-txt2img"

# Function to call the API and generate an image
def generate_image(prompt, negative_prompt, img_width, img_height, num_inference_steps, guidance_scale, seed):
    # Payload for the API
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "style": "base",
        "samples": 1,
        "scheduler": "UniPC",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "strength": 0.2,
        "high_noise_fraction": 0.8,
        "seed": seed,
        "img_width": img_width,
        "img_height": img_height,
        "refiner": True,
        "base64": True  # Ensure we get the response as base64
    }
    headers = {'x-api-key': API_KEY}
    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        if "image" in result:
            # Decode the base64 image
            image_data = base64.b64decode(result["image"])
            return Image.open(BytesIO(image_data))
        else:
            st.error("No image found in the response.")
    else:
        st.error(f"Failed to generate image. Error: {response.status_code}, {response.text}")
    return None

# Streamlit app interface
st.title("Image Generation App")
st.markdown("Enter a prompt to generate an AI-powered image using the SDXL API.")

# Input fields
prompt = st.text_area("Prompt", "Enter your image description here...")
negative_prompt = st.text_area("Negative Prompt", "ugly, poorly drawn, blurry...")
img_width = st.slider("Image Width", 512, 1024, 896, step=32)
img_height = st.slider("Image Height", 512, 1024, 1152, step=32)
num_inference_steps = st.slider("Inference Steps", 10, 50, 25)
guidance_scale = st.slider("Guidance Scale", 1, 20, 8)
seed = st.number_input("Seed (for reproducibility)", value=42)

# Button to generate the image
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        generated_image = generate_image(
            prompt, negative_prompt, img_width, img_height, num_inference_steps, guidance_scale, seed
        )
        if generated_image:
            st.image(generated_image, caption="Generated Image", use_column_width=True)
