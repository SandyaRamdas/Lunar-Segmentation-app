from PIL import Image
import io
import streamlit as st
import requests
from utils import preprocess_image

# URL of your FastAPI backend endpoint for image segmentation
FASTAPI_URL = "http://localhost:8000/segment/"

# Streamlit application setup
st.set_page_config(page_title="Image Segmentation App", layout="wide")
st.title("🔍 Image Segmentation App")

# Sidebar settings
st.sidebar.title("Settings")
theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark", "Classic"])
threshold = st.sidebar.slider("Segmentation Threshold", 0.0, 1.0, 0.3)

# File uploader widget to allow users to upload an image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert the uploaded file into an in-memory bytes object
    image_bytes = io.BytesIO(uploaded_file.read())

    try:
        # Preprocess the uploaded image using the custom preprocessing function
        preprocessed_image = preprocess_image(image_bytes, streamlit_use=True)

        # Display original and preprocessed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
        with col2:
            st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

        # Button to trigger the segmentation process
        if st.button("🚀 Segment Image"):
            with st.spinner("Segmenting image..."):
                # Convert the preprocessed image to bytes for sending in the POST request
                preprocessed_image_bytes = io.BytesIO()
                preprocessed_image.save(preprocessed_image_bytes, format="PNG")
                preprocessed_image_bytes.seek(0)

                # Prepare the file to send in the POST request
                files = {"file": ("preprocessed_image.png", preprocessed_image_bytes, "image/png")}

                # Send a POST request to the FastAPI backend with the preprocessed image
                response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                # If the request is successful, display the segmented image returned by the backend
                result_image = Image.open(io.BytesIO(response.content))
                st.image(result_image, caption="Segmented Image", use_column_width=True)

                # Provide download button for segmented image
                st.download_button(
                    label="💾 Download Segmented Image",
                    data=response.content,
                    file_name="segmented_image.png",
                    mime="image/png",
                )
            else:
                # If there's an error, display an error message
                st.error("Error: Unable to process the image. Please try again.")

    except ValueError as e:
        # Display error message if preprocessing fails
        st.error(f"Error: {e}")
