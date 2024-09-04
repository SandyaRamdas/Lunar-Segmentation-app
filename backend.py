from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import numpy as np
import io
import tensorflow as tf

from utils import preprocess_image, get_color_map, load_model

# Initialize FastAPI app
app = FastAPI()


# Load the pre-trained segmentation model, including the custom objects
model_path = 'C:\\Users\\user\\OneDrive\\Desktop\\Image segmentation project\\seresnet50LunarModel.h5'
model = load_model(model_path)


@app.get("/")
async def read_root():
    return {"App": "Working"}

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    """
    Endpoint for segmenting an uploaded image.
    Args:
        file (UploadFile): The image file to be segmented.
    Returns:
        StreamingResponse: The segmentation result image in PNG format.
        JSONResponse: Error details if something goes wrong.
    """
    try:
        # Read the image file into a BytesIO object
        image_bytes = await file.read()
        image_file = io.BytesIO(image_bytes)

        # Preprocess the image for the model
        image_array = preprocess_image(image_file, streamlit_use=False)

        # Perform segmentation using the loaded model
        pred_mask = model.predict(np.expand_dims(image_array, axis=0))
        pred_mask = np.argmax(pred_mask, axis=-1)  # Determine the most probable class for each pixel
        pred_mask = pred_mask[0]  # Remove batch dimension

        # Map the predicted mask to colors
        color_map = get_color_map()
        segmentation_img = color_map[pred_mask]  # Convert class indices to color values

        # Convert the colored segmentation image to a PIL Image
        segmentation_img_pil = Image.fromarray(segmentation_img)

        # Save the PIL image to a BytesIO object
        img_byte_arr = io.BytesIO()
        segmentation_img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the segmentation image as a streaming response
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except HTTPException as e:
        # Return an HTTP exception response with details
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        # Return a generic error response if an unexpected error occurs
        return JSONResponse(content={"error": str(e)}, status_code=500)
