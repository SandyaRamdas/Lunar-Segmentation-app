import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import io
from PIL import Image
import numpy as np
from skimage.io import imread  # Import the imread function from skimage for image reading
import tensorflow as tf

def preprocess_image(image_file: io.BytesIO, streamlit_use=True) -> np.ndarray:
    """
    Preprocess the image for the model.
    Args:
        image_file (io.BytesIO): In-memory image file as bytes.
        streamlit_use (bool): Flag to determine if the image should be returned as a PIL Image 
                              (for Streamlit) or as a numpy array (for model prediction).
    Returns:
        np.ndarray or PIL.Image: Preprocessed image as a numpy array or PIL Image, 
                                 depending on the `streamlit_use` flag.
    """
    H, W = 480, 480  # Define target height and width for cropping

    # Read the image from the in-memory file object
    img = imread(image_file)

    # Check if the image dimensions meet the minimum required size
    original_height, original_width = img.shape[:2]
    if original_height < H or original_width < W:
        raise ValueError("Image must be at least 480x480 pixels. Provided image dimensions are too small.")

    # Crop the image to the target dimensions (HxW)
    img_resized = img[:H, :W, :]

    # Normalize the image to the range [0, 1] and convert to float32
    normalized_img = img_resized / 255.0
    float32_img = normalized_img.astype(np.float32)

    if streamlit_use:
        # Convert the normalized numpy array back to a PIL Image for display in Streamlit
        preprocessed_image = Image.fromarray((float32_img * 255).astype(np.uint8))
        return preprocessed_image
    else:
        # Return the preprocessed image as a numpy array for model prediction
        return float32_img

def get_color_map():
    """
    Define a color map for visualizing segmentation masks.
    Returns:
        np.ndarray: Array mapping class indices to RGB colors.
    """
    return np.array([
        [0, 0, 0],       # Class 0: Black (Lunar Soil / Background)
        [255, 0, 0],     # Class 1: Red (Large Rocks)
        [0, 255, 0],     # Class 2: Green (Sky)
        [0, 0, 255]      # Class 3: Blue (Small Rocks)
        # Add more colors as needed for additional classes
    ], dtype=np.uint8)
    
def load_model(model_path: str):
    """
    Load a pre-trained TensorFlow model from a file.
    Args:
        model_path (str): Path to the saved model file.
    Returns:
        tf.keras.Model: Loaded TensorFlow model.
    """
    return tf.keras.models.load_model(model_path, compile=False)





