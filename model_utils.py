import torchvision.models.segmentation as tv_segmentation
import os
import torch
import base64
import logging
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from typing import Union

def load_torch_model(model_name: str) -> torch.nn.Module:
    """
    Checks if the given model exists in torchvision.models.segmentation
    and if so, loads the corresponding segmentation model.

    Args:
        model_name (str): Name of the segmentation model to load.

    Returns:
        torch.nn.Module: Instantiated PyTorch model.

    Raises:
        ValueError: If the model is not found in torchvision.models.segmentation.
    """
    if hasattr(tv_segmentation, model_name):
        model_class = getattr(tv_segmentation, model_name)
        model = model_class(pretrained=True)
        model.eval()
        logging.info("Model loaded successfully!")
        return model
    else:
        raise ValueError(f"ERROR: Model '{model_name}' was not found in torchvision.models.segmentation.")
    
def validate_image_path(image_path: str) -> bool:
    """
    Validate if the given path exists and is a valid image.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        bool: True if the path exists and is a valid image, False otherwise.
    """
    if not os.path.exists(image_path):
        logging.error(f"Error: The file '{image_path}' does not exist.")
        return False
    try:
        # Try opening the image to confirm it's valid
        with Image.open(image_path) as img:
            img.verify()  # Verify the image format
        return True
    except (IOError, SyntaxError) as e:
        logging.error(f"Error: The file '{image_path}' is not a valid image. {e}")
        return False

def decode_image(image_data_base64: str, image_format: str) -> Union[Image.Image, str, transforms.ToTensor]:
    """
    Decodes base64 image data and converts it to the specified format.

    Args:
        image_data_base64 (str): Base64 encoded image data.
        image_format (str): Desired format ("PIL", "tensor", or "path").

    Returns:
        Union[Image.Image, str, transforms.ToTensor]: Processed image in the requested format.

    Raises:
        ValueError: If an unsupported image format is specified.
    """
    # Decode base64 image data
    image_data = base64.b64decode(image_data_base64)

    if image_format == "PIL":
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image

    elif image_format == "tensor":
        image = Image.open(BytesIO(image_data)).convert("RGB")
        transform = transforms.ToTensor()
        return transform(image).unsqueeze(0)  # Add batch dimension for tensor

    elif image_format == "path":
        # If 'path' format is specified, save to a temporary file
        temp_image_path = "temp_image.png"
        with open(temp_image_path, "wb") as f:
            f.write(image_data)
        return temp_image_path

    else:
        raise ValueError("Unsupported image format.")
    
def encode_image(image_path: str) -> str:
    """
    Read an image file, convert it to bytes, and encode it in base64 format.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: Base64 encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return base64.b64encode(image_bytes).decode("utf-8")

def display_segmentation_maps(segmentation_map: List[List[int]]) -> None:
    """
    Display segmentation map returned from the server.

    Args:
        segmentation_map (list): A 2D list representing the segmentation map.
    """
    plt.figure(figsize=(5, 5))
    plt.title("Segmentation Map")
    plt.imshow(np.array(segmentation_map), cmap="viridis")
    plt.show()