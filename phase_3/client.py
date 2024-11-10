import requests
import logging
import argparse
import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_utils import validate_image_path, encode_image, display_segmentation_maps

# Constants for default values
DEFAULT_MODEL_NAME = "deeplabv3_mobilenet_v3_large"
DEFAULT_IMAGE_PATH = os.path.join(project_root, "images", "cat2.jpg")
DEFAULT_USE_ONNX = False
API_URL = "http://127.0.0.1:8000/inference"
IMAGE_FORMAT = "PIL"

def main() -> None:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Send image data for model inference.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="The name of the model to use."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="The path to the input image."
    )
    parser.add_argument(
        "--use_onnx",
        action="store_true",
        help="Flag to indicate whether to use the ONNX model (default is False)."
    )
    args = parser.parse_args()

    # Get values from arguments or use defaults
    model_name = args.model_name
    image_path = args.image_path
    use_onnx = args.use_onnx

    # Validate the image path
    if not validate_image_path(image_path):
        return  # Exit if the image path is invalid

    # Create data request
    data = {
        "model_name": model_name,
        "use_onnx": use_onnx,
        "image_data": encode_image(image_path),
        "image_format": IMAGE_FORMAT
    }

    # Send request to the API
    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        segmentation_map = response.json()["segmentation_map"]
        display_segmentation_maps(segmentation_map)
    else:
        logging.error(f"Error: {response.json()['detail']}")

if __name__ == "__main__":
    main()
