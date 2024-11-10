import torch
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
import time
import sys
import os
import logging
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_utils import load_torch_model, validate_image_path

# Constants for image preprocessing
IMAGE_SIZE = (224, 224)                  # Target image size for resizing
NORMALIZE_MEAN = [0.485, 0.456, 0.406]   # Mean for normalization
NORMALIZE_STD = [0.229, 0.224, 0.225]    # Standard deviation for normalization
IMAGE_AMOUNT = 1
CHANNELS = 3
WIDTH = 224
HEIGHT = 224
DEFAULT_ONNX_PATH = "deeplabv3_mobilenet_v3.onnx"
DEFAULT_MODEL_NAME = "deeplabv3_mobilenet_v3_large"
DEFAULT_IMAGE_PATH = os.path.join(project_root, "images", "cat2.jpg")

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Loads and preprocesses an image for the model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor with an added batch dimension.
    """
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return input_tensor

def convert_to_onnx(model: torch.nn.Module, output_path: str = DEFAULT_ONNX_PATH) -> None:
    """
    Converts a PyTorch model to ONNX format and saves it to the specified path.

    Args:
        model (torch.nn.Module): PyTorch model to convert.
        output_path (str): Path to save the ONNX model. Defaults to DEFAULT_ONNX_PATH.
    """
    dummy_input = torch.randn(IMAGE_AMOUNT, CHANNELS, WIDTH, HEIGHT)
    torch.onnx.export(
        model,                      # The PyTorch model to be converted.
        dummy_input,                # to trace the model’s computation graph.
        output_path,                # The path where the ONNX model will be saved.
        export_params=True,         # Includes model weights in the ONNX file.
        opset_version=11,           # Sets the ONNX version for compatibility.
        do_constant_folding=True,   # Optimizes the model by folding constant expressions.
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Allows the batch size to be dynamic.
    )
    logging.info(f"Model has been converted to ONNX format and saved to {output_path}")

def run_pytorch_inference(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> tuple[np.ndarray, float]:
    """
    Runs inference on a PyTorch model and measures runtime.

    Args:
        model (torch.nn.Module): PyTorch model for inference.
        input_tensor (torch.Tensor): Input image tensor.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        tuple: A tuple containing the model output as a numpy array and runtime in seconds.
    """
    # Moves the input tensor to the appropriate device (CPU or GPU)
    input_tensor = input_tensor.to(device)
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)["out"]
    runtime = time.time() - start_time
    output = output.cpu().numpy()
    return output, runtime

def run_onnx_inference(onnx_model_path: str, input_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    """
    Runs inference on an ONNX model and measures runtime.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        input_tensor (torch.Tensor): Input image tensor.

    Returns:
        tuple: A tuple containing the model output as a numpy array and runtime in seconds.
    """
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    input_data = input_tensor.numpy()

    start_time = time.time()
    output = ort_session.run(None, {input_name: input_data})[0]
    runtime = time.time() - start_time
    return output, runtime

def compute_l2_difference(output_pytorch: np.ndarray, output_onnx: np.ndarray) -> float:
    """
    Computes the L2 (Euclidean) norm of the difference between PyTorch and ONNX model outputs.

    Args:
        output_pytorch (np.ndarray): PyTorch model output.
        output_onnx (np.ndarray): ONNX model output.

    Returns:
        float: L2 norm of the difference between the outputs.
    """
    return np.linalg.norm(output_pytorch - output_onnx)

def visualize_differences(output_pytorch: np.ndarray, output_onnx: np.ndarray) -> None:
    """
    Visualizes the differences between the PyTorch and ONNX model outputs.

    Args:
        output_pytorch (np.ndarray): PyTorch model output.
        output_onnx (np.ndarray): ONNX model output.
    """
    output_pytorch = output_pytorch[0]
    output_onnx = output_onnx[0]

    # Calculate the absolute difference across all class probabilities
    diff = np.abs(output_pytorch - output_onnx).sum(axis=0)
    diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    plt.figure(figsize=(14, 6))

    # Display the class predictions of the PyTorch model
    plt.subplot(1, 3, 1)
    plt.title("PyTorch Output (Class Predictions)")
    plt.imshow(output_pytorch.argmax(axis=0), cmap="viridis")

    # Display the class predictions of the ONNX model
    plt.subplot(1, 3, 2)
    plt.title("ONNX Output (Class Predictions)")
    plt.imshow(output_onnx.argmax(axis=0), cmap="viridis")

    # Display the raw difference (heatmap) between PyTorch and ONNX outputs
    plt.subplot(1, 3, 3)
    plt.title("Difference in Probabilities (Heatmap)")
    heatmap = plt.imshow(diff_normalized, cmap="plasma")
    cbar = plt.colorbar(heatmap, label="Difference Intensity", pad=0.1)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
    plt.show()

# Main Program Execution
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run model inference and conversion.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="The name of the model to load."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="The path to the input image."
    )
    args = parser.parse_args()

    # Get model name and image path from arguments or use default values
    model_name = args.model_name
    image_path = args.image_path

    # Validate the image path
    if not validate_image_path(image_path):
        sys.exit()  # Exit if the image path is invalid

    # Get the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the PyTorch model
    model = load_torch_model(model_name).to(device)
    
    # Convert the PyTorch model into ONNX mode
    onnx_output_path = f"{model_name}.onnx"
    convert_to_onnx(model, onnx_output_path)

    # Preprocess the input image
    input_tensor = preprocess_image(image_path)

    # Run inference on PyTorch model
    output_pytorch, runtime_pytorch = run_pytorch_inference(model, input_tensor, device)
    print(f"PyTorch runtime: {runtime_pytorch:.4f} seconds")

    # Run inference on ONNX model
    output_onnx, runtime_onnx = run_onnx_inference(onnx_output_path, input_tensor)
    print(f"ONNX runtime: {runtime_onnx:.4f} seconds")

    # Calculate and print L2 difference
    l2_difference = compute_l2_difference(output_pytorch, output_onnx)
    print(f"L2 Difference between PyTorch and ONNX outputs: {l2_difference:.4f}")

    # Visualize the differences
    visualize_differences(output_pytorch, output_onnx)

if __name__ == "__main__":
    main()