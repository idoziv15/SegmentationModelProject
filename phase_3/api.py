import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from phase_1.convert import convert_to_onnx
from model_utils import load_torch_model, decode_image
from phase_3.inference_request import InferenceRequest
from phase_2.segmentation_model.segmentation_ai import SegmentationModelAI

# Initialize the FastAPI app
app = FastAPI()

# Define path for ONNX models directory
ONNX_DIR = "onnx_models/"
os.makedirs(ONNX_DIR, exist_ok=True)

@app.post("/inference")
def inference(request: InferenceRequest) -> JSONResponse:
    """
    Perform inference on an uploaded image using either the PyTorch or ONNX model.

    Args:
        request (InferenceRequest): The request payload containing model name, use_onnx flag, image data, and image format.

    Returns:
        JSONResponse: A JSON object containing the segmentation map as a list of lists.

    Raises:
        HTTPException: If an error occurs during model loading or inference.
    """
    try:
        pytorch_model = load_torch_model(request.model_name)

        if request.use_onnx:
            onnx_path = os.path.join(ONNX_DIR, f"{request.model_name}.onnx")
            if not os.path.exists(onnx_path):
                convert_to_onnx(pytorch_model, onnx_path)
            model_instance = SegmentationModelAI(onnx_path)
        else:
            model_instance = SegmentationModelAI(pytorch_model)

        # Decode and preprocess the image based on format
        image = decode_image(request.image_data, request.image_format)

        # Run inference
        segmentation_map = model_instance(image)

        # Convert segmentation map to a list format for JSON serialization
        if hasattr(segmentation_map, "tolist"):
            segmentation_map_list = segmentation_map.tolist()
        else:
            segmentation_map_list = list(segmentation_map)

        return JSONResponse(content={"segmentation_map": segmentation_map_list})

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
