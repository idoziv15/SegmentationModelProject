import sys
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, Response
from PIL import Image
import io
import numpy as np

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from phase_1.convert import convert_to_onnx
from model_utils import load_torch_model, decode_image
from phase_3.inference_request import InferenceRequest
from phase_2.segmentation_model.segmentation_ai import SegmentationModelAI

# Initialize the FastAPI app
app = FastAPI(
    title="Segmentation Model API",
    description="API for image segmentation using PyTorch or ONNX models",
    version="1.0.0"
)

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

@app.post("/segment_image", 
         summary="Segment an uploaded image",
         description="Upload an image and receive its segmentation map as a PNG image",
         response_class=Response,
         responses={
             200: {
                 "content": {"image/png": {}},
                 "description": "Returns the segmentation map as a PNG image",
                 "headers": {
                     "Content-Type": {"description": "image/png"},
                     "Content-Disposition": {
                         "description": "attachment; filename=segmentation_map.png"
                     }
                 }
             }
         })
async def segment_image(
    file: UploadFile = File(..., description="Image file to segment (JPG, PNG)"),
    model_name: str = "deeplabv3_mobilenet_v3_large",
    use_onnx: bool = False
) -> Response:
    """
    Segment an uploaded image and return the segmentation map as an image.

    Args:
        file (UploadFile): The uploaded image file
        model_name (str): Name of the model to use for segmentation
        use_onnx (bool): Whether to use ONNX model

    Returns:
        Response: PNG image of the segmentation map
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert image to RGB if it has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Load and prepare the model
        pytorch_model = load_torch_model(model_name)
        
        if use_onnx:
            onnx_path = os.path.join(ONNX_DIR, f"{model_name}.onnx")
            if not os.path.exists(onnx_path):                convert_to_onnx(pytorch_model, onnx_path)
            model_instance = SegmentationModelAI(onnx_path)
        else:
            model_instance = SegmentationModelAI(pytorch_model)

        # Run inference
        segmentation_map = model_instance(image)
        
        # Convert segmentation map to image
        # Scale to 0-255 range for better visualization
        seg_image = (segmentation_map * 255 / segmentation_map.max()).astype(np.uint8)
        pil_image = Image.fromarray(seg_image)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(
            content=img_byte_arr, 
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=segmentation_map.png"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
