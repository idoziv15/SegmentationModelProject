from pydantic import BaseModel

class InferenceRequest(BaseModel):
    """
    Data model for the inference request payload.

    Attributes:
        model_name (str): Name of the segmentation model to use.
        use_onnx (bool): Specifies whether to use ONNX model if available.
        image_data (str): Base64 encoded image data.
        image_format (str): Format of the image, e.g., "PIL", "tensor", or "path".
    """
    model_name: str
    use_onnx: bool
    image_data: str  # Base64 encoded image data
    image_format: str