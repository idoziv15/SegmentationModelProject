import numpy as np
from typing import Any
import onnxruntime as ort
from .base_model import SegmentationModel
from ..image_preprocessors import get_preprocessor

class ONNXSegmentationModel(SegmentationModel):
    """
    A subclass of SegmentationModel for ONNX-based segmentation models.

    This class manages loading, preprocessing, and inference operations for an ONNX 
    segmentation model using ONNX Runtime, implementing the base class interface.
    
    Attributes:
        ort_session (onnxruntime.InferenceSession): The ONNX Runtime session used for inference.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the ONNXSegmentationModel with a specified ONNX model path.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        super().__init__()
        self.ort_session = None
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Loads the ONNX model from the specified file path and initializes the ONNX Runtime session.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        self.ort_session = ort.InferenceSession(model_path)

    def preprocess_image(self, image: Any) -> np.ndarray:
        """
        Preprocesses the input image and prepares it for the ONNX model.

        Args:
            image: The input image in various formats (file path, PIL image, tensor etc.).
        
        Returns:
            numpy.ndarray: The preprocessed image array, formatted for ONNX model input.
        """
        # Skip preprocessing if already in numpy format
        if isinstance(image, np.ndarray):
            return image
        
        preprocessor = get_preprocessor(image)
        preprocessed_image = preprocessor.preprocess(image).numpy()
        
        return preprocessed_image

    def run_inference(self, image: Any) -> np.ndarray:
        """
        Runs inference on the preprocessed image and returns the segmentation map.

        Args:
            image: The input image to run inference on.
        
        Returns:
            numpy.ndarray: The segmentation map output, where each pixel is assigned a class index.
        """
        # Preprocess the image for inference
        preprocessed_image = self.preprocess_image(image)
        input_name = self.ort_session.get_inputs()[0].name
        
        output = self.ort_session.run(None, {input_name: preprocessed_image})[0]
        
        # Returns the segmentation map
        return output[0].argmax(axis=0)
