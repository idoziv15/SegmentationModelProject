import torch
from typing import Any
from torch import Tensor
import numpy as np
from .base_model import SegmentationModel
from ..image_preprocessors import get_preprocessor

class TorchSegmentationModel(SegmentationModel):
    """
    A subclass of SegmentationModel for PyTorch-based segmentation models.

    This class handles loading, preprocessing, and running inference on a PyTorch model,
    utilizing the base class interface defined in SegmentationModel.
    
    Attributes:
        model (torch.nn.Module): The PyTorch model used for segmentation.
        device (torch.device): Specifies whether to use a GPU (if available) or CPU.
    """
    
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initializes the TorchSegmentationModel with a given PyTorch model.

        Args:
            model (torch.nn.Module): The pre-trained PyTorch model instance for segmentation.
        """
        super().__init__()
        self.model = None
        # Set the device to GPU if available; otherwise, use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model)

    def load_model(self, model: torch.nn.Module) -> None:
        """
        Loads the provided PyTorch model and moves it to the appropriate device.

        Args:
            model (torch.nn.Module): The PyTorch model to load.
        """
        # Move the model to the selected device (GPU or CPU)
        self.model = model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: Any) -> Tensor:
        """
        Preprocesses the input image based on its format and prepares it for the model.

        Args:
            image: The input image in various formats (file path, PIL image, tensor etc.).
        
        Returns:
            Tensor: The preprocessed image tensor, moved to the appropriate device.
        """
        # Select the appropriate preprocessor for the image type
        preprocessor = get_preprocessor(image)
        # Preprocess the image and move it to the correct device (GPU or CPU)
        return preprocessor.preprocess(image).to(self.device)

    def run_inference(self, image: Any) -> np.ndarray:
        """
        Runs inference on the preprocessed image and returns the segmentation map.

        Args:
            image: The input image to run inference on.
        
        Returns:
            numpy.ndarray: The segmentation map output, with each pixel assigned a class index.
        """
        # Preprocess the image for inference
        preprocessed_image = self.preprocess_image(image)
        with torch.no_grad():
            output = self.model(preprocessed_image)["out"].cpu().numpy()
        
        # Select the class with the highest score for each pixel in the output
        return output[0].argmax(axis=0)
