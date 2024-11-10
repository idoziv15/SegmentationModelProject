from abc import ABC, abstractmethod
from torchvision import transforms
from typing import Any

"""
This module defines the abstract base class `SegmentationModel`, which provides an interface
for building segmentation models compatible with various frameworks (like PyTorch, ONNX etc.).
It includes basic image preprocessing transformations and enforces a standard interface
for model loading, image preprocessing, and inference operations.
"""
# Constants for image preprocessing
IMAGE_SIZE = (224, 224)                  # Target image size for resizing
NORMALIZE_MEAN = [0.485, 0.456, 0.406]   # Mean for normalization
NORMALIZE_STD = [0.229, 0.224, 0.225]    # Standard deviation for normalization

class SegmentationModel(ABC):
    """
    Abstract base class for segmentation models.
    This class defines the abstract base class `SegmentationModel`, which provides an interface
    for building segmentation models compatible with various frameworks (like PyTorch, ONNX etc.).
    It includes basic image preprocessing transformations and enforces a standard interface
    for model loading, image preprocessing, and inference operations.
    """
    
    def __init__(self) -> None:
        """
        Initializes the SegmentationModel with a default image preprocessing pipeline.

        The preprocessing pipeline includes:
        - Resizing the image to a standard size.
        - Converting the image to a PyTorch tensor.
        - Normalizing the image with mean and standard deviation values based on ImageNet.
        """
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Abstract method for loading the model.

        This method is implemented by subclasses to load the appropriate
        model architecture and weights.
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: Any) -> Any:
        """
        Abstract method for preprocessing the input image.

        This method is implemented by subclasses to ensure the input image is
        in a compatible format for the model. Preprocessing steps may vary depending on
        the type of model being used.

        Args:
            image: The input image to preprocess. It could be either file path, PIL image,
                   or tensor.
        
        Returns:
            The preprocessed image in a format compatible with the model.
        """
        pass

    @abstractmethod
    def run_inference(self, image: Any) -> Any:
        """
        Abstract method for performing inference on the input image.

        This method is implemented by subclasses to handle model-specific
        inference processes and return a segmentation map outputs.

        Args:
            image: The preprocessed input image to run inference on.
        
        Returns:
            The model's output - a segmentation map.
        """
        pass
