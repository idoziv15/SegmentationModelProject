from abc import ABC, abstractmethod
from typing import Any
from torchvision import transforms
from torch import Tensor

# Constants for image preprocessing
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

class ImagePreprocessor(ABC):
    """
    Abstract base class for image preprocessors.

    This class defines a standardized preprocessing pipeline for images, including resizing, 
    normalization, and conversion to a tensor format compatible with PyTorch models.
    It also enforces a preprocess method to be implemented by subclasses, allowing for
    flexible handling of various image input types (e.g., file paths, PIL images, or tensors).
    """
    def __init__(self) -> None:
        """
        Initializes the ImagePreprocessor with a default preprocessing pipeline.

        The pipeline includes:
        - Resizing the image to a fixed size.
        - Converting the image to a PyTorch tensor.
        - Normalizing the image using mean and standard deviation values.
        """
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

    @abstractmethod
    def preprocess(self, image: Any) -> Tensor:
        """
        Abstract method for preprocessing an image.

        This method should be implemented by subclasses to preprocess the input image
        according to the format of the image (e.g., file path, PIL image, tensor).

        Args:
            image (Any): The input image to preprocess.

        Returns:
            Tensor: The preprocessed image as a tensor, ready for model input.
        """
        pass
