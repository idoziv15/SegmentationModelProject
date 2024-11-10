from PIL import Image
from torch import Tensor
from typing import Union
from .base_preprocessor import ImagePreprocessor

class PathPreprocessor(ImagePreprocessor):
    """
    Preprocessor for image file paths. Loads an image from a file path and applies standard
    transformations, including resizing, normalization, and conversion to a tensor.

    Inherits from:
        ImagePreprocessor: An abstract base class for image preprocessing.
    """

    def preprocess(self, image: Union[str, Image.Image]) -> Tensor:
        """
        Loads an image from a file path, converts it to RGB, and applies transformations.

        Args:
            image (Union[str, Image.Image]): The file path to the image or a PIL image.

        Returns:
            Tensor: A tensor representing the transformed image, with an added batch dimension.
        """
        image = Image.open(image).convert("RGB")
        return self.transform(image).unsqueeze(0)
