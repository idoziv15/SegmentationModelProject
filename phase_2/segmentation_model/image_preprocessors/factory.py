from PIL import Image
import torch
from typing import Union, Any
from .path_preprocessor import PathPreprocessor
from .pil_preprocessor import PILPreprocessor
from .tensor_preprocessor import TensorPreprocessor
from .base_preprocessor import ImagePreprocessor

def get_preprocessor(image: Union[str, Image.Image, torch.Tensor]) -> ImagePreprocessor:
    """
    Returns the appropriate preprocessor based on the type of the input image.

    Depending on whether the input image is a file path (string), PIL image, or PyTorch tensor,
    this function will return an instance of the corresponding preprocessor.

    Args:
        image (Union[str, Image.Image, torch.Tensor]): The input image to preprocess.
            - str: Assumed to be a file path to an image.
            - Image.Image: PIL image instance.
            - torch.Tensor: PyTorch tensor.

    Returns:
        ImagePreprocessor: An instance of a preprocessor suitable for the input image type.

    Raises:
        TypeError: If the image type is unsupported.
    """
    if isinstance(image, str):
        return PathPreprocessor()
    elif isinstance(image, Image.Image):
        return PILPreprocessor()
    elif isinstance(image, torch.Tensor):
        return TensorPreprocessor()
    else:
        raise TypeError("Unsupported image type.")
