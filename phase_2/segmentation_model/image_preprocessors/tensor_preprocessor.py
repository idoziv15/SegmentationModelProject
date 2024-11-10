from torch import Tensor
from .base_preprocessor import ImagePreprocessor

class TensorPreprocessor(ImagePreprocessor):
    """
    Preprocessor for PyTorch tensors. Ensures the tensor has a batch dimension, adding one
    if necessary to match model input requirements.

    Inherits from:
        ImagePreprocessor: An abstract base class for image preprocessing.
    """

    def preprocess(self, image: Tensor) -> Tensor:
        """
        Ensures the input tensor has a batch dimension, adding a batch dimension of size 1
        if it’s missing.

        Args:
            image (Tensor): The input image tensor.

        Returns:
            Tensor: The processed image tensor with a batch dimension.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image
