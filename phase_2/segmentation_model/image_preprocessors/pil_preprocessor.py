from torch import Tensor
from PIL import Image
from .base_preprocessor import ImagePreprocessor

class PILPreprocessor(ImagePreprocessor):
    """
    Preprocessor for PIL images. Applies transformations including resizing, normalization,
    and conversion to a tensor, adding a batch dimension to the output.

    Inherits from:
        ImagePreprocessor: An abstract base class for image preprocessing.
    """

    def preprocess(self, image: Image.Image) -> Tensor:
        """
        Applies transformations to a PIL image, including resizing, normalization, and
        conversion to a tensor with an added batch dimension.

        Args:
            image (Image.Image): The input PIL image.

        Returns:
            Tensor: A tensor representing the transformed image with an added batch dimension.
        """
        return self.transform(image).unsqueeze(0)
