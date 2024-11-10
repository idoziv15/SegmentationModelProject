import torch
from .models_classes.torch_model import TorchSegmentationModel
from .models_classes.onnx_model import ONNXSegmentationModel

class SegmentationModelAI:
    def __init__(self, model):
        """
        Factory for creating segmentation models.
        
        Args:
            model (torch.nn.Module or str): PyTorch model instance or path to ONNX model.
        """
        if isinstance(model, torch.nn.Module):
            self.model_instance = TorchSegmentationModel(model)
        elif isinstance(model, str):
            self.model_instance = ONNXSegmentationModel(model)
        else:
            raise ValueError("Unsupported model type. Provide a PyTorch model or path to ONNX model.")
        
    def __call__(self, image):
        """
        Run inference on an image using the selected model instance.
        
        Args:
            image (str, PIL.Image.Image, or torch.Tensor): The image input in various formats.
        
        Returns:
            numpy.ndarray: The segmentation map.
        """
        # Preprocessing the image
        preprocessed_image = self.model_instance.preprocess_image(image)

        # Run inference
        return self.model_instance.run_inference(preprocessed_image)
