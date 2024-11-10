# Segmentation Model Project

This project is designed for model conversion, preprocessing, and segmentation using various machine learning tools. It is structured into three phases:

- **Phase 1**: Model conversion from PyTorch to ONNX format.
- **Phase 2**: Contains utility functions and classes for segmentation models but does not require standalone execution.
- **Phase 3**: API-based inference and client interactions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Phase Instructions](#phase-instructions)
   - [Phase 1 - Convert.py](#phase-1---convertpy)
   - [Phase 2 - Utilities](#phase-2---utilities)
   - [Phase 3 - API and Client](#phase-3---api-and-client)
4. [Requirements](#requirements)

## Project Overview

This project provides tools for converting a PyTorch model to ONNX format, running inferences using the converted model, and serving predictions through an API. The folder structure is organized as follows:

- **Phase 1**: Contains `convert.py` for model conversion.
- **Phase 2**: Contains utility modules for segmentation model processing.
- **Phase 3**: Contains `api.py` for serving model predictions via an API and `client.py` for interacting with the API.
- **model utils**: Contains functions used in few phases.

## Setup Instructions

1. **Clone the repository**:

```bash
    git clone https://github.com/idoziv15/SegmentationModelProject.git
    cd SegmentationModelProject
```

2. **Create a virtual environment (optional but recommended)**:

On Linux/MacOS:
```
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```
python3 -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```
pip install -r requirements.txt
```

## Phase Instructions
### Phase 1 - Convert.py
The convert.py script converts a PyTorch model to ONNX format and runs inferences on both PyTorch and ONNX models to validate the conversion.

Usage:
```
python phase_1/convert.py --model_name <model_name> --image_path <path_to_image>

Or

cd phase_1
python convert.py --model_name <model_name> --image_path <path_to_image>
```
model_name (Optional): The name of the PyTorch model to convert (default: "deeplabv3_mobilenet_v3_large").

image_path (Optional): Path to the input image (default: "cat2.jpg" in the images directory).

### Phase 2 - Utilities
Phase 2 provides utility functions and classes for segmentation processing. This phase does not contain any standalone scripts to run directly.

### Phase 3 - API and Client
Phase 3 contains two main scripts: api.py and client.py. You will need to first run the API (api.py) and then you will be able to run client.py.

#### Running api.py:
The api.py script serves model predictions through an API endpoint. It requires model and image input as arguments.
The API will run by default on localhost:8000.
You should see at http://127.0.0.1:8000/docs a FastAPI swagger interface, to make sure the API is properly running.

Usage:
```
python phase_3/api.py --model_name <model_name> --image_path <path_to_image> --use_onnx

Or 

cd phase_3
python api.py --model_name <model_name> --image_path <path_to_image> --use_onnx
```

model_name (Optional): The name of the model to use (default: "deeplabv3_mobilenet_v3").

image_path (Optional): Path to the image to process (default: "cat2.jpg").
use_onnx: Flag to indicate if the ONNX model should be used (default is False).

#### Running client.py:
The client.py script interacts with the API, sending requests for model inference. Make sure api.py is running on the specified host and port before executing client.py.

Usage:
```
python phase_3/client.py --model_name <model_name> --image_path <path_to_image> --use_onnx
```

Replace <model_name> (Optional) and <path_to_image> (Optional) with appropriate values of your own, if desired.

