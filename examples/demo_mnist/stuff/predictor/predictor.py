import numpy as np
import io
import torch
import mlflow
from PIL import Image
from pydantic import BaseModel, Field
from opyrator.components.types import FileContent

def preprocess(x_inference):
    x_inference = x_inference.reshape(-1, 28, 28)
    x_inference_tensor = torch.Tensor(x_inference)

    return  x_inference_tensor

def predict(model, x_inference):
    x_inference = preprocess(x_inference)
    logits = model(x_inference)
    preds = torch.argmax(logits, dim=1)
    predictions = preds.cpu().detach().numpy()

    return predictions[0]

def get_model(model_name, model_version, model_stage=None):
    if model_stage:
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
        print(f"Loading model: {model_name}:{model_stage}")
    else:
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        print(f"Loading model: {model_name}:{model_version}")

    return model

class Input(BaseModel):
    image_file: FileContent = Field(..., mime_type="image/png")

class Output(BaseModel):
    message: str

def main(input: Input) -> Output:
    """Returns the `prediction` of the input image."""
    x_test = np.array(Image.open(io.BytesIO(input.image_file.as_bytes())))

    model = get_model("mnist_cnn", "1")
    # model = get_model("mnist_cnn", "Production")
    prediction = predict(model, x_test)

    return Output(message=str(prediction))
