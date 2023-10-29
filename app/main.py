from fastapi import FastAPI, File, UploadFile
import uvicorn 
from starlette.responses import RedirectResponse
from typing import Annotated
import cv2
import uuid
from prediction import ImageToWordModel
from configs import BaseModelConfigs
import io
from PIL import Image
import numpy as np
from huggingface_hub import from_pretrained_keras
import onnxruntime as rt
import aiofiles
from mltu.utils.text_utils import ctc_decoder, get_cer

# model = from_pretrained_keras("keras-io/ocr-for-captcha")


app_desc = 'This is demo images captcha recognition '

app = FastAPI(title='Cpatcha Projects', description=app_desc)

configs = BaseModelConfigs.load("/home/tuandinh/Desktop/Captra_projects/Models/02_captcha_to_text/202310071259/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
char_list = configs.vocab


def read_image(content: bytes) -> np.ndarray:
    """
    Image bytes to OpenCV image

    :param content: Image bytes
    :returns OpenCV image
    :raises TypeError: If content is not bytes
    :raises ValueError: If content does not represent an image
    """
    if not isinstance(content, bytes):
        raise TypeError(f"Expected 'content' to be bytes, received: {type(content)}")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Expected 'content' to be image bytes")
    return image


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


data = []
@app.post("/predict/image")
async def prediction_route(file: UploadFile = File(...)):

    content = await file.read()
    image = read_image(content=content)
    image_np = np.array(image)
    image_np = cv2.resize(image_np, model.input_shape[:2][::-1])
    image_np = np.expand_dims(image, axis=0).astype(np.float32)
    text = model.predict(image=image_np)
    return text

@app.post("/show_data/")
async def show_data():
    return data

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0", reload=True)
