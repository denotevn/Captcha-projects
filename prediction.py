import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from configs import BaseModelConfigs
from PIL import Image
from io import BytesIO


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        # image = cv2.resize(image, self.input_shape[:2][::-1])

        # image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

# def read_image_file(image_encoded):
#     image = cv2.imread(image_encoded)
#     return image
#     # pil_image = Image.open(BytesIO(image_encoded))
#     # return pil_image

# def preprocessing(image):
#     image = cv2.resize(image, (200, 50))
#     image_pred = np.expand_dims(image, axis=0).astype(np.float32)
#     return image_pred

# def load_model():
#     configs = BaseModelConfigs.load("/home/tuandinh/Desktop/Captra_projects/Models/02_captcha_to_text/202310071259/configs.yaml")
#     model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
#     return model

# def predict_image(image: np.ndarray):
#     global model
#     if model is None:
#         model= load_model()
#     prediction_text = model.predict(image)
#     return prediction_text