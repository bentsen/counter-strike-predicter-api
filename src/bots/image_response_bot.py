import base64
import os
import json
import logging
import numpy as np
import tensorflow as tf

from PIL import Image
from io import BytesIO
from config.settings import MODEL_DIR, DATA_DIR
from data.base64_iamge import base64_image_string
from src.models.cnn.cnn_trainer import train_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_image(image, target_size=(150, 150)):
    try:
        logger.info("Starting preprocessing image.")

        if image.mode != "RGB":
            logger.info(f"Image mode is {image.mode}, converting to RGB.")
            image = image.convert("RGB")

        logger.info(f"Resizing image to {target_size}.")
        image = image.resize(target_size)

        logger.info("Converting image to numpy array and normalizing.")
        image = np.array(image) / 255.0

        logger.info("Expanding dimensions of the image array.")
        image = np.expand_dims(image, axis=0)

        logger.info("Finished preprocessing image.")
        return image
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise


def load_class_indices(class_indices_path):
    with open(class_indices_path, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}
    return class_labels


class ImageResponseBot:
    def __init__(self):
        model_path = os.path.join(MODEL_DIR, 'cnn', 'cnn_model.keras')
        class_indices_path = os.path.join(DATA_DIR, 'class_indices.json')

        if not os.path.exists(model_path):
            logger.info("Model file not found. Training a new model...")
            train_model()

        if not os.path.exists(class_indices_path):
            logger.error(f"Class indices file not found at {class_indices_path}")
            raise FileNotFoundError(f"Class indices file not found at {class_indices_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = load_class_indices(class_indices_path)
        logger.info("Model and class indices loaded successfully.")

    def generate_response(self, image_base64):
        try:
            logger.info("Decoding base64 image data.")
            image_data = image_base64.split(",")[1] if "," in image_base64 else image_base64
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = preprocess_image(image)
            prediction = self.model.predict(image)
            response = self.decode_prediction(prediction)
            return response
        except Exception as e:
            logger.error(f"Error during image processing: {e}")
            return None

    def decode_prediction(self, prediction):
        predicted_class_index = np.argmax(prediction)
        predicted_class = self.class_labels.get(predicted_class_index, "Unknown")
        return predicted_class


if __name__ == "__main__":
    bot = ImageResponseBot()
    test_base64_input = base64_image_string
    result = bot.generate_response(test_base64_input)
    print(f"Generated response: {result}")