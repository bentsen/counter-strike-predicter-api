import json
import os
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import logging
from chatbot.config import MODEL_DIR, RESOURCES_DIR

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


def handle_image_query(image_url):
    logger.info("Starting image query processing.")
    # Load the trained model and class labels
    model_path = os.path.join(MODEL_DIR, 'weapon_classifier_model.keras')
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return "Model not found."

    logger.info(f"Loading model from: {model_path}")
    print(model_path)
    model = tf.keras.models.load_model(model_path)

    class_indices_path = os.path.join(RESOURCES_DIR, 'class_indices.json')
    if not os.path.exists(class_indices_path):
        logger.error(f"Class indices path does not exist: {class_indices_path}")
        return "Class indices not found."

    logger.info(f"Loading class indices from: {class_indices_path}")
    with open(class_indices_path, 'r', encoding='utf8') as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        logger.info(f"Fetching image from URL: {image_url}")
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        if img.format == 'WEBP':
            img = img.convert("RGB")
        img_preprocessed = preprocess_image(img)

        logger.info("Making predictions.")
        predictions = model.predict(img_preprocessed)
        predicted_class = class_labels[np.argmax(predictions)]
        logger.info(f"Predicted class: {predicted_class}")
        return predicted_class

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        return "Failed to download image."
    except Exception as e:
        logger.error(f"An error occurred while processing the image: {e}")
        return "An error occurred while processing the image."
