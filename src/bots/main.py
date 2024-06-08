import logging
import os
from pydantic import BaseModel

from src.bots.image_response_bot import ImageResponseBot
from src.bots.text_response_bot import TextResponseBot
from config.settings import MODEL_DIR
from src.models.cnn.cnn_trainer import train_model


class Description(BaseModel):
    description: str


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chatbot_response(description: Description, text_bot: TextResponseBot):
    return text_bot.generate_response(description.description)


def chatbot_image_response(description: Description, image_bot: ImageResponseBot, text_bot: TextResponseBot):
    predicted_class = image_bot.generate_response(description.description)
    if not predicted_class:
        return "Failed to predict image."
    return text_bot.generate_response(predicted_class)


def main():
    # Ensure model is trained before running queries
    model_path = os.path.join(MODEL_DIR, 'cnn', 'cnn_model.keras')
    if not os.path.exists(model_path):
        logger.info("Model not found. Training the model...")
        train_model()

    # Initialize bots
    image_bot = ImageResponseBot()
    text_bot = TextResponseBot()

    # Example usage for text query
    description = Description(description="What weapon has the highest damage in CS:GO?")
    text_query_result = chatbot_response(description, text_bot)
    print("Text Query Result:", text_query_result)

    # Example usage for image query
    image_description = Description(
        description="https://files.bo3.gg/uploads/news/13783/title_image/960x480-78d1e1c4be23bd914939cd7963ca56e4.webp")
    predicted_class = image_bot.generate_response(image_description.description)
    print("Image Query Predicted Class:", predicted_class)

    if predicted_class:
        image_query_result = text_bot.generate_response(predicted_class)
        print("Image Query Result:", image_query_result)


if __name__ == "__main__":
    main()
