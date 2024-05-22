from chatbot.chatbot.image_bot import handle_image_query
from chatbot.chatbot.text_bot import initialize_chain, handle_query
from chatbot.config import MODEL_DIR, TRAIN_DIR, VALIDATION_DIR
from chatbot.model.train_model import train_model
import logging
import os
from pydantic import BaseModel


class Description(BaseModel):
    description: str


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chatbot_response(description: Description):
    chain = initialize_chain()
    if not chain:
        logger.error("Failed to initialize the text query chain.")
        return

    return handle_query(description.description, chain)


def chatbot_image_response(description: Description):
    chain = initialize_chain()

    if not chain:
        logger.error("Failed to initialize the text query chain.")
        return

    predicted_class = handle_image_query(description.description)

    if not predicted_class:
        return "Failed to predict image."

    return handle_query(predicted_class, chain)


def main():
    # Ensure model is trained before running queries
    model_path = os.path.join(MODEL_DIR, 'weapon_classifier_model.keras')
    if not os.path.exists(model_path):
        logger.info("Model not found. Training the model...")
        train_model()

    # Initialize the text query chain
    chain = initialize_chain()
    if not chain:
        logger.error("Failed to initialize the text query chain.")
        return

    # Example usage for text query
    description = "What weapon has the highest damage in CS:GO?"
    print("Text Query Result:", handle_query(description, chain))

    # Example usage for image query
    image_url = "https://files.bo3.gg/uploads/news/13783/title_image/960x480-78d1e1c4be23bd914939cd7963ca56e4.webp"
    predicted_class = handle_image_query(image_url)
    print("Image Query Predicted Class:", predicted_class)

    if predicted_class:
        print("Image Query Result:", handle_query(predicted_class, chain))


if __name__ == "__main__":
    main()
