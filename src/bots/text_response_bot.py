import logging
from src.models.nlp.nlp_trainer import initialize_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextResponseBot:
    def __init__(self):
        self.chain = initialize_chain()
        if not self.chain:
            logger.error("Failed to initialize the response chain.")
            raise Exception("Failed to initialize the response chain.")

    def generate_response(self, text_input):
        response = self.chain({"query": text_input})
        return response["result"]

if __name__ == "__main__":
    bot = TextResponseBot()
    user_input = "give me all the skins for the awp"
    response = bot.generate_response(user_input)
    print(response)
