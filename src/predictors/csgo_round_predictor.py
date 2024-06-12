import os
import pandas as pd
import logging

from src.predictors.graph_functions import generate_graphs
from src.predictors.train_model import train_and_save_model, load_model, process_data
from src.types.round_type import GameData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSGORoundPredictor:
    def __init__(self):
        model_path = os.path.join('src', 'predictors', 'rf_model.pkl')
        if not os.path.exists(model_path):
            logger.info("Model file not found. Training a new model...")
            train_and_save_model()

        self.model = load_model()
        logger.info("Model loaded successfully.")

        graph_path = os.path.join('src', 'graphs', 'round_predictor')
        if not os.path.exists(graph_path):
            logger.info("Graph file not found. Creating a new graph...")
            os.makedirs(graph_path)
            generate_graphs()

    def predict_round(self, game_data: GameData):
        try:
            processed_data = process_data(game_data)
            input_df = pd.DataFrame(processed_data, index=[0])
            rd_probabilities_ct, rd_probabilities_t = self.model.predict_proba(input_df)[0]
            return {
                "ct": rd_probabilities_ct * 100,
                "t": rd_probabilities_t * 100
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
