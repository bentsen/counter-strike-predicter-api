import locale
import os
import sys

from fastapi import FastAPI
from dotenv import load_dotenv

from chatbot.main import chatbot_response, Description, chatbot_image_response
from chatbot.config import MODEL_DIR as MODEL_DIR_CHATBOT
from chatbot.model.train_model import train_model
from predictor.config import MODEL_DIR as MODEL_DIR_PREDICTOR
from predictor.csgo_round_ai import GameData, predict_round, process_data, train_and_save_model
from starlette.middleware.cors import CORSMiddleware


# Set UTF-8 encoding explicitly
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    # Check if the model file exists, train and save if it doesn't
    predict_round_model_path = os.path.join(MODEL_DIR_PREDICTOR, 'rf_model.pkl')
    chatbot_model_path = os.path.join(MODEL_DIR_CHATBOT, 'weapon_classifier_model.keras')

    if not os.path.exists(predict_round_model_path):
        train_and_save_model()
        print("Predict Model trained and saved.")
    else:
        print("Predict Model already trained.")

    if not os.path.exists(chatbot_model_path):
        train_model()
        print("Chatbot Model trained and saved.")
    else:
        print("Chatbot Model already trained.")


@app.post("/round")
async def receive_data(game_data: GameData):
    processed_data = process_data(game_data)
    return predict_round(processed_data)


@app.post("/chat/image")
async def receive_data(description: Description):
    return chatbot_image_response(description)


@app.post("/chat")
async def receive_data(description: Description):
    return chatbot_response(description)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
