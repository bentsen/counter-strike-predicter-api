import locale
import os
import sys
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

from src.bots.image_response_bot import ImageResponseBot
from src.bots.text_response_bot import TextResponseBot
from src.predictors.csgo_round_predictor import CSGORoundPredictor, process_data
from src.types.round_type import GameData
from src.types.chatbot_type import Chatbot
from config.settings import MODEL_DIR as MODEL_DIR_PREDICTOR, MODEL_DIR as MODEL_DIR_CHATBOT
from src.models.cnn.cnn_trainer import train_model

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

# Initialize bots
image_bot = ImageResponseBot()
text_bot = TextResponseBot()
round_predictor = CSGORoundPredictor()

@app.post("/predict/chat")
async def predict_image(response: Chatbot):
    if response.image and response.description:
        predicted_class = image_bot.generate_response(response.image)
        return text_bot.generate_response(response.description + " " + predicted_class)
    elif response.image and not response.description:
        return image_bot.generate_response(response.image)
    elif not response.image and response.description:
        return text_bot.generate_response(response.description)


@app.post("/predict/round")
async def predict_round(game_data: GameData):
    try:
        processed_data = process_data(game_data)
        result = round_predictor.predict(processed_data)
        if result is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        return JSONResponse(content={"response": result})
    except Exception as e:
        logging.error(f"Error predicting round outcome: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
